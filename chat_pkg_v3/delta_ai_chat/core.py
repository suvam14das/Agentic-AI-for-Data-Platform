from datetime import datetime
import json
import os
import sys
import uuid
import pandas as pd
import warnings
import oci
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_classic.memory import ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_oci.embeddings import OCIGenAIEmbeddings
import re
from langchain_community.vectorstores import FAISS
from langchain_classic.tools import StructuredTool
from pydantic import BaseModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try :
    from delta_ai_chat.conn_dataflow import DataflowConnector
    from delta_ai_chat.generate_vector_store import generate_vector_store
except ImportError:
    from conn_dataflow import DataflowConnector
    from generate_vector_store import generate_vector_store


# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import MessagesState

warnings.filterwarnings("ignore")

COLOR_BLUE = "\033[94m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"

system_prompt = """
History : {chat_history}

Rules :
1. Respond concisely with only relevant details and still be polite and helpful. Use Markdown formatting for better readability, such as bullets for lists, tables for data, bold/italics for emphasis, and proper paragraphs with line breaks. If the user is asking a general Compute domain question first look for in the documentations. 
2. If the user is asking a Compute domain question that requires data from the DeltaLake DB then use the run_sql tool with appropriate SQL after confirming the query looks good from the user and getting user affirmation.
3. Do not use tables that are not present in the database. Verify that columns are present for a given table from the retrived knowledge before using it in query.  
4. Columns must be consistent to the table schema queried. Do not wrap the entire SQL in backticks. ALWAYS wrap column names that contains $ with single backticks. Always use full name of the column along with proper table alias in the SQL. Try to find the relevant columns within the same table to build the query. 
5. Always use tables in SQL query in the format <database>.<table> e.g. cdi.hosts, cdi.instances etc. 
6. For relevant SQLs that supports limit if the limit of rows is not specified or evident use LIMIT 10. 
7. For general questions, provide a polite direct and relevant response and if the answer is not known just say "Sorry I did not get you. My AI is not AIing!".
8. If the user affirms a previous proposal, proceed with the action in the next response.
9. If unable to resolve within 5 attempts, seek human input by asking for clarification in your response.

-------------------------
TOOL SELECTION POLICY (STRICT)
-------------------------

You MUST follow these rules when selecting tools:

1. Schema / Metadata Questions → ALWAYS use `retrieval`
   This includes:
   - Listing tables
   - Listing columns
   - Table schema or structure
   - Column descriptions
   - Understanding what a table or column represents

   DO NOT use run_sql for these.

2. Data Query Questions → use `run_sql`
   This includes:
   - Fetching actual data
   - Aggregations, counts, filters
   - Any query that requires table rows

3. For any action requiring investigation (e.g.,retrieval or run_sql), first propose the action and seek user confirmation in your response (without tool call). Do not execute without affirmation in the history. Example: "Proposed SQL: SELECT * FROM cdi.hosts LIMIT 5. Confirm to proceed?" If the last human message affirms, then call the tool.

4. If unsure whether the question is schema-related or data-related:
   → ALWAYS use `retrieval` first.

5. You MUST call `retrieval` BEFORE `run_sql` for any database-related question.

6. NEVER generate SQL for:
   - "list tables"
   - "show columns"
   - "describe table"
   - "what columns exist"

7. Use 'visualize' to generate charts from data, usually after getting JSON from run_sql and if the user requests a chart or visualization. Call with input_str as a JSON string: {{"data": [list of dicts from run_sql rows], "chart_type": "bar" or "line" or "pie", "x": "label_column", "y": "value_column"}}. Choose appropriate chart_type, x, y based on data or what is asked by user.

8. Use 'format_to_html' to convert JSON from run_sql to HTML table when you need to display tabular data, usually after getting JSON from run_sql and if no visualization is requested. Pass the exact JSON string from run_sql.

Decision Logic:
- For every new user question or refinement: Always start by calling 'retrieval' to verify relevant schema, metadata, or documentation. Use the retrieved information to inform subsequent actions, such as building SQL queries or deciding on visualization.
- If proposal needed for retrieval/run_sql and no confirmation in history: Respond with proposal message (no tool call).
- If confirmation received: Call the appropriate tool.
- For direct answers (non-database): Respond with message (no tool call).
- After receiving ToolMessage from run_sql with JSON data:
  - If user query involves charting/visualization: Call 'visualize' with constructed input_str. ALWAYS generate the full chart config JSON in the tool and return it.
  - Otherwise: Call 'format_to_html' with the JSON string.
  - If visualization fails (e.g., error in ToolMessage): Respond with error explanation.
- Take care of possible errors in visualization like: All arrays must be of the same length - choose suitable x/y or adjust data.
"""


class RetrievalInput(BaseModel):
    query: str

class RunSQLInput(BaseModel):
    sql: str

class FormatToHTMLInput(BaseModel):
    json_str: str

class VisualizeInput(BaseModel):
    input_str: str

class DeltaAIChat:

    def __init__(self, profile_name='bmc-sie-prod', summary_file="delta_ai_chat/general_docs/chat_history_summary.txt"):

        self.oc1_delta_conn = DataflowConnector(profile_name)
        self.summary_file = summary_file
        self.auth_profile = profile_name

        self.llm = ChatOCIGenAI(
            # model_id="ocid1.generativeaimodel.oc1.iad.amaaaaaask7dceyaeo4ehrn25guuats5s45hnvswlhxo6riop275l2bkr2vq", #gemini flash
            model_id="ocid1.generativeaimodel.oc1....", #gemini pro
            service_endpoint="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com",
            compartment_id="ocid1.tenancy....",
            auth_type="SECURITY_TOKEN",
            auth_profile=self.auth_profile,
            provider="generic",
            model_kwargs={"temperature": 0,"top_k": 1, "top_p": 0.1}
        )

        self.embeddings = OCIGenAIEmbeddings(
            model_id="cohere.embed-english-v3.0",
            service_endpoint="https://inference.generativeai.uk-london-1.oci.oraclecloud.com",
            compartment_id="ocid1.compartment....",
            model_kwargs={"truncate": True},
            auth_type="SECURITY_TOKEN",
            auth_profile=profile_name,
        )

        self.vectorstore = FAISS.load_local(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore"), embeddings=self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="input", return_messages=True, k=5)

        # Define tools as StructuredTool
        retrieval_tool = StructuredTool.from_function(
            func=lambda query: "\n\n".join([d.page_content for d in self.retriever.invoke(query)]),
            name="retrieval",
            description="Retrieve information from documentation for general compute domain questions.",
            args_schema=RetrievalInput
        )

        run_sql_tool = StructuredTool.from_function(
            func=lambda sql: self.execute_query(sql, "")[0],
            name="run_sql",
            description="Execute a Spark SQL query on the Delta Lake database and return JSON string for data or formatted error message.",
            args_schema=RunSQLInput
        )

        format_to_html_tool = StructuredTool.from_function(
            func=lambda json_str: self.json_to_html(json.loads(json_str)),
            name="format_to_html",
            description="Convert a JSON string (with 'columns' and 'rows') to an HTML table string for display.",
            args_schema=FormatToHTMLInput
        )

        def create_chart(input_str):
            try:
                input_data = json.loads(input_str)
                data = input_data['data']
                chart_type = input_data.get('chart_type', 'bar')
                x = input_data.get('x')
                y = input_data.get('y')

                df = pd.DataFrame(data)
                labels = df[x].tolist()
                values = df[y].tolist()

                chart_config = {
                    "type": chart_type,
                    "data": {
                        "labels": labels,
                        "datasets": [{
                            "label": y,
                            "data": values,
                            "backgroundColor": "rgba(75, 192, 192, 0.2)",
                            "borderColor": "rgba(75, 192, 192, 1)",
                            "borderWidth": 1
                        }]
                    },
                    "options": {
                        "scales": {
                            "y": {
                                "beginAtZero": True
                            }
                        },
                        "plugins": {
                            "legend": {
                                "display": True
                            }
                        },
                        "responsive": True,
                        "maintainAspectRatio": False
                    }
                }

                if chart_type == 'pie':
                    chart_config["data"]["datasets"][0].pop("borderColor", None)
                    chart_config["data"]["datasets"][0].pop("borderWidth", None)
                    chart_config["data"]["datasets"][0]["backgroundColor"] = [
                        "rgba(255, 99, 132, 0.2)",
                        "rgba(54, 162, 235, 0.2)",
                        "rgba(255, 206, 86, 0.2)",
                        "rgba(75, 192, 192, 0.2)",
                        "rgba(153, 102, 255, 0.2)"
                    ]  # Example colors

                return json.dumps(chart_config)

            except Exception as e:
                return str(e)

        visualize_tool = StructuredTool.from_function(
            func=create_chart,
            name="visualize",
            description="Generate Chart.js configuration JSON from JSON data string (with 'data' as list of dicts, 'chart_type' (bar, line, pie), 'x', 'y').",
            args_schema=VisualizeInput
        )

        self.tools = [retrieval_tool, run_sql_tool, format_to_html_tool, visualize_tool]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_dict = {t.name: t for t in self.tools}

        # Define nodes and graph
        def agent_node(state: MessagesState):
            print("\nEntering agent_node")
            print("Current state messages:", [msg.content for msg in state["messages"]])
            messages = [SystemMessage(content=system_prompt.format(chat_history=state["messages"]))] + state["messages"]
            try:
                response = self.llm_with_tools.invoke(messages)
                print("Agent response:", response.content)
                if response.tool_calls:
                    print("Tool calls:", response.tool_calls)
            except Exception as e:
                error_str = str(e)
                if '401' in error_str:
                    print(f"{COLOR_RED}401 error detected in agent. Reconnecting...{COLOR_RESET}")
                    self.oc1_delta_conn.connect()
                    self.llm = self.create_llm()
                    self.llm_with_tools = self.llm.bind_tools(self.tools)
                    response = self.llm_with_tools.invoke(messages)
                    print("Agent response after reconnect:", response.content)
                    if response.tool_calls:
                        print("Tool calls after reconnect:", response.tool_calls)
                else:
                    raise e
            print("Exiting agent_node\n")
            return {"messages": [response]}

        def tool_node(state: MessagesState):
            print("\nEntering tool_node")
            last_message = state["messages"][-1]
            print("Last message (AIMessage):", last_message.content)
            print("Tool calls to execute:", last_message.tool_calls)
            tool_results = []
            for tool_call in last_message.tool_calls:
                tool = self.tool_dict[tool_call["name"]]
                print(f"Executing tool: {tool_call['name']} with args: {tool_call['args']}")
                try:
                    result = tool.run(tool_call["args"])
                    print(f"Tool result: {result}")
                except Exception as e:
                    result = f"Error executing {tool_call['name']}: {str(e)}"
                    print(f"{COLOR_RED}{result}{COLOR_RESET}")
                tool_results.append(
                    ToolMessage(
                        content=str(result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            print("Exiting tool_node\n")
            return {"messages": tool_results}

        def should_continue(state: MessagesState):
            print("\nEntering should_continue")
            last_message = state["messages"][-1]
            print("Last message type:", type(last_message).__name__)
            if isinstance(last_message, AIMessage):
                if last_message.tool_calls:
                    print("Routing to 'tools' (has tool_calls)")
                    return "tools"
                print("Routing to END (no tool_calls)")
                return END
            if isinstance(last_message, ToolMessage):
                if last_message.name in ["format_to_html", "visualize"]:
                    print(f"Routing to END (terminal tool: {last_message.name})")
                    return END
                print(f"Routing to 'agent' (non-terminal tool: {last_message.name})")
                return "agent"
            print("Default routing to END")
            return END

        builder = StateGraph(MessagesState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tool_node)
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        builder.add_conditional_edges("tools", should_continue, {"agent": "agent", END: END})
        self.graph = builder.compile()

    def get_agent_response(self, user_query):
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        state = self.graph.invoke({"messages": chat_history + [HumanMessage(content=user_query)]})
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            response_text = last_msg.content
        elif isinstance(last_msg, ToolMessage):
            if last_msg.name == "visualize":
                response_text = f'<chart-data>{last_msg.content}</chart-data>'
            elif last_msg.name == "format_to_html":
                response_text = last_msg.content
            else:
                response_text = str(last_msg.content)
        else:
            response_text = "Unexpected state."
        self.memory.save_context({"input": user_query}, {"output": response_text})
        print(f"{COLOR_YELLOW}Agent: {response_text}{COLOR_RESET}")
        return response_text, None

    def format_results_with_agent(self, raw_data, is_error=False):
        
        if is_error:
            refinement_prompt = (
                f"The following error message was returned from a database query:\n {raw_data}\n"
                "Format this into a concise, user-friendly message using Markdown for better readability. If the error contains technical details, extract the key issue and present it in a way that a non-technical user can understand. Do not include stack traces or overly technical jargon. Focus on the main problem and potential next steps for resolution."
            )
            result = self.llm.invoke(refinement_prompt)  # Changed from qa_chain to llm
            return  result.content.strip()
        else:
            refinement_prompt = (
                f"""
                
                Raw DataFrame: \n {raw_data}\n
                
                Data formatting : If data header or rows values is in tuples, convert to strings by joining each character in the tuple. Use '' as joining delimiter. For example, (a,b,c,1,.,1) should be converted to 'abc1.1' , (c, o, u, n, t, (, D, I, S, T, I, N, C, T,  , i, d, )) should be converted to 'count(DISTINCT id)' , and similar. Include all characters including numbers and special characters without spaces in between. Do not remove any character from the raw data. 
                Output format : Return in JSON format only. Do not generate HTML tables. Do not generate additionals comments. The JSON should have two keys: "columns" which is a list of column names, and "rows" which is a list of lists, where each inner list represents a row of data corresponding to the columns. For example: {{'columns': ['col_name', 'data_type', 'comment'], 'rows': [['KievTxnID', 'bigint', None], ['hostsIngested', 'string', None], ['hpcIslandId', 'string', None], ['id', 'string', None], ['multiFaultDomain', 'string', None], ['networkBlockId', 'string', None]]}}
                """
            )
            result = self.llm.invoke(refinement_prompt)
            print(f"{COLOR_YELLOW}Formatting output: {result.content.strip()}{COLOR_RESET}")
            data = json.loads(result.content.strip().replace("\n", "").replace("```json", "").replace("```", ""))
            # print(f"{COLOR_YELLOW}Refined Result: {data}{COLOR_RESET}")
            return  str(json.dumps(data))

    def execute_query(self, sql_query, user_query):
            try:
                self.oc1_delta_conn.check_connection()
                db_data = self.oc1_delta_conn.pull_data(sql_query)
                formatted_response = self.format_results_with_agent(db_data)
                print(f"{COLOR_YELLOW}{formatted_response}{COLOR_RESET}")
                return formatted_response, None
            
            except Exception as e:
                error_str = str(e)
                if '401' in error_str:
                    print(f"{COLOR_RED}401 error detected. Reconnecting...{COLOR_RESET}")
                    self.oc1_delta_conn.connect()
                    print(f"{COLOR_RED}Reconnected. Retrying query...{COLOR_RESET}")
                    # Retry the query once after reconnect
                    try:
                        db_data = self.oc1_delta_conn.pull_data(sql_query)
                        formatted_response = self.format_results_with_agent(db_data)
                        print(f"{COLOR_YELLOW}{formatted_response}{COLOR_RESET}")
                        return formatted_response, None
                    except Exception as retry_e:
                        print(f"{COLOR_RED}Retry failed: {str(retry_e)[:256]}{COLOR_RESET}")
                        formatted_response = self.format_results_with_agent(str(retry_e), is_error=True)
                        return formatted_response, None
                else:
                    print(f"{COLOR_RED}Error : {error_str[:256]}{COLOR_RESET}")
                    formatted_response = self.format_results_with_agent(error_str, is_error=True)
                    return formatted_response, None

                
    def json_to_html(self, data):
        columns = data["columns"]
        rows = data["rows"]

        style = """
        <style>
        table {
            border-collapse: collapse;
            width: auto;
            font-family: Calibri, sans-serif;
            font-size: 11pt;
        }

        th, td {
            border: 1px solid #D3D3D3;
            padding: 8px;
        }

        th {
            background-color: #DCE6F1;
            font-weight: bold;
            text-align: center;
            border-bottom: 2px solid #D3D3D3;
        }

        tbody tr:nth-child(even) {
            background-color: #F9F9F9;
        }

        td.text {
            text-align: left;
        }

        td.numeric {
            text-align: right;
        }
        </style>
        """

        html = style
        html += "<table>"

        # Header
        html += "<thead><tr>"
        for col in columns:
            html += f"<th>{col}</th>"
        html += "</tr></thead>"

        # Body
        html += "<tbody>"
        for row in rows:
            html += "<tr>"
            for value in row:

                # detect numeric values
                if isinstance(value, (int, float)):
                    cell_class = "numeric"
                else:
                    # try converting numeric strings
                    try:
                        float(value)
                        cell_class = "numeric"
                    except:
                        cell_class = "text"

                html += f'<td class="{cell_class}">{value}</td>'

            html += "</tr>"
        html += "</tbody></table>"

        return  html

    def save_summary(self):
        if self.memory.buffer:
            self.summary_memory = ConversationSummaryMemory(llm=self.llm)
            messages = self.memory.chat_memory.messages
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    self.summary_memory.save_context(
                        {"input": messages[i].content},
                        {"output": messages[i+1].content}
                    )
            
            new_summary = f"""
                "timestamp": {datetime.now().isoformat()} \n"""
            
            summary_messages = self.summary_memory.load_memory_variables({})["history"]
            new_summary += f""" "summary" : "{summary_messages}" \n"""

            with open(self.summary_file, "a") as f:
                f.write("{\n" + new_summary + "\n}" + "\n\n")

            print(f"{COLOR_YELLOW}History saved to {self.summary_file}.{COLOR_RESET}")
            generate_vector_store()
        else:
            print(f"{COLOR_YELLOW}No history.{COLOR_RESET}")

    def close(self):
        print(f"Cleaning up resources...")
        self.oc1_delta_conn.check_connection()    
        self.oc1_delta_conn.close()
        sys.exit(0)

    # Used Only for terminal runs
    def process_input(self, user_input):
        if user_input.lower() == "memorize":
            self.save_summary()
            return
        
        if user_input.lower() == 'exit':
            self.close()
            return
        
        self.get_agent_response(user_input)
        # print(f"{COLOR_YELLOW}{response_text}{COLOR_RESET}")

# For testing as script
if __name__ == "__main__":
    chat = DeltaAIChat()
    print(f"{COLOR_YELLOW}Agent: Welcome to Delta AI Chat! How can I help you today?{COLOR_RESET}")
    while True:
        user_input = input(f"{COLOR_BLUE}You: {COLOR_RESET}").strip()
        chat.process_input(user_input)
