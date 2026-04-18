from datetime import datetime
import json
import os
import sys
import uuid
import pandas as pd
from delta_ai_chat.conn_dataflow import DataflowConnector
from delta_ai_chat.generate_vector_store import generate_vector_store
import warnings
import oci
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_classic.memory import ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_oci.embeddings import OCIGenAIEmbeddings
import re
from langchain_community.vectorstores import FAISS
from langchain_classic.agents import Tool, create_react_agent, AgentExecutor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64

warnings.filterwarnings("ignore")

COLOR_BLUE = "\033[94m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"

prompt_template = """
History : {chat_history}

Rules :
1. Respond concisely with only relevant details and still be polite and helpful. Use Markdown formatting for better readability, such as bullets for lists, tables for data, bold/italics for emphasis, and proper paragraphs with line breaks. If the user is asking a general Compute domain question first look for in the documentations. 
2. If the user is asking a Compute domain question that requires data from the DeltaLake DB then use the run_sql tool with appropriate SQL after confirming the query looks good from the user and getting user affirmation.
3. Do not use tables that are not present in the database. Verify that columns are present for a given table from the retrived knowledge before using it in query.  
4. Columns must be consistent to the table schema queried. Do not wrap the entire SQL in backticks. ALWAYS wrap column names that contains $ with single backticks. Always use full name of the column along with proper table alias in the SQL. Try to find the relevant columns within the same table to build the query. 
5. Always use tables in SQL query in the format <database>.<table> e.g. cdi.hosts, cdi.instances etc. 
6. For relevant SQLs that supports limit if the limit of rows is not specified or evident use 'LIMIT 10' always by default. 
7. For general questions, provide a polite direct and relevant response and if the answer is not known just say "Sorry I did not get you. My AI is not AIing!".
8. If the user affirms a previous proposal, proceed with the action in the next response.
9. If unable to resolve within 5 attempts, seek human input by asking for clarification in Final Answer.

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

3. For any action requiring investigation (e.g.,retrieval or run_sql), first propose the action and seek user confirmation in your Final Answer. Do not execute without affirmation. Example: "Proposed SQL: SELECT * FROM cdi.hosts LIMIT 5. Confirm to proceed?"

4. If unsure whether the question is schema-related or data-related:
   → ALWAYS use `retrieval` first.

5. You MUST call `retrieval` BEFORE `run_sql` for any database-related question.

6. NEVER generate SQL for:
   - "list tables"
   - "show columns"
   - "describe table"
   - "what columns exist"

7. Use 'visualize' to generate charts from data, usually after getting JSON from run_sql. If the observation from 'visualize' is a JSON string, include it in your Final Answer wrapped in <chart-data> tags: <chart-data>{{json}}</chart-data> along with any descriptive text. Take care of possible errors like :
    - All arrays must be of the same length

8. Use 'format_to_html' to convert JSON from run_sql to HTML table when you need to display data  usually after getting JSON from run_sql. When using format_to_html, pass the exact JSON output string from run_sql directly as the Action Input. Do not enclose it in another JSON object like {{"json_string": ...}}. The input should be the raw JSON string containing 'columns' and 'rows'.
"""


# 7. Use 'visualize' to generate charts from data, usually after getting JSON from run_sql. If the observation from 'visualize' starts with "Chart saved at:", extract the filename from the path (the part after the last /) and include in your Final Answer an HTML img tag: <img src="/charts/<filename>" alt="Generated Chart"> along with any descriptive text.

class DeltaAIChat:
    def __init__(self, profile_name='bmc-sie-prod', summary_file="delta_ai_chat/general_docs/chat_history_summary.txt"):
        self.oc1_delta_conn = DataflowConnector(profile_name)
        self.summary_file = summary_file

        self.llm = ChatOCIGenAI(
            model_id="ocid1.generativeaimodel.oc1.iad.amaaaaaask7dceyaeo4ehrn25guuats5s45hnvswlhxo6riop275l2bkr2vq", #gemini flash
            # model_id="ocid1.generativeaimodel.oc1.iad.amaaaaaask7dceyargceyuaysrjzo2metq2rinavayxqmpu7tkm6mmfojcvq", #gemini pro
            service_endpoint="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com",
            compartment_id="ocid1.tenancy.oc1..aaaaaaaat3gxqmhzhjniz6udhx6ak6nngup2quzdahdztnhl7p4oznurigfq",
            auth_type="SECURITY_TOKEN",
            auth_profile=profile_name,
            provider="generic",
            model_kwargs={"temperature": 0,"top_k": 1, "top_p": 0.1}
        )

        self.embeddings = OCIGenAIEmbeddings(
            model_id="cohere.embed-english-v3.0",
            service_endpoint="https://inference.generativeai.uk-london-1.oci.oraclecloud.com",
            compartment_id="ocid1.compartment.oc1..aaaaaaaaac64gw2jhiwemjswhxb5odbwpaktqxt5ublisya2uotjn7g6wxqa",
            model_kwargs={"truncate": True},
            auth_type="SECURITY_TOKEN",
            auth_profile=profile_name,
        )

        self.vectorstore = FAISS.load_local(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore"), embeddings=self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="input", return_messages=True, k=5)

        # Define tools
        retrieval_tool = Tool(
            name="retrieval",
            func=lambda q:  "\n\n".join([d.page_content for d in self.retriever.invoke(q)]),
            description="Retrieve information from documentation for general compute domain questions."
        )

        run_sql_tool = Tool(
            name="run_sql",
            func=lambda sql: self.execute_query(sql, "")[0],
            description="Execute a Spark SQL query on the Delta Lake database and return JSON string for data or formatted error message."
        )

        format_to_html_tool = Tool(
            name="format_to_html",
            func=lambda json_str: self.json_to_html(json.loads(json_str)),
            description="Convert a JSON string (with 'columns' and 'rows') to an HTML table string for display."
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

                return  f"<chart-data>{json.dumps(chart_config)}</chart-data>"

            except Exception as e:
                return str(e)

        visualize_tool = Tool(
            name="visualize",
            func=create_chart,
            description="Generate Chart.js configuration JSON from JSON data string (with 'data' as list of dicts, 'chart_type' (bar, line, pie), 'x', 'y')."
        )

        self.tools = [retrieval_tool, run_sql_tool, format_to_html_tool, visualize_tool]
        tool_names = [t.name for t in self.tools]
        tools_descriptions = "\n".join([f"{t.name}: {t.description}" for t in self.tools])

        # Agent prompt template
        react_template = prompt_template + """
            You have access to the following tools:

            {tools}

            -------------------------
            RESPONSE FORMAT (STRICT)
            -------------------------

            You MUST follow ONE of the formats below exactly.

            ### 1. When using a tool:
            Thought: <short internal reasoning>
            Action: <one of [{tool_names}]>
            Action Input: <input to the tool>
            Observation : <result from the tool>

            --- wait for Observation ---

            ### 2. After receiving an Observation:

            You MUST respond with ONE of the following:

            (a) Continue reasoning:
            Thought: <short internal reasoning>
            Action: <next tool>
            Action Input: <input>

            OR

            (b) Finish the task:
            Final Thought: I now know the final answer.
            Final Answer: <user-facing answer>

            -------------------------
            SPECIAL CASES (IMPORTANT)
            -------------------------

            ### Case: No tool required
            If you can answer directly:

            Final Thought: I now know the final answer.
            Final Answer: <answer>

            Then Do NOT use Action in this case.

            ---

            ### Case: Terminal tools (visualize, format_to_html)

            If you use `visualize` or `format_to_html`, they already produce final output. If the Observation after this tool contains table/HTML/chart-data, ALWAYS give Final Answer after using these tools, do NOT use Thought. But if Observation is a error message, then you can continue reasoning with Thought and use another tool if needed.

            After their Observation, respond ONLY with:

            Final Answer: <brief explanation + include the result>

            Do NOT generate another Thought step.
            Do NOT call another Action.

            ---

            ### Case: Observation already contains final result

            If Observation contains a complete result (table, HTML, chart-data, or final data):

            Final Thought: I now know the final answer.
            Final Answer: <answer>

            ---

            ### Case: Asking user confirmation

            Final Thought: I need user confirmation.
            Final Answer: <confirmation question>

            ---

            -------------------------
            STRICT RULES
            -------------------------

            1. Thought must be SHORT and INTERNAL only.
            Do NOT say: "Here is", "Of course", "I have generated"
            These belong ONLY in Final Answer

            2. Every response MUST end with either:
            - Action + Action Input
            OR
            - Final Answer

            3. NEVER end with Thought alone.

            4. After every Observation, you MUST produce either:
            - Thought → Action
            OR
            - Thought → Final Answer
            OR (for terminal tools e.g. visualize/format_to_html)
            - Final Answer directly

            5. "Action:" is ONLY used when calling a tool.
            It is NOT required after every Thought.

            6. It is VALID to go directly from:
            Thought → Final Answer

            7. Do NOT skip steps. Do NOT change format.

            8. If unsure, ask for clarification in Final Answer.

            -------------------------
            EXAMPLE
            -------------------------

            Question: What is 2 + 2?
            Thought: I need to calculate.
            Action: calculator
            Action Input: 2 + 2
            Observation: 4
            Final Thought: I now know the final answer.
            Final Answer: 4

            -------------------------

            Question: {input}

            {agent_scratchpad}
            """
        self.agent_prompt = PromptTemplate(template=react_template, input_variables=["chat_history", "input", "tools", "tool_names", "agent_scratchpad"])
        
        agent = create_react_agent(self.llm, self.tools, self.agent_prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )

        # Removed self.current_sql

    def get_agent_response(self, user_query):
        result = self.agent_executor.invoke({
            "input": user_query,
            "tools": "\n".join([f"{t.name}: {t.description}" for t in self.tools]),
            "tool_names": ", ".join([t.name for t in self.tools])
        })
        response_text = result["output"]
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
            # print(f"{COLOR_YELLOW}Formatting output: {result.content.strip()}{COLOR_RESET}")
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
                # TODO: Implement reconnect and retry with new session creation from POST service
                print(f"{COLOR_RED}401 error detected. Reconnecting...{COLOR_RESET}")
                self.oc1_delta_conn.connect()
                print(f"{COLOR_RED}Reconnected. Retry query...{COLOR_RESET}")
                return "401 Error. Reconnected.", None

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
        
        # Removed special "run sql" handling, let agent handle
        response_text, _ = self.get_agent_response(user_input)
        print(f"{COLOR_YELLOW}{response_text}{COLOR_RESET}")

# For testing as script
if __name__ == "__main__":
    chat = DeltaAIChat()
    print(f"{COLOR_YELLOW}Agent: Welcome to Delta AI Chat! How can I help you today?{COLOR_RESET}")
    while True:
        user_input = input(f"{COLOR_BLUE}You: {COLOR_RESET}").strip()
        chat.process_input(user_input)
