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

# NOTE (v4 packaging):
# - LangGraph is used ONLY inside this core (DeltaAIChat) as the orchestration engine.
# - Structured tools are defined in `tools_registry.py` and are exposed via the standalone MCP server
#   `tools_server.py` for external agents (Cline/others). Core can also use them locally via ToolsManager.
from pydantic import BaseModel
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
try :
    from delta_ai_chat.generate_vector_store import generate_vector_store
    from delta_ai_chat.tools_registry import ToolsManager
except ImportError:
    from generate_vector_store import generate_vector_store
    from tools_registry import ToolsManager


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

        self.summary_file = summary_file
        self.auth_profile = profile_name

        self.llm = ChatOCIGenAI(
            # model_id="ocid1.generativeaimodel.oc1.iad.amaaaaaask7dceyaeo4ehrn25guuats5s45hnvswlhxo6riop275l2bkr2vq", #gemini flash
            model_id="ocid1.generativeaimodel.oc1.iad.amaaaaaask7dceyargceyuaysrjzo2metq2rinavayxqmpu7tkm6mmfojcvq", #gemini pro
            service_endpoint="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com",
            compartment_id="ocid1.tenancy.oc1..aaaaaaaat3gxqmhzhjniz6udhx6ak6nngup2quzdahdztnhl7p4oznurigfq",
            auth_type="SECURITY_TOKEN",
            auth_profile=self.auth_profile,
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

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", input_key="input", return_messages=True, k=5
        )

        # Local tool registry (single source of truth)
        self.tools_manager = ToolsManager(profile_name=self.auth_profile)
        self.tool_specs =  self.tools_manager.build_tool_specs()

        self.tools = [
            StructuredTool.from_function(
                func=lambda _spec=spec, **kwargs: _spec.handler(_spec.input_model.model_validate(kwargs)),
                name=spec.name,
                description=spec.description,
                args_schema=spec.input_model,
            )
            for spec in self.tool_specs
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_dict = {t.name: t for t in self.tools}

        # Define nodes and graph
        def agent_node(state: MessagesState):
            print("\nEntering agent_node")
            print("Current state messages:", [msg.content for msg in state["messages"]])
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            try:
                response = self.llm_with_tools.invoke(messages)
                print("Agent response:", response.content)
                if response.tool_calls:
                    print("Tool calls:", response.tool_calls)
            except Exception as e:
                error_str = str(e)
                print(f"{COLOR_RED}Error in agent_node: {error_str}{COLOR_RESET}")
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
                        tool_call_id=tool_call["id"]
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
        self.tools_manager.cleanup()
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
