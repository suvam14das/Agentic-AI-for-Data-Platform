from ast import arguments
from datetime import datetime
import asyncio
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
# NOTE:
# - LangGraph is used inside this core (DeltaAIChat) as the orchestration engine.
# - Tools are bound directly from the local tools registry.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
try:
    from delta_ai_chat.generate_vector_store import generate_vector_store
except ImportError:
    from generate_vector_store import generate_vector_store

from langchain_mcp_adapters.client import MultiServerMCPClient


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

7. Use 'visualize' to generate charts from data only after a successful run_sql and only if the user requests a chart or visualization. run_sql returns an <artifact> payload containing the CSV artifact path. Use the latest available csv_path from that artifact and call visualize with csv_path, chart_type, x, and y. Choose appropriate chart_type, x, and y based on data or what is asked by the user.

Decision Logic:
- For every new user question or refinement: Always start by calling 'retrieval' to verify relevant schema, metadata, or documentation. Use the retrieved information to inform subsequent actions, such as building SQL queries or deciding on visualization.
- If proposal needed for retrieval/run_sql and no confirmation in history: Respond with proposal message (no tool call).
- If confirmation received: Call the appropriate tool.
- For direct answers (non-database): Respond with message (no tool call).
- After receiving ToolMessage from run_sql:
  - If status is error: fix query and retry.
  - If status is ok and type is table/csv: present the CSV artifact reference (frontend will render it in a floating panel). If the user later asks for a graph, use the csv_path from the latest <artifact> output as input to visualize.
- After receiving ToolMessage from visualize with chart config JSON: return it. visualize is a terminal tool and should be used with csv_path, chart_type, x, and y.
- Take care of possible errors in visualization like: All arrays must be of the same length - choose suitable x/y or adjust data.
"""






class DeltaAIChat:

    def __init__(self, profile_name='bmc-sie-prod', summary_file="delta_ai_chat/general_docs/chat_history_summary.txt"):

        self.summary_file = summary_file
        self.auth_profile = profile_name
        self.authenticate()
        self.initialize_clients()

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", input_key="input", return_messages=True, k=5
        )

        # Tools are initialized lazily on first request from the MCP SSE server.
        self._tool_schemas_loaded = False
        self.tools = []
        self.llm_with_tools = None
        self.mcp_client = None
        self.mcp_server_url = os.environ.get("DELTA_AI_MCP_SSE_URL", "http://127.0.0.1:8765/sse")

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
                if "401" in error_str:
                    self.authenticate()
                    self.initialize_clients()
                    if self._tool_schemas_loaded and self.tools:
                        self.llm_with_tools = self.llm.bind_tools(self.tools)
                    response = self.llm_with_tools.invoke(messages)
                    print("Agent response after retry:", response.content)
                    if response.tool_calls:
                        print("Tool calls after retry:", response.tool_calls)
                else:
                    raise e

            print("Exiting agent_node\n")
            return {"messages": [response]}

        async def tool_node(state: MessagesState):
            print("\nEntering tool_node")
            last_message = state["messages"][-1]
            print("Last message (AIMessage):", last_message.content)
            print("Tool calls to execute:", last_message.tool_calls)
            tool_results = []

            def normalize_tool_result(result):
                if isinstance(result, str):
                    return result
                if isinstance(result, list) and result:
                    first_item = result[0]
                    if isinstance(first_item, dict) and first_item.get("type") == "text":
                        return first_item.get("text", str(result))
                return str(result)

            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args") or {}
                print(f"Executing tool: {tool_name} with args: {tool_args}")
                try:
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    if tool is None:
                        raise RuntimeError(f"Tool not found in bound tools: {tool_name}")
                    
                    try:
                        result = await tool.ainvoke(tool_args)
                    except RuntimeError as exc:
                        raise RuntimeError(f"Error invoking tool '{tool_name}': {str(exc)}") from exc

                    result = normalize_tool_result(result)
                    print(f"Tool result: {result}")
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
                    print(f"{COLOR_RED}{result}{COLOR_RESET}")

                tool_results.append(
                    ToolMessage(
                        content=result,
                        name=tool_name,
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
                if last_message.name == "visualize":
                    print(f"Routing to END (terminal tool: {last_message.name})")
                    return END
                if last_message.name == "run_sql":
                    try:
                        payload = json.loads(last_message.content)
                        if payload.get("status") == "ok":
                            print("Routing to END (run_sql ok)")
                            return END
                        print("Routing to 'agent' (run_sql error)")
                        return "agent"
                    except Exception:
                        print("Routing to 'agent' (run_sql non-JSON result)")
                        return "agent"
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

    def authenticate(self):
        os.system(
            "oci session authenticate --profile-name bmc-sie-prod "
            "--region us-ashburn-1 --tenancy-name bmc_operator_access --auth security_token"
        )

    def initialize_clients(self):
        self.llm = ChatOCIGenAI(
            model_id="ocid1.generativeaimodel....",
            service_endpoint="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com",
            compartment_id="ocid1.tenancy.oc1...",
            auth_type="SECURITY_TOKEN",
            auth_profile=self.auth_profile,
            provider="generic",
            model_kwargs={"temperature": 0,"top_k": 1, "top_p": 0.1}
        )

        self.embeddings = OCIGenAIEmbeddings(
            model_id="cohere.embed-english-v3.0",
            service_endpoint="https://inference.generativeai.uk-london-1.oci.oraclecloud.com",
            compartment_id="ocid1.compartment.oc1...",
            model_kwargs={"truncate": True},
            auth_type="SECURITY_TOKEN",
            auth_profile=self.auth_profile
        )

        self.vectorstore = FAISS.load_local(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore"), embeddings=self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    async def _load_mcp_tools_async(self) -> None:
        if self._tool_schemas_loaded:
            return

        self.mcp_client = MultiServerMCPClient(
            {
                "delta-ai-tools": {
                    "transport": "sse",
                    "url": self.mcp_server_url,
                }
            }
        )
        self.tools = await self.mcp_client.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self._tool_schemas_loaded = True

    async def _ensure_tools_loaded(self) -> None:
        if self._tool_schemas_loaded:
            return
        try:
            await self._load_mcp_tools_async()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tools from MCP server. Ensure the MCP server is running and accessible at the specified URL : {str(exc)}"
            ) from exc

    async def get_agent_response(self, user_query):
        await self._ensure_tools_loaded()

        chat_history = self.memory.load_memory_variables({})["chat_history"]
        state = await self.graph.ainvoke({"messages": chat_history + [HumanMessage(content=user_query)]})
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            response_text = last_msg.content
        elif isinstance(last_msg, ToolMessage):
            if last_msg.name == "visualize":
                response_text = f"<chart-data>{last_msg.content}</chart-data>"
            elif last_msg.name == "run_sql":
                # run_sql returns artifact JSON (ok/error). For ok: include machine-readable payload.
                response_text = f"Query executed. Here is the result : <artifact>{last_msg.content}</artifact>"
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

    async def aclose(self):
        close_method = getattr(self.mcp_client, "aclose", None)
        if callable(close_method):
            await close_method()

    # Used Only for terminal runs
    async def process_input(self, user_input):
        if user_input.lower() == "memorize":
            self.save_summary()
            return
        
        if user_input.lower() == "exit":
            await self.aclose()
            return

        await self.get_agent_response(user_input)

# For testing as script
if __name__ == "__main__":
    chat = DeltaAIChat()
    print(f"{COLOR_YELLOW}Agent: Welcome to Delta AI Chat! How can I help you today?{COLOR_RESET}")
    while True:
        user_input = input(f"{COLOR_BLUE}You: {COLOR_RESET}").strip()
        asyncio.run(chat.process_input(user_input))
