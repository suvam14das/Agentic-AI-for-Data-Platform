from datetime import datetime
import json
import os
import sys
import pandas as pd
from delta_ai_chat.conn_dataflow import DataflowConnector
from delta_ai_chat.generate_vector_store import generate_vector_store
import warnings
import oci
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_classic.memory import ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_oci.embeddings import OCIGenAIEmbeddings
import re
from langchain_community.vectorstores import FAISS
from delta_ai_chat.LoadProperties import LoadProperties

warnings.filterwarnings("ignore")

COLOR_BLUE = "\033[94m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"

prompt_template = """
{context}

History : {chat_history}

User: {question}

Rules :
1. Respond concisely with only relevant details and still be polite and helpful. Use Markdown formatting for better readability, such as bullets for lists, tables for data, bold/italics for emphasis, and proper paragraphs with line breaks. If the user is asking a general Compute domain question first look for in the documentations. 
2. If the user is asking a Compute domain question that requires data from the DeltaLake DB then generate or refine SQL and output the SparkSQL code wrapped in ```sql\nSQL HERE\n``` along with a short reasoning behind the query. 
3. If the user gives you are query to run don't change the inherent tables, columns and joins. If the SQl needs refinement to run on DeltaLake then refine the SQL with the same tables, columns and joins but make it compatible for SparkSQL and output the refined SQL wrapped in ```sql\nSQL HERE\n``` along with a short reasoning behind the refinement.
4. Do not use tables that are not present in the database. Do not use columns that are not present in the respective  table.  
5. Columns must be consistent to the table schema queried. Do not wrap the entire SQL in backticks. Wrap only column names that contains $ with backticks. Always use full name of the column along with proper table alias in the SQL. Try to find the relevant columns within the same table to build the query. 
6.Always use tables in SQL query in the format <database>.<table> e.g. cdi.hosts, cdi.instances etc. 
7. For relevant SQLs that supports limit if the limit of rows is not specified or evident use LIMIT 10. 
8. For general questions, provide a polite direct and relevant response and if the answer is not known just say "Sorry I did not get you. My AI is not AIing!".
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context","chat_history","question"]
)

class DeltaAIChat:
    def __init__(self, profile_name='bmc-sie-prod', summary_file="delta_ai_chat/general_docs/chat_history_summary.txt"):
        self.properties = LoadProperties()
        self.oc1_delta_conn = DataflowConnector(profile_name)
        self.summary_file = summary_file

        self.llm = ChatOCIGenAI(
            # model_id="ocid1.generativeaimodel.oc1.iad.amaaaaaask7dceyaeo4ehrn25guuats5s45hnvswlhxo6riop275l2bkr2vq", #gemini flash
            model_id="ocid1.generativeaimodel.oc1.iad.amaaaaaask7dceyargceyuaysrjzo2metq2rinavayxqmpu7tkm6mmfojcvq", #gemini pro
            service_endpoint="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com",
            compartment_id="ocid1.tenancy.oc1..aaaaaaaat3gxqmhzhjniz6udhx6ak6nngup2quzdahdztnhl7p4oznurigfq",
            auth_type="SECURITY_TOKEN",
            auth_profile=profile_name,
            provider="generic",
            model_kwargs={"temperature": 0,"top_k": 1, "top_p": 0.1}
        )

        self.embeddings = OCIGenAIEmbeddings(
            model_id=self.properties.getEmbeddingModelName(),
            service_endpoint=self.properties.getEndpoint(),
            compartment_id=self.properties.getCompartment(),
            model_kwargs={"truncate": True},
            auth_type="SECURITY_TOKEN",
            auth_profile=profile_name,
        )

        self.vectorstore = FAISS.load_local(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore"), embeddings=self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

        self.current_sql = None

    def get_agent_response(self, user_query):
        result = self.qa_chain.invoke({"question": user_query})
        response_text = result["answer"].strip()
        print(f"{COLOR_YELLOW}Agent: {response_text}{COLOR_RESET}")
                
        sql_match = re.search(r'```sql\n(.*?)```', response_text, re.DOTALL)
        if sql_match:
            return response_text, sql_match.group(1).strip()
        return response_text, None

    def format_results_with_agent(self, raw_data, is_error=False):
        
        if is_error:
            refinement_prompt = (
                f"The following error message was returned from a database query:\n {raw_data}\n"
                "Format this into a concise, user-friendly message using Markdown for better readability. If the error contains technical details, extract the key issue and present it in a way that a non-technical user can understand. Do not include stack traces or overly technical jargon. Focus on the main problem and potential next steps for resolution."
            )
            result = self.qa_chain.invoke({"question": refinement_prompt})
            return result["answer"].strip()
        else:
            refinement_prompt = (
                f"""
                
                Raw DataFrame: \n {raw_data}\n
                
                Data formatting : If data header or rows values is in tuples, convert to strings by joining each character in the tuple. Use '' as joining delimiter. For example, (a,b,c,1,.,1) should be converted to 'abc1.1' , (c, o, u, n, t, (, D, I, S, T, I, N, C, T,  , i, d, )) should be converted to 'count(DISTINCT id)' , and similar. Include all characters including numbers and special characters without spaces in between. Do not remove any character from the raw data. 
                Output format : Return in JSON format only. Do not generate HTML tables. Do not generate additionals comments. The JSON should have two keys: "columns" which is a list of column names, and "rows" which is a list of lists, where each inner list represents a row of data corresponding to the columns. For example:
                """
            )
            result = self.llm.invoke(refinement_prompt)
            data = json.loads(result.content.strip().replace("\n", "").replace("```json", "").replace("```", ""))
            print(f"{COLOR_YELLOW}Refined Result: {data}{COLOR_RESET}")
            html = self.json_to_html(data)
            return html

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
                    print(f"{COLOR_RED}Error : {error_str}{COLOR_RESET}")
                    formatted_response = self.format_results_with_agent(error_str, is_error=True)
                    sql_match = re.search(r'```sql\n(.*?)```', formatted_response, re.DOTALL)
                    if sql_match:
                        return formatted_response, sql_match.group(1).strip()
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

        return html

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
        
        if "run sql" in user_input.lower() or "get data" in user_input.lower():
            if self.current_sql:
                print(f"{COLOR_YELLOW}Executing query...{COLOR_RESET}")
                self.execute_query(self.current_sql, user_input)
                print(f"{COLOR_YELLOW}Query executed. Back to chat.{COLOR_RESET}")
            else:
                print(f"{COLOR_YELLOW}No SQL query has been generated yet.{COLOR_RESET}")
            return
        
        response_text, new_sql = self.get_agent_response(user_input)
        if new_sql:
            self.current_sql = new_sql

# For testing as script
if __name__ == "__main__":
    chat = DeltaAIChat()
    print(f"{COLOR_YELLOW}Agent: Welcome to Delta AI Chat! How can I help you today?{COLOR_RESET}")
    while True:
        user_input = input(f"{COLOR_BLUE}You: {COLOR_RESET}").strip()
        chat.process_input(user_input)
