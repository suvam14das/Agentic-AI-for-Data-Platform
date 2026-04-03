from __future__ import annotations

import json
from mimetypes import init
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_oci.embeddings import OCIGenAIEmbeddings
from pydantic import BaseModel, Field


# -----------------------------
# Pydantic input schemas
# -----------------------------


class RetrievalInput(BaseModel):
    query: str = Field(..., description="Natural language query to retrieve relevant docs/schema snippets.")


class RunSQLInput(BaseModel):
    sql: str = Field(..., description="Spark SQL query to run against the Delta Lake database.")


class FormatToHTMLInput(BaseModel):
    json_str: str = Field(..., description="JSON string with keys: columns (list) and rows (list of lists).")


class VisualizeInput(BaseModel):
    input_str: str = Field(
        ...,
        description=(
            "JSON string: {'data': [list of dict rows], 'chart_type': 'bar'|'line'|'pie', 'x': col, 'y': col}"
        ),
    )


# -----------------------------
# Tool spec / registry
# -----------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: Type[BaseModel]
    handler: Callable[[BaseModel], Any]

    
# -----------------------------
# Tool implementations
# -----------------------------

class ToolsManager:

    def __init__(self, profile_name: str = "bmc-sie-prod"):
        # Local import to avoid import-time side effects for clients.
        try:
            from delta_ai_chat.conn_dataflow import DataflowConnector
        except ImportError:
            from conn_dataflow import DataflowConnector

        self.profile_name = profile_name
        self.oc1_delta_conn = DataflowConnector(profile_name)

        # LLM used for formatting SQL results into strict JSON
        self.llm = ChatOCIGenAI(
            model_id="ocid1.generativeaimodel.oc1.iad.amaaaaaask7dceyargceyuaysrjzo2metq2rinavayxqmpu7tkm6mmfojcvq",  # gemini pro
            service_endpoint="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com",
            compartment_id="ocid1.tenancy.oc1..aaaaaaaat3gxqmhzhjniz6udhx6ak6nngup2quzdahdztnhl7p4oznurigfq",
            auth_type="SECURITY_TOKEN",
            auth_profile=self.profile_name,
            provider="generic",
            model_kwargs={"temperature": 0, "top_k": 1, "top_p": 0.1},
        )

        self.embeddings = OCIGenAIEmbeddings(
            model_id="cohere.embed-english-v3.0",
            service_endpoint="https://inference.generativeai.uk-london-1.oci.oraclecloud.com",
            compartment_id="ocid1.compartment.oc1..aaaaaaaaac64gw2jhiwemjswhxb5odbwpaktqxt5ublisya2uotjn7g6wxqa",
            model_kwargs={"truncate": True},
            auth_type="SECURITY_TOKEN",
            auth_profile=self.profile_name,
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.vectorstore = FAISS.load_local(
            os.path.join(base_dir, "vectorstore"),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def _format_results_with_agent(self, raw_data: Any, is_error: bool = False) -> str:
        if is_error:
            refinement_prompt = (
                f"The following error message was returned from a database query:\n {raw_data}\n"
                "Format this into a concise, user-friendly message using Markdown for better readability. "
                "If the error contains technical details, extract the key issue and present it in a way that a non-technical "
                "user can understand. Do not include stack traces or overly technical jargon. "
                "Focus on the main problem and potential next steps for resolution."
            )
            result = self.llm.invoke(refinement_prompt)
            return result.content.strip()

        refinement_prompt = f"""
            Raw DataFrame: \n {raw_data}\n

            Data formatting : If data header or rows values is in tuples, convert to strings by joining each character in the tuple.
            Use '' as joining delimiter. For example, (a,b,c,1,.,1) should be converted to 'abc1.1'
            , (c, o, u, n, t, (, D, I, S, T, I, N, C, T,  , i, d, )) should be converted to 'count(DISTINCT id)' , and similar.
            Include all characters including numbers and special characters without spaces in between. Do not remove any character
            from the raw data.
            Output format : Return in JSON format only. Do not generate HTML tables. Do not generate additionals comments.
            The JSON should have two keys: "columns" which is a list of column names, and "rows" which is a list of lists, where
            each inner list represents a row of data corresponding to the columns.
            """
        result = self.llm.invoke(refinement_prompt)
        cleaned = (
            result.content.strip()
            .replace("\n", "")
            .replace("```json", "")
            .replace("```", "")
        )
        data = json.loads(cleaned)
        return str(json.dumps(data))

    def _execute_query(self, sql_query: str) -> str:
        try:
            self.oc1_delta_conn.check_connection()
            db_data = self.oc1_delta_conn.pull_data(sql_query)
            return self._format_results_with_agent(db_data, is_error=False)
        except Exception as e:
            error_str = str(e)
            if "401" in error_str:
                self.oc1_delta_conn.connect()
                try:
                    db_data = self.oc1_delta_conn.pull_data(sql_query)
                    return self._format_results_with_agent(db_data, is_error=False)
                except Exception as retry_e:
                    return self._format_results_with_agent(str(retry_e), is_error=True)
            return self._format_results_with_agent(error_str, is_error=True)

    @staticmethod
    def _json_to_html(data: Dict[str, Any]) -> str:
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
            td.text { text-align: left; }
            td.numeric { text-align: right; }
            </style>
            """
        html = style + "<table>"
        html += "<thead><tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr></thead>"
        html += "<tbody>"
        for row in rows:
            html += "<tr>"
            for value in row:
                cell_class = "text"
                if isinstance(value, (int, float)):
                    cell_class = "numeric"
                else:
                    try:
                        float(value)
                        cell_class = "numeric"
                    except Exception:
                        cell_class = "text"
                html += f'<td class="{cell_class}">{value}</td>'
            html += "</tr>"
        html += "</tbody></table>"
        return html

    @staticmethod
    def _create_chart(input_str: str) -> str:
        try:
            input_data = json.loads(input_str)
            data = input_data["data"]
            chart_type = input_data.get("chart_type", "bar")
            x = input_data.get("x")
            y = input_data.get("y")

            df = pd.DataFrame(data)
            labels = df[x].tolist()
            values = df[y].tolist()

            chart_config: Dict[str, Any] = {
                "type": chart_type,
                "data": {
                    "labels": labels,
                    "datasets": [
                        {
                            "label": y,
                            "data": values,
                            "backgroundColor": "rgba(75, 192, 192, 0.2)",
                            "borderColor": "rgba(75, 192, 192, 1)",
                            "borderWidth": 1,
                        }
                    ],
                },
                "options": {
                    "scales": {"y": {"beginAtZero": True}},
                    "plugins": {"legend": {"display": True}},
                    "responsive": True,
                    "maintainAspectRatio": False,
                },
            }

            if chart_type == "pie":
                chart_config["data"]["datasets"][0].pop("borderColor", None)
                chart_config["data"]["datasets"][0].pop("borderWidth", None)
                chart_config["data"]["datasets"][0]["backgroundColor"] = [
                    "rgba(255, 99, 132, 0.2)",
                    "rgba(54, 162, 235, 0.2)",
                    "rgba(255, 206, 86, 0.2)",
                    "rgba(75, 192, 192, 0.2)",
                    "rgba(153, 102, 255, 0.2)",
                ]
            return json.dumps(chart_config)
        except Exception as e:
            return str(e)

    def build_tool_specs(self) -> List[ToolSpec]:
        def retrieval_handler(inp: RetrievalInput) -> str:
            try : 
                docs = self.retriever.invoke(inp.query)
                return "\n\n".join([d.page_content for d in docs])
            except Exception as e:
                if "401" in str(e):
                    self.oc1_delta_conn.connect()
                    docs = self.retriever.invoke(inp.query)
                    return "\n\n".join([d.page_content for d in docs])
                else:
                    return str(e)

        def run_sql_handler(inp: RunSQLInput) -> str:
            return self._execute_query(inp.sql)

        def format_to_html_handler(inp: FormatToHTMLInput) -> str:
            return ToolsManager._json_to_html(json.loads(inp.json_str))

        def visualize_handler(inp: VisualizeInput) -> str:
            return ToolsManager._create_chart(inp.input_str)

        return [
            ToolSpec(
                name="retrieval",
                description="Retriever for CDI/COMPUTE_DATA database schema docs including tables, columns, usage and cross table relationships. Also retrieves Compute Domain context from documentation about various services and projects in OCI.",
                input_model=RetrievalInput,
                handler=retrieval_handler,
            ),
            ToolSpec(
                name="run_sql",
                description="Execute a Spark SQL query on the Delta Lake database and return JSON string.",
                input_model=RunSQLInput,
                handler=run_sql_handler,
            ),
            ToolSpec(
                name="format_to_html",
                description="Convert JSON string (with columns/rows) to an HTML table string for display.",
                input_model=FormatToHTMLInput,
                handler=format_to_html_handler,
            ),
            ToolSpec(
                name="visualize",
                description="Generate Chart.js configuration JSON from JSON data string.",
                input_model=VisualizeInput,
                handler=visualize_handler,
            )
        ]
    
    def cleanup(self):
        print(f"Cleaning up resources...")
        self.oc1_delta_conn.check_connection()
        self.oc1_delta_conn.close()
