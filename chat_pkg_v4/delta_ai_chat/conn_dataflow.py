import re
import subprocess
import sys
from time import sleep, time
import math
from typing import Any, Dict, List, Optional

import pandas as pd
import jaydebeapi
import jpype
import os

# retry wrapper function
def wrapper_retry_timer(self, func, n_retries=1, n_delay=1):
    m_retries, m_delay = n_retries, n_delay

    def inner(*args, **kargs):
        nonlocal m_retries, m_delay
        start = time()

        name_func = func.__name__
        if name_func.startswith("main"):
            m_retries = 1

        while m_retries:
            try:
                result = func(*args, **kargs)
                if result is not None and result != -1:
                    break
            except Exception as e:
                print(f'retry left: {m_retries}')
                m_retries -= 1
                msg = f'{e}, Retrying in {m_delay} seconds...'
                print(msg)
                sleep(m_delay)
                result = None

        elapsed = time() - start

        if elapsed < 60:
            print(f"f: {name_func} --- used {elapsed:.2f}s")
        else:
            print(f"f: {name_func} --- used {elapsed:.2f}s -> {elapsed / 60.0:.2f}m")

        return result

    return inner

class DataflowConnector:
    _TUPLE_STR_RE = re.compile(r"^\(([^()]+)\)$")
    JDBC_JAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SimbaSparkJDBC-2.6.18.2067/SimbaSparkJDBC42-2.6.18.2067/SparkJDBC42.jar")
    BASE_JDBC_URL = "<JDBC-url>"
    JDBC_DRIVER = "com.simba.spark.jdbc.Driver"

    def __init__(self, profile_name):
        self.profile_name = profile_name
        self.jdbc_url = f"{self.BASE_JDBC_URL};ociProfile={self.profile_name}"
        self.connection = None
        self.connect()

    def connect(self, retry=3):
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[self.JDBC_JAR])

        os.system("oci session authenticate --profile-name bmc-sie-prod --region us-ashburn-1 --tenancy-name bmc_operator_access --auth security_token")

        print(f"Connecting to Dataflow SQL endpoint with profile {self.profile_name}")

        for i in range(1, retry + 1):
            try:
                self.connection = jaydebeapi.connect(
                    jclassname=self.JDBC_DRIVER,
                    url=self.jdbc_url,
                    driver_args={},
                    jars=self.JDBC_JAR
                )
                print("Connection success")
                break
            except Exception as e:
                print(e)
                print(f"Retry {i}, waiting {i} seconds")
                sleep(i)

        if not self.connection:
            raise ConnectionError("Failed to connect after retries")

        cursor = self.connection.cursor()
        print("Successfully connected to Dataflow SQL endpoint")
        return self.connection

    def dataframe_to_clean_string(self, df: pd.DataFrame) -> str:
        """
        Debug helper (not used for downstream parsing anymore).

        Converts a pandas DataFrame to a formatted string,
        then collapses tuple-like values (p, h, x, 1, 4, ., 4)
        into proper strings (phx14.4) while preserving table formatting.
        """
        df_string = df.to_string()

        pattern = re.compile(r"\(([^()]+)\)")

        def collapse_match(match):
            content = match.group(1)
            chars = [x.strip() for x in content.split(",")]
            return "".join(chars) + "|"

        cleaned_string = pattern.sub(collapse_match, df_string)
        return cleaned_string

    def _collapse_malformed_tuple_value(self, v: Any) -> Any:
        """
        Deterministically normalize cell values that come back as malformed tuples.

        Supported inputs:
        - Python tuple/list: ('p','h','x') -> 'phx'
        - String that looks like a tuple: "(p, h, x)" -> "phx"
        """
        if v is None:
            return None

        # pandas NaN values
        if isinstance(v, float) and math.isnan(v):
            return None

        # pandas NaT
        if v is pd.NaT:
            return None

        if isinstance(v, (tuple, list)):
            return "".join([str(x).strip() for x in v])

        if isinstance(v, str):
            s = v.strip()
            m = self._TUPLE_STR_RE.match(s)
            if m:
                parts = [p.strip() for p in m.group(1).split(",")]
                return "".join(parts)
            return v

        return v

    def dataframe_to_columns_rows(self, df: pd.DataFrame, max_rows: Optional[int] = None) -> Dict[str, Any]:
        """
        Convert DataFrame to a deterministic python object:
          {"columns": [...], "rows": [[...],[...]]}

        This avoids any LLM formatting and avoids truncation/token limits.
        """
        if max_rows is not None:
            df = df.head(max_rows)

        columns: List[str] = [str(c) for c in df.columns.tolist()]
        rows: List[List[Any]] = []

        for row in df.itertuples(index=False, name=None):
            rows.append([self._collapse_malformed_tuple_value(v) for v in row])

        return {"columns": columns, "rows": rows}
        
    def check_connection(self):
        try:
            with self.connection.cursor() as cursor:
                # Use a lightweight query suitable for your DB
                cursor.execute("SELECT 1")
                cursor.fetchall()
            return True
        except Exception:
            return False
    
    def pull_data(self, query, max_rows: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute SQL and return a deterministic python object with columns/rows.

        This is intentionally NOT a large formatted string to avoid downstream
        truncation/token-limit issues.

        Args:
          query: SQL query string
          max_rows: optional cap for rows returned in-memory (CSV persistence can still
                    choose to write full data if desired). If None, returns all rows.
        """
        print("Retrieving data")
        try:
            df = pd.read_sql_query(query, self.connection)
            # Optional debug (do not use for parsing):
            # print(self.dataframe_to_clean_string(df))
            print("Data retrieval successful, processing into columns/rows format")
            print(self.dataframe_to_columns_rows(df, max_rows=max_rows))
            return self.dataframe_to_columns_rows(df, max_rows=max_rows)
        
        except Exception as e:
            raise (e)

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            print("Connection closed")


if __name__ == "__main__":
    dataflow_conn = DataflowConnector('bmc-sie-prod')
    sql_query = "SHOW DATABASES"
    df = dataflow_conn.pull_data(sql_query)
    if df is not None:
        print(df)
        print("✅ Extracted data successfully!")
    dataflow_conn.close()
