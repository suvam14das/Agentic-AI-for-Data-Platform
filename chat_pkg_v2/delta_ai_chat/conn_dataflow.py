import re
import subprocess
import sys
from time import sleep, time
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
    JDBC_JAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SimbaSparkJDBC-2.6.18.2067/SimbaSparkJDBC42-2.6.18.2067/SparkJDBC42.jar")
    BASE_JDBC_URL = "<jdbc-url>"
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
        Converts a pandas DataFrame to a formatted string,
        then collapses tuple-like values (p, h, x, 1, 4, ., 4)
        into proper strings (phx14.4) while preserving table formatting.
        """
        
        # Step 1: Convert dataframe to formatted string
        df_string = df.to_string()
        
        # Step 2: Regex to find tuple-like patterns
        # Matches: (p, h, x, 1, 4, ., 4)
        pattern = re.compile(r"\(([^()]+)\)")
        
        def collapse_match(match):
            content = match.group(1)
            # Split by comma and strip spaces
            chars = [x.strip() for x in content.split(",")]
            # Join everything together
            return "".join(chars)
        
        # Step 3: Replace tuples with collapsed string
        cleaned_string = pattern.sub(collapse_match, df_string)
        
        return cleaned_string
        
    def check_connection(self):
        try:
            with self.connection.cursor() as cursor:
                # Use a lightweight query suitable for your DB
                cursor.execute("SELECT 1")
                cursor.fetchall()
            return True
        except Exception:
            return False
    
    def pull_data(self, query):
        # if not self.check_connection():
        #     self.connect()

        print("Retrieving data")
        try:
            df = pd.read_sql_query(query, self.connection)
            # print(df)
            # return df
            print(self.dataframe_to_clean_string(df))
            return self.dataframe_to_clean_string(df)
        except Exception as e:
            raise(e)

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            print("Connection closed")


# if __name__ == "__main__":
#     dataflow_conn = DataflowConnector('bmc-sie-prod')
#     sql_query = "SHOW DATABASES"
#     df = dataflow_conn.pull_data(sql_query)
#     if df is not None:
#         print(df)
#         print("✅ Extracted data successfully!")
#     dataflow_conn.close()
