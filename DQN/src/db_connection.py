import pandas as pd
import pyodbc
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


#Class dedicated to established connection to database and fetch relevant data using targeted sql query.

class SalesDatabase:
    def __init__(self, server, database, username, password, port, sql_filepath):
        # Define the connection parameters directly inside the class
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.port = port
        self.driver = '{ODBC Driver 17 for SQL Server}'
        self.sql_query_path = sql_filepath
        self.connection = None

    def connect(self):
        """Establish a connection to the database."""
        connection_string = f'DRIVER={self.driver};SERVER={self.server};PORT={self.port};DATABASE={self.database};UID={self.username};PWD={self.password}'
        self.connection = pyodbc.connect(connection_string)
        print("Connection established successfully.")

    def get_historical_sales_data(self):
        """Read an SQL query from a file and execute it."""
        self.connect()
        if not self.connection:
            raise Exception("Connection is not established. Please connect to the database first.")
        
        with open(self.sql_query_path, 'r') as query_file:
            sql_input = query_file.read()
        
        return pd.read_sql(sql_input, self.connection)
    


# # Usage:
# sales_db = SalesDatabase(server, database, username, password, port, sql_filepath)  # No need to pass any parameters
# df = sales_db.get_historical_sales_data()
# print(df.shape)
# print(df.head())
# print(df.info())
# print(df['CustomerCode'].value_counts())
# print(df['ItemCode'].value_counts())

