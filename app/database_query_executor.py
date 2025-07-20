"""
Database query executor for data agents.
"""
import json
from typing import Dict, Any, Optional, List
from app.vault_manager import vault_manager
import sqlparse

class DatabaseQueryExecutor:
    """Execute database queries for data agents using vault credentials with caching support."""
    
    def __init__(self):
        pass
    
    def execute_query(
        self, 
        vault_key: str, 
        connection_type: str, 
        sql_query: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Execute SQL query against the database with vault caching support.
        
        Args:
            vault_key: Vault key for database credentials
            connection_type: Type of database (postgresql, mysql, etc.)
            sql_query: SQL query to execute
            limit: Maximum number of rows to return
            
        Returns:
            Dictionary with query results
        """
        try:
            print(f"[DatabaseQueryExecutor] Executing {connection_type} query with vault key: {vault_key}")
            
            # Get credentials from vault (with caching)
            credentials = vault_manager.get_secret(vault_key)
            if not credentials:
                print(f"[DatabaseQueryExecutor] No credentials found for vault key: {vault_key}")
                return {
                    "status": "error",
                    "message": f"No credentials found for vault key: {vault_key}"
                }
            
            print(f"[DatabaseQueryExecutor] Retrieved credentials for vault key: {vault_key}")
            print(f"[DatabaseQueryExecutor] Credential keys: {list(credentials.keys())}")
            
            # Debug: Check if password looks base64 encoded
            password = credentials.get("password", "")
            if password:
                print(f"[DatabaseQueryExecutor] Password length: {len(password)}, contains special chars: {any(c in password for c in '!@#$%^&*()_+-={}[]|;:,.<>?')}")
            else:
                print(f"[DatabaseQueryExecutor] WARNING: No password found in credentials")
            
            # Safety validation temporarily disabled
            # if not self._is_safe_query(sql_query):
            #     return {
            #         "status": "error",
            #         "message": "Query contains potentially unsafe operations"
            #     }
            
            # Add limit if not present
            sql_query = self._add_limit_to_query(sql_query, limit)
            
            # Execute based on connection type
            if connection_type.lower() == "postgresql":
                return self._execute_postgresql(credentials, sql_query)
            elif connection_type.lower() == "mysql":
                return self._execute_mysql(credentials, sql_query)
            elif connection_type.lower() == "bigquery":
                return self._execute_bigquery(credentials, sql_query)
            elif connection_type.lower() == "databricks":
                return self._execute_databricks(credentials, sql_query)
            elif connection_type.lower() in ["mssql", "sqlserver"]:
                return self._execute_mssql(credentials, sql_query)
            elif connection_type.lower() == "db2":
                return self._execute_db2(credentials, sql_query)
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported connection type: {connection_type}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing query: {str(e)}"
            }
    
    def _is_safe_query(self, sql_query: str) -> bool:
        """
        Check if SQL query is safe (only allows SELECT statements with comprehensive validation).
        
        Allows:
        - SELECT statements with WHERE, ORDER BY, GROUP BY, HAVING, LIMIT
        - JOINs (INNER, LEFT, RIGHT, FULL OUTER)
        - Aggregate functions (COUNT, SUM, AVG, etc.)
        - Common SQL functions and operators
        
        Blocks:
        - INSERT, UPDATE, DELETE, DROP, ALTER, CREATE statements
        - EXEC, EXECUTE procedure calls
        - Semicolon-separated multiple statements
        - SQL comments that might hide malicious code
        """
        try:
            # Remove extra whitespace and normalize
            query_clean = ' '.join(sql_query.strip().split())
            
            # Check for multiple statements (semicolon separation)
            if query_clean.count(';') > 1 or (query_clean.count(';') == 1 and not query_clean.endswith(';')):
                return False
            
            # Remove trailing semicolon for parsing
            query_for_parsing = query_clean.rstrip(';')
            
            # Check for dangerous keywords (case-insensitive)
            dangerous_keywords = [
                'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE',
                'EXEC', 'EXECUTE', 'CALL', 'MERGE', 'REPLACE', 'LOAD', 'COPY',
                'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'SAVEPOINT',
                'SET', 'DECLARE', 'BEGIN', 'END', 'IF', 'WHILE', 'FOR'
            ]
            
            query_upper = query_for_parsing.upper()
            for keyword in dangerous_keywords:
                # Check if keyword appears as a standalone word (not part of another word)
                import re
                if re.search(r'\b' + keyword + r'\b', query_upper):
                    return False
            
            # Parse the SQL query using sqlparse
            parsed = sqlparse.parse(query_for_parsing)
            if not parsed:
                return False
            
            # Check each statement
            for statement in parsed:
                if not self._is_select_statement(statement):
                    return False
            
            return True
            
        except Exception as e:
            print(f"[DatabaseQueryExecutor] SQL parsing error: {e}")
            # If parsing fails, be conservative and reject
            return False
    
    def _is_select_statement(self, statement) -> bool:
        """Check if a parsed statement is a SELECT statement."""
        try:
            # Get the first meaningful token
            first_token = None
            for token in statement.tokens:
                if not token.is_whitespace and str(token).strip():
                    first_token = token
                    break
            
            if not first_token:
                return False
            
            # Check if it's a SELECT keyword
            token_str = str(first_token).upper().strip()
            
            # Direct SELECT token
            if hasattr(first_token, 'ttype') and first_token.ttype is sqlparse.tokens.Keyword.DML:
                return token_str == 'SELECT'
            
            # SELECT might be in a token group (like "SELECT DISTINCT")
            if hasattr(first_token, 'tokens') and first_token.tokens:
                for sub_token in first_token.tokens:
                    if not sub_token.is_whitespace and str(sub_token).strip():
                        sub_token_str = str(sub_token).upper().strip()
                        if hasattr(sub_token, 'ttype') and sub_token.ttype is sqlparse.tokens.Keyword.DML:
                            return sub_token_str == 'SELECT'
                        # Check if the first non-whitespace part contains SELECT
                        if 'SELECT' in sub_token_str:
                            return sub_token_str.startswith('SELECT')
                        break
            
            # Check if the token string starts with SELECT
            if token_str.startswith('SELECT'):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _add_limit_to_query(self, sql_query: str, limit: int) -> str:
        """Add LIMIT clause to query if not present."""
        query_upper = sql_query.upper()
        if "LIMIT" not in query_upper:
            return f"{sql_query.rstrip(';')} LIMIT {limit}"
        return sql_query
    
    def _execute_postgresql(self, credentials: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Execute query against PostgreSQL database."""
        try:
            import psycopg2
            import psycopg2.extras
            
            # Build connection parameters
            conn_params = {
                "host": credentials.get("host"),
                "port": credentials.get("port", 5432),
                "database": credentials.get("database"),
                "user": credentials.get("username"),
                "password": credentials.get("password")
            }
            
            # Remove None values
            conn_params = {k: v for k, v in conn_params.items() if v is not None}
            
            # Connect and execute
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql_query)
                    rows = cur.fetchall()
                    
                    # Convert to list of dictionaries
                    results = [dict(row) for row in rows]
                    
                    return {
                        "status": "success",
                        "row_count": len(results),
                        "data": results,
                        "query": sql_query
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "message": f"PostgreSQL execution error: {str(e)}"
            }
    
    def _execute_mysql(self, credentials: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Execute query against MySQL database."""
        try:
            import mysql.connector
            
            # Build connection parameters
            conn_params = {
                "host": credentials.get("host"),
                "port": credentials.get("port", 3306),
                "database": credentials.get("database"),
                "user": credentials.get("username"),
                "password": credentials.get("password")
            }
            
            # Remove None values
            conn_params = {k: v for k, v in conn_params.items() if v is not None}
            
            # Connect and execute
            with mysql.connector.connect(**conn_params) as conn:
                with conn.cursor(dictionary=True) as cur:
                    cur.execute(sql_query)
                    rows = cur.fetchall()
                    
                    return {
                        "status": "success",
                        "row_count": len(rows),
                        "data": rows,
                        "query": sql_query
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "message": f"MySQL execution error: {str(e)}"
            }
    
    def _execute_bigquery(self, credentials: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Execute query against Google BigQuery."""
        try:
            from google.cloud import bigquery
            import json
            import tempfile
            import os
            
            # Handle service account credentials
            service_account_json = credentials.get("service_account_json")
            if service_account_json:
                # Create temporary credentials file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    if isinstance(service_account_json, str):
                        f.write(service_account_json)
                    else:
                        json.dump(service_account_json, f)
                    temp_creds_path = f.name
                
                try:
                    # Initialize client with service account
                    client = bigquery.Client.from_service_account_json(temp_creds_path)
                finally:
                    # Clean up temporary file
                    os.unlink(temp_creds_path)
            else:
                # Use default credentials or project ID
                project_id = credentials.get("project_id")
                client = bigquery.Client(project=project_id)
            
            # Execute query
            query_job = client.query(sql_query)
            results = query_job.result()
            
            # Convert to list of dictionaries
            data = []
            for row in results:
                row_dict = {}
                for field in results.schema:
                    row_dict[field.name] = row[field.name]
                data.append(row_dict)
            
            return {
                "status": "success",
                "row_count": len(data),
                "data": data,
                "query": sql_query
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"BigQuery execution error: {str(e)}"
            }
    
    def _execute_databricks(self, credentials: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Execute query against Databricks."""
        try:
            from databricks import sql
            
            # Build connection parameters
            server_hostname = credentials.get("server_hostname")
            http_path = credentials.get("http_path")
            access_token = credentials.get("access_token")
            
            if not all([server_hostname, http_path, access_token]):
                return {
                    "status": "error",
                    "message": "Missing required Databricks credentials: server_hostname, http_path, or access_token"
                }
            
            # Connect and execute
            with sql.connect(
                server_hostname=server_hostname,
                http_path=http_path,
                access_token=access_token
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql_query)
                    rows = cursor.fetchall()
                    
                    # Get column names
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    
                    # Convert to list of dictionaries
                    results = []
                    for row in rows:
                        row_dict = {columns[i]: row[i] for i in range(len(columns))}
                        results.append(row_dict)
                    
                    return {
                        "status": "success",
                        "row_count": len(results),
                        "data": results,
                        "query": sql_query
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "message": f"Databricks execution error: {str(e)}"
            }
    
    def _execute_mssql(self, credentials: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Execute query against Microsoft SQL Server."""
        try:
            import pyodbc
            
            # Build connection string
            server = credentials.get("server")
            database = credentials.get("database")
            username = credentials.get("username")
            password = credentials.get("password")
            port = credentials.get("port", 1433)
            driver = credentials.get("driver", "ODBC Driver 17 for SQL Server")
            
            if not all([server, database]):
                return {
                    "status": "error",
                    "message": "Missing required SQL Server credentials: server and database are required"
                }
            
            # Build connection string
            if username and password:
                # SQL Server Authentication
                conn_str = (
                    f"DRIVER={{{driver}}};"
                    f"SERVER={server},{port};"
                    f"DATABASE={database};"
                    f"UID={username};"
                    f"PWD={password};"
                )
            else:
                # Windows Authentication
                conn_str = (
                    f"DRIVER={{{driver}}};"
                    f"SERVER={server},{port};"
                    f"DATABASE={database};"
                    f"Trusted_Connection=yes;"
                )
            
            # Connect and execute
            with pyodbc.connect(conn_str) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql_query)
                    
                    # Get column names
                    columns = [column[0] for column in cursor.description] if cursor.description else []
                    
                    # Fetch all rows
                    rows = cursor.fetchall()
                    
                    # Convert to list of dictionaries
                    results = []
                    for row in rows:
                        row_dict = {columns[i]: row[i] for i in range(len(columns))}
                        results.append(row_dict)
                    
                    return {
                        "status": "success",
                        "row_count": len(results),
                        "data": results,
                        "query": sql_query
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "message": f"SQL Server execution error: {str(e)}"
            }
    
    def _execute_db2(self, credentials: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Execute query against IBM DB2."""
        try:
            import ibm_db
            import ibm_db_dbi
            
            # Build connection string
            database = credentials.get("database")
            hostname = credentials.get("hostname") or credentials.get("host")
            port = credentials.get("port", 50000)
            username = credentials.get("username")
            password = credentials.get("password")
            
            if not all([database, hostname, username, password]):
                return {
                    "status": "error",
                    "message": "Missing required DB2 credentials: database, hostname, username, and password are required"
                }
            
            # Build DB2 connection string
            conn_str = (
                f"DATABASE={database};"
                f"HOSTNAME={hostname};"
                f"PORT={port};"
                f"UID={username};"
                f"PWD={password};"
                f"PROTOCOL=TCPIP;"
            )
            
            # Connect using ibm_db
            conn = ibm_db.connect(conn_str, "", "")
            if not conn:
                return {
                    "status": "error",
                    "message": f"DB2 connection failed: {ibm_db.conn_errormsg()}"
                }
            
            try:
                # Execute query
                stmt = ibm_db.exec_immediate(conn, sql_query)
                if not stmt:
                    return {
                        "status": "error",
                        "message": f"DB2 query execution failed: {ibm_db.stmt_errormsg()}"
                    }
                
                # Fetch results
                results = []
                row = ibm_db.fetch_assoc(stmt)
                while row:
                    results.append(dict(row))
                    row = ibm_db.fetch_assoc(stmt)
                
                return {
                    "status": "success",
                    "row_count": len(results),
                    "data": results,
                    "query": sql_query
                }
                
            finally:
                ibm_db.close(conn)
                    
        except Exception as e:
            return {
                "status": "error",
                "message": f"DB2 execution error: {str(e)}"
            }

# Global instance
database_query_executor = DatabaseQueryExecutor()
