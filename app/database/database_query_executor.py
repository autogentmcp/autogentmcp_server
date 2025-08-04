"""
Database query executor for data agents.
"""
import json
from typing import Dict, Any, Optional, List
from app.auth.vault_manager import vault_manager
import sqlparse

print("[DatabaseQueryExecutor] Module loaded - debugging is active!")

class DatabaseQueryExecutor:
    """Execute database queries for data agents using vault credentials with caching support."""
    
    def __init__(self):
        pass
    
    def test_connection(self, vault_key: str, connection_type: str) -> Dict[str, Any]:
        """Test database connection without executing a query."""
        try:
            print(f"[DatabaseQueryExecutor] Testing {connection_type} connection with vault key: {vault_key}")
            
            # Get credentials from vault
            credentials = vault_manager.get_secret(vault_key)
            if not credentials:
                return {
                    "status": "error",
                    "message": f"No credentials found for vault key: {vault_key}"
                }
            
            if connection_type.lower() in ["mssql", "sqlserver"]:
                return self._test_mssql_connection(credentials)
            elif connection_type.lower() == "postgresql":
                return self._test_postgresql_connection(credentials)
            elif connection_type.lower() == "mysql":
                return self._test_mysql_connection(credentials)
            elif connection_type.lower() == "db2":
                return self._test_db2_connection(credentials)
            else:
                return {
                    "status": "error",
                    "message": f"Connection testing not implemented for: {connection_type}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error testing connection: {str(e)}"
            }
    
    def _test_mssql_connection(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Test MSSQL connection."""
        try:
            import pyodbc
            
            server = credentials.get("server") or credentials.get("host")
            database = credentials.get("database")
            username = credentials.get("username") or credentials.get("user")
            password = credentials.get("password")
            port = credentials.get("port", 1433)
            
            # Check available drivers
            available_drivers = [d for d in pyodbc.drivers() if 'SQL Server' in d]
            
            if not available_drivers:
                return {
                    "status": "error",
                    "message": "No SQL Server ODBC drivers found. Please install Microsoft ODBC Driver for SQL Server.",
                    "available_drivers": pyodbc.drivers()
                }
            
            driver = available_drivers[0]  # Use first available
            
            # Build minimal connection string for testing
            conn_str_parts = [
                f"DRIVER={{{driver}}}",
                f"SERVER={server}" + (f",{port}" if port != 1433 else ""),
                f"DATABASE={database}",
                "TrustServerCertificate=yes",
                "Encrypt=no"
            ]
            
            if username and password:
                conn_str_parts.extend([f"UID={username}", f"PWD={password}"])
            else:
                conn_str_parts.append("Trusted_Connection=yes")
            
            conn_str = ";".join(conn_str_parts) + ";"
            
            # Test connection
            with pyodbc.connect(conn_str, timeout=10) as conn:
                with conn.cursor() as cursor:
                    # Simple test query
                    cursor.execute("SELECT 1 as test_column")
                    row = cursor.fetchone()
                    
                    return {
                        "status": "success",
                        "message": "MSSQL connection successful",
                        "test_result": row[0] if row else None,
                        "driver_used": driver,
                        "server": server,
                        "database": database,
                        "available_drivers": available_drivers
                    }
                    
        except Exception as e:
            return {
                "status": "error", 
                "message": f"MSSQL connection test failed: {str(e)}",
                "available_drivers": pyodbc.drivers() if 'pyodbc' in locals() else []
            }
    
    def _test_postgresql_connection(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Test PostgreSQL connection."""
        try:
            import psycopg2
            
            conn_params = {
                "host": credentials.get("host"),
                "port": credentials.get("port", 5432),
                "database": credentials.get("database"),
                "user": credentials.get("username"),
                "password": credentials.get("password")
            }
            
            # Remove None values
            conn_params = {k: v for k, v in conn_params.items() if v is not None}
            
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 as test_column")
                    row = cur.fetchone()
                    
                    return {
                        "status": "success",
                        "message": "PostgreSQL connection successful",
                        "test_result": row[0] if row else None
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "message": f"PostgreSQL connection test failed: {str(e)}"
            }
    
    def _test_mysql_connection(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Test MySQL connection."""
        try:
            import mysql.connector
            
            conn_params = {
                "host": credentials.get("host"),
                "port": credentials.get("port", 3306),
                "database": credentials.get("database"),
                "user": credentials.get("username"),
                "password": credentials.get("password")
            }
            
            # Remove None values
            conn_params = {k: v for k, v in conn_params.items() if v is not None}
            
            with mysql.connector.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 as test_column")
                    row = cur.fetchone()
                    
                    return {
                        "status": "success",
                        "message": "MySQL connection successful",
                        "test_result": row[0] if row else None
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "message": f"MySQL connection test failed: {str(e)}"
            }
    
    def _test_db2_connection(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Test DB2 connection using various methods."""
        try:
            # Try the same connection methods as _try_db2_connection_methods
            success, connection, method, error_msg = self._try_db2_connection_methods(credentials)
            
            if success and connection:
                try:
                    if method == 'native':
                        # For ibm_db connections
                        import ibm_db
                        result = ibm_db.exec_immediate(connection, "SELECT 1 FROM SYSIBM.SYSDUMMY1")
                        ibm_db.close(connection)
                        test_result = 1
                    elif method in ['odbc', 'jdbc']:
                        # For pyodbc or jdbc connections
                        cursor = connection.cursor()
                        cursor.execute("SELECT 1 FROM SYSIBM.SYSDUMMY1")
                        row = cursor.fetchone()
                        test_result = row[0] if row else None
                        cursor.close()
                        connection.close()
                    
                    return {
                        "status": "success",
                        "message": f"DB2 connection successful using {method} method",
                        "test_result": test_result,
                        "connection_details": {
                            "method": method,
                            "server": credentials.get("server"),
                            "database": credentials.get("database"),
                            "port": credentials.get("port", 50000)
                        }
                    }
                except Exception as test_error:
                    return {
                        "status": "error",
                        "message": f"DB2 connection established but test query failed: {str(test_error)}"
                    }
            else:
                return {
                    "status": "error", 
                    "message": f"DB2 connection failed: {error_msg}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"DB2 connection test error: {str(e)}"
            }
    
    def execute_query(
        self, 
        vault_key: str, 
        connection_type: str, 
        sql_query: str,
        limit: int = 100,
        connection_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute SQL query against the database with vault caching support.
        
        Args:
            vault_key: Vault key for database credentials
            connection_type: Type of database (postgresql, mysql, etc.)
            sql_query: SQL query to execute
            limit: Maximum number of rows to return
            connection_config: Optional connection config from registry (takes priority over vault credentials)
            
        Returns:
            Dictionary with query results
        """
        try:
            print(f"[DatabaseQueryExecutor.execute_query] 🔍 STARTING EXECUTION")
            print(f"[DatabaseQueryExecutor.execute_query] Connection type: {connection_type}")
            print(f"[DatabaseQueryExecutor.execute_query] Vault key: {vault_key}")
            print(f"[DatabaseQueryExecutor.execute_query] Original query: {sql_query}")
            print(f"[DatabaseQueryExecutor.execute_query] Limit: {limit}")
            print(f"[DatabaseQueryExecutor.execute_query] Connection config provided: {bool(connection_config)}")
            
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
            
            # Merge connection config from registry if provided (takes priority)
            if connection_config:
                print(f"[DatabaseQueryExecutor] Merging connection config from registry: {connection_config}")
                # Create a copy of credentials and add connectionConfig
                credentials = credentials.copy()
                credentials["connectionConfig"] = connection_config
                print(f"[DatabaseQueryExecutor] Updated credential keys after merge: {list(credentials.keys())}")
            
            # Debug: Check if password looks base64 encoded (only for databases that use passwords)
            password_based_dbs = ["postgresql", "mysql", "mssql", "sqlserver", "db2"]
            if connection_type.lower() in password_based_dbs:
                password = credentials.get("password", "")
                if password:
                    print(f"[DatabaseQueryExecutor] Password length: {len(password)}, contains special chars: {any(c in password for c in '!@#$%^&*()_+-={}[]|;:,.<>?')}")
                else:
                    print(f"[DatabaseQueryExecutor] WARNING: No password found in credentials for {connection_type}")
            else:
                print(f"[DatabaseQueryExecutor] Database type {connection_type} uses service account or token-based authentication")
            
            # Safety validation temporarily disabled
            # if not self._is_safe_query(sql_query):
            #     return {
            #         "status": "error",
            #         "message": "Query contains potentially unsafe operations"
            #     }
            
            # Add limit if not present
            print(f"[DatabaseQueryExecutor.execute_query] ⚡ CALLING _add_limit_to_query")
            sql_query = self._add_limit_to_query(sql_query, limit, connection_type)
            print(f"[DatabaseQueryExecutor.execute_query] ✅ Query after limit processing: {sql_query}")
            
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
    
    def _add_limit_to_query(self, sql_query: str, limit: int, connection_type: str = None) -> str:
        """Add LIMIT/TOP clause to query if not present, based on database type."""
        query_upper = sql_query.upper()
        
        # Debug: Log the connection type and query for troubleshooting
        print(f"[DatabaseQueryExecutor._add_limit_to_query] Connection type: {connection_type}, Query has TOP: {'TOP' in query_upper}, Query has LIMIT: {'LIMIT' in query_upper}")
        
        # SQL Server uses TOP instead of LIMIT
        if connection_type and connection_type.lower() in ["mssql", "sqlserver", "sql-server", "microsoft-sql-server"]:
            print(f"[DatabaseQueryExecutor._add_limit_to_query] Detected SQL Server database")
            
            # CRITICAL FIX: Convert LIMIT to TOP for SQL Server
            if "LIMIT" in query_upper:
                print(f"[DatabaseQueryExecutor._add_limit_to_query] 🔧 CONVERTING LIMIT TO TOP for SQL Server")
                # Extract LIMIT value
                import re
                limit_match = re.search(r'\bLIMIT\s+(\d+)', sql_query, re.IGNORECASE)
                if limit_match:
                    limit_value = limit_match.group(1)
                    # Remove LIMIT clause
                    sql_query_without_limit = re.sub(r'\s*LIMIT\s+\d+\s*', '', sql_query, flags=re.IGNORECASE).rstrip(';')
                    
                    # Add TOP clause after SELECT
                    query_parts = sql_query_without_limit.strip().split()
                    if query_parts and query_parts[0].upper() == "SELECT":
                        if len(query_parts) > 1 and query_parts[1].upper() == "DISTINCT":
                            # Handle SELECT DISTINCT
                            converted_query = f"SELECT DISTINCT TOP {limit_value} " + " ".join(query_parts[2:])
                        else:
                            # Regular SELECT
                            converted_query = f"SELECT TOP {limit_value} " + " ".join(query_parts[1:])
                        print(f"[DatabaseQueryExecutor._add_limit_to_query] ✅ CONVERTED: {sql_query} -> {converted_query}")
                        return converted_query
            
            elif "TOP" not in query_upper:
                # Insert TOP after SELECT (original logic)
                query_parts = sql_query.strip().split()
                if query_parts and query_parts[0].upper() == "SELECT":
                    if len(query_parts) > 1 and query_parts[1].upper() == "DISTINCT":
                        # Handle SELECT DISTINCT
                        return f"SELECT DISTINCT TOP {limit} " + " ".join(query_parts[2:])
                    else:
                        # Regular SELECT
                        return f"SELECT TOP {limit} " + " ".join(query_parts[1:])
                elif "SELECT" in query_upper:
                    # More complex query, try to insert TOP after first SELECT
                    select_index = query_upper.find("SELECT")
                    if select_index >= 0:
                        before_select = sql_query[:select_index]
                        after_select = sql_query[select_index + 6:].strip()
                        if after_select.upper().startswith("DISTINCT"):
                            return f"{before_select}SELECT DISTINCT TOP {limit} {after_select[8:].strip()}"
                        else:
                            return f"{before_select}SELECT TOP {limit} {after_select}"
            print(f"[DatabaseQueryExecutor._add_limit_to_query] SQL Server query returned as-is: {sql_query[:100]}...")
            return sql_query.rstrip(';')
        else:
            # PostgreSQL, MySQL, etc. use LIMIT
            # Only add LIMIT if no limiting clause exists (TOP or LIMIT)
            print(f"[DatabaseQueryExecutor._add_limit_to_query] Detected non-SQL Server database ({connection_type})")
            if "LIMIT" not in query_upper and "TOP" not in query_upper:
                print(f"[DatabaseQueryExecutor._add_limit_to_query] Adding LIMIT to query")
                return f"{sql_query.rstrip(';')} LIMIT {limit}"
            print(f"[DatabaseQueryExecutor._add_limit_to_query] Query already has limiting clause, returning as-is")
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
            
            # Handle service account credentials - check both camelCase and snake_case
            service_account_json = credentials.get("serviceAccountJson") or credentials.get("service_account_json")
            
            # Check for connection config first (higher priority than service account JSON)
            connection_config = credentials.get("connectionConfig", {})
            print(f"[DatabaseQueryExecutor] Using connection config for BigQuery {connection_config}")
            print(f"[DatabaseQueryExecutor] Available credential keys: {list(credentials.keys())}")
            
            # If connectionConfig is empty, check if connection details are directly in credentials
            if not connection_config:
                # Check for direct connection fields in credentials
                direct_config = {}
                for key in ["projectId", "project_id", "database", "dataset", "host", "port", "schema"]:
                    if key in credentials:
                        direct_config[key] = credentials[key]
                if direct_config:
                    print(f"[DatabaseQueryExecutor] Found direct connection config: {direct_config}")
                    connection_config = direct_config
            
            config_project_id = (connection_config.get("projectId") or 
                                connection_config.get("project_id") or
                                connection_config.get("database") or
                                connection_config.get("dataset"))
            
            # Fallback to direct credentials
            fallback_project_id = credentials.get("projectId") or credentials.get("project_id")
            
            # Final project ID selection with logging
            project_id = config_project_id or fallback_project_id
            
            print(f"[DatabaseQueryExecutor] BigQuery project selection:")
            print(f"  - Connection config project: {config_project_id}")
            print(f"  - Credentials project: {fallback_project_id}")
            print(f"  - Selected project: {project_id}")
            
            if service_account_json:
                print(f"[DatabaseQueryExecutor] Using service account authentication for BigQuery")
                # Create temporary credentials file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    if isinstance(service_account_json, str):
                        f.write(service_account_json)
                    else:
                        json.dump(service_account_json, f)
                    temp_creds_path = f.name
                
                try:
                    # Initialize client with service account and override project if specified
                    if project_id:
                        print(f"[DatabaseQueryExecutor] Overriding service account project with: {project_id}")
                        client = bigquery.Client.from_service_account_json(temp_creds_path, project=project_id)
                    else:
                        client = bigquery.Client.from_service_account_json(temp_creds_path)
                finally:
                    # Clean up temporary file
                    os.unlink(temp_creds_path)
            else:
                print(f"[DatabaseQueryExecutor] Using default credentials with project ID: {project_id}")
                # Use default credentials or project ID
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
            
            # Build connection parameters
            server = credentials.get("server") or credentials.get("host")
            database = credentials.get("database")
            username = credentials.get("username") or credentials.get("user")
            password = credentials.get("password")
            port = credentials.get("port", 1433)
            
            # Try different driver names in order of preference
            possible_drivers = [
                credentials.get("driver"),
                "ODBC Driver 18 for SQL Server",
                "ODBC Driver 17 for SQL Server", 
                "ODBC Driver 13 for SQL Server",
                "SQL Server Native Client 11.0",
                "SQL Server"
            ]
            
            driver = None
            available_drivers = [d for d in pyodbc.drivers() if 'SQL Server' in d]
            print(f"[MSSQL] Available ODBC drivers: {available_drivers}")
            
            for possible_driver in possible_drivers:
                if possible_driver and possible_driver in available_drivers:
                    driver = possible_driver
                    break
            
            if not driver and available_drivers:
                driver = available_drivers[0]  # Use first available driver
            
            if not driver:
                return {
                    "status": "error",
                    "message": f"No suitable ODBC driver found. Available drivers: {available_drivers}"
                }
            
            print(f"[MSSQL] Using driver: {driver}")
            
            if not all([server, database]):
                return {
                    "status": "error",
                    "message": "Missing required SQL Server credentials: server and database are required"
                }
            
            # Build connection string with error handling
            conn_str_parts = [f"DRIVER={{{driver}}}"]
            
            # Handle server and port
            if port and port != 1433:
                conn_str_parts.append(f"SERVER={server},{port}")
            else:
                conn_str_parts.append(f"SERVER={server}")
            
            conn_str_parts.append(f"DATABASE={database}")
            
            # Authentication method
            if username and password:
                # SQL Server Authentication
                conn_str_parts.extend([
                    f"UID={username}",
                    f"PWD={password}"
                ])
            else:
                # Windows Authentication
                conn_str_parts.append("Trusted_Connection=yes")
            
            # Connection options to handle common issues
            conn_str_parts.extend([
                "TrustServerCertificate=yes",  # Handle SSL certificate issues
                "Encrypt=no"  # Disable encryption for compatibility (enable if needed)
            ])
            
            conn_str = ";".join(conn_str_parts) + ";"
            
            print(f"[MSSQL] Connection string (password masked): {conn_str.replace(password or '', '***') if password else conn_str}")
            
            # Connect and execute with timeout
            try:
                with pyodbc.connect(conn_str, timeout=30) as conn:
                    conn.timeout = 30  # Set query timeout
                    with conn.cursor() as cursor:
                        print(f"[MSSQL] Executing query: {sql_query[:100]}...")
                        cursor.execute(sql_query)
                        
                        # Get column names
                        columns = [column[0] for column in cursor.description] if cursor.description else []
                        print(f"[MSSQL] Column names: {columns}")
                        
                        # Fetch all rows
                        rows = cursor.fetchall()
                        print(f"[MSSQL] Fetched {len(rows)} rows")
                        
                        # Convert to list of dictionaries with type handling
                        results = []
                        for row in rows:
                            row_dict = {}
                            for i, value in enumerate(row):
                                column_name = columns[i] if i < len(columns) else f"column_{i}"
                                
                                # Handle common SQL Server data types
                                if hasattr(value, 'isoformat'):  # datetime objects
                                    row_dict[column_name] = value.isoformat()
                                elif hasattr(value, 'hex'):  # binary data
                                    row_dict[column_name] = value.hex()
                                elif value is None:
                                    row_dict[column_name] = None
                                else:
                                    row_dict[column_name] = value
                                    
                            results.append(row_dict)
                        
                        return {
                            "status": "success",
                            "row_count": len(results),
                            "data": results,
                            "query": sql_query,
                            "connection_info": {
                                "driver": driver,
                                "server": server,
                                "database": database
                            }
                        }
            
            except pyodbc.Error as db_error:
                error_msg = str(db_error)
                print(f"[MSSQL] Database error: {error_msg}")
                
                # Provide helpful error messages for common issues
                if "Login failed" in error_msg:
                    return {
                        "status": "error",
                        "message": f"SQL Server authentication failed. Check username/password. Original error: {error_msg}"
                    }
                elif "Cannot open database" in error_msg:
                    return {
                        "status": "error", 
                        "message": f"Cannot access database '{database}'. Check database name and permissions. Original error: {error_msg}"
                    }
                elif "TCP Provider" in error_msg or "Named Pipes Provider" in error_msg:
                    return {
                        "status": "error",
                        "message": f"Cannot connect to SQL Server '{server}'. Check server name, port ({port}), and network connectivity. Original error: {error_msg}"
                    }
                elif "SSL Provider" in error_msg or "certificate" in error_msg.lower():
                    return {
                        "status": "error",
                        "message": f"SSL/Certificate error. Try adding TrustServerCertificate=yes to connection. Original error: {error_msg}"
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"SQL Server database error: {error_msg}"
                    }
                    
        except ImportError:
            return {
                "status": "error",
                "message": "pyodbc library not installed. Please install with: pip install pyodbc"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"SQL Server execution error: {str(e)}"
            }
    
    def _execute_db2(self, credentials: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Execute query against IBM DB2."""
        try:
            # Try different DB2 connection methods in order of preference
            return self._try_db2_connection_methods(credentials, sql_query)
                    
        except Exception as e:
            return {
                "status": "error",
                "message": f"DB2 execution error: {str(e)}"
            }
    
    def _try_db2_connection_methods(self, credentials: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Try different DB2 connection methods."""
        
        # Method 1: Try ibm_db (native IBM driver)
        try:
            return self._execute_db2_native(credentials, sql_query)
        except ImportError as e:
            print(f"[DB2] ibm_db not available: {e}")
        except Exception as e:
            print(f"[DB2] Native connection failed: {e}")
        
        # Method 2: Try ODBC connection (works on Windows with DB2 ODBC driver)
        try:
            return self._execute_db2_odbc(credentials, sql_query)
        except ImportError as e:
            print(f"[DB2] pyodbc not available: {e}")
        except Exception as e:
            print(f"[DB2] ODBC connection failed: {e}")
        
        # Method 3: Try JDBC via jaydebeapi (if available)
        try:
            return self._execute_db2_jdbc(credentials, sql_query)
        except ImportError as e:
            print(f"[DB2] jaydebeapi not available: {e}")
        except Exception as e:
            print(f"[DB2] JDBC connection failed: {e}")
        
        # All methods failed
        return {
            "status": "error",
            "message": "All DB2 connection methods failed. Please ensure DB2 drivers are properly installed. "
                      "For Windows: Install IBM DB2 Client or IBM Data Server Driver Package. "
                      "For Linux: Install db2 client libraries. "
                      "Alternative: Use ODBC with 'IBM DB2 ODBC DRIVER' or JDBC with db2jcc4.jar"
        }
    
    def _execute_db2_native(self, credentials: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Execute DB2 query using native ibm_db driver."""
        import ibm_db
        import ibm_db_dbi
        
        # Build connection parameters
        database = credentials.get("database")
        hostname = credentials.get("hostname") or credentials.get("host") or credentials.get("server")
        port = credentials.get("port", 50000)
        username = credentials.get("username") or credentials.get("user")
        password = credentials.get("password")
        
        print(f"[DB2-Native] Connecting to {hostname}:{port}/{database} as {username}")
        
        if not all([database, hostname, username, password]):
            return {
                "status": "error",
                "message": "Missing required DB2 credentials: database, hostname, username, and password are required"
            }
        
        # Build DB2 connection string with additional options
        conn_str_parts = [
            f"DATABASE={database}",
            f"HOSTNAME={hostname}",
            f"PORT={port}",
            f"UID={username}",
            f"PWD={password}",
            f"PROTOCOL=TCPIP"
        ]
        
        # Add optional connection parameters
        if credentials.get("security"):
            conn_str_parts.append(f"SECURITY={credentials.get('security')}")
        if credentials.get("authentication"):
            conn_str_parts.append(f"AUTHENTICATION={credentials.get('authentication')}")
        
        conn_str = ";".join(conn_str_parts) + ";"
        
        print(f"[DB2-Native] Connection string (masked): {conn_str.replace(password, '***')}")
        
        # Connect using ibm_db with timeout
        try:
            conn = ibm_db.connect(conn_str, "", "", {"SQL_ATTR_LOGIN_TIMEOUT": 30})
            if not conn:
                error_msg = ibm_db.conn_errormsg()
                print(f"[DB2-Native] Connection failed: {error_msg}")
                return {
                    "status": "error",
                    "message": f"DB2 connection failed: {error_msg}"
                }
        except Exception as e:
            print(f"[DB2-Native] Connection exception: {e}")
            if "DLL load failed" in str(e) or "specified module could not be found" in str(e).lower():
                return {
                    "status": "error",
                    "message": f"DB2 DLL loading error. Please ensure IBM DB2 Client is properly installed and PATH includes DB2 bin directory. Error: {e}"
                }
            raise
        
        try:
            print(f"[DB2-Native] Executing query: {sql_query[:100]}...")
            
            # Execute query with timeout
            stmt = ibm_db.exec_immediate(conn, sql_query)
            if not stmt:
                error_msg = ibm_db.stmt_errormsg()
                print(f"[DB2-Native] Query execution failed: {error_msg}")
                return {
                    "status": "error",
                    "message": f"DB2 query execution failed: {error_msg}"
                }
            
            # Fetch results
            results = []
            row = ibm_db.fetch_assoc(stmt)
            while row:
                # Convert values for JSON serialization
                row_dict = {}
                for key, value in row.items():
                    if hasattr(value, 'isoformat'):  # datetime objects
                        row_dict[key] = value.isoformat()
                    elif value is None:
                        row_dict[key] = None
                    else:
                        row_dict[key] = value
                results.append(row_dict)
                row = ibm_db.fetch_assoc(stmt)
            
            print(f"[DB2-Native] Fetched {len(results)} rows")
            
            return {
                "status": "success",
                "row_count": len(results),
                "data": results,
                "query": sql_query,
                "connection_method": "ibm_db_native"
            }
            
        finally:
            ibm_db.close(conn)
    
    def _execute_db2_odbc(self, credentials: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Execute DB2 query using ODBC driver."""
        import pyodbc
        
        # Build connection parameters
        database = credentials.get("database")
        server = credentials.get("hostname") or credentials.get("host") or credentials.get("server")
        port = credentials.get("port", 50000)
        username = credentials.get("username") or credentials.get("user")
        password = credentials.get("password")
        
        print(f"[DB2-ODBC] Connecting to {server}:{port}/{database} as {username}")
        
        if not all([database, server, username, password]):
            return {
                "status": "error",
                "message": "Missing required DB2 credentials for ODBC connection"
            }
        
        # Try different DB2 ODBC drivers
        possible_drivers = [
            credentials.get("driver"),
            "IBM DB2 ODBC DRIVER",
            "IBM DB2 ODBC DRIVER - DB2COPY1",
            "IBM DATA SERVER DRIVER for ODBC - DB2COPY1",
            "IBM DATA SERVER DRIVER for ODBC",
        ]
        
        available_drivers = [d for d in pyodbc.drivers() if 'DB2' in d.upper()]
        print(f"[DB2-ODBC] Available DB2 drivers: {available_drivers}")
        
        driver = None
        for possible_driver in possible_drivers:
            if possible_driver and possible_driver in available_drivers:
                driver = possible_driver
                break
        
        if not driver and available_drivers:
            driver = available_drivers[0]
        
        if not driver:
            return {
                "status": "error",
                "message": f"No DB2 ODBC driver found. Available drivers: {available_drivers}. "
                          f"Please install IBM DB2 Client or IBM Data Server Driver Package."
            }
        
        print(f"[DB2-ODBC] Using driver: {driver}")
        
        # Build ODBC connection string
        conn_str_parts = [
            f"DRIVER={{{driver}}}",
            f"DATABASE={database}",
            f"HOSTNAME={server}",
            f"PORT={port}",
            f"UID={username}",
            f"PWD={password}",
            f"PROTOCOL=TCPIP"
        ]
        
        conn_str = ";".join(conn_str_parts) + ";"
        
        print(f"[DB2-ODBC] Connection string (masked): {conn_str.replace(password, '***')}")
        
        # Connect and execute
        with pyodbc.connect(conn_str, timeout=30) as conn:
            with conn.cursor() as cursor:
                print(f"[DB2-ODBC] Executing query: {sql_query[:100]}...")
                cursor.execute(sql_query)
                
                # Get column names
                columns = [column[0] for column in cursor.description] if cursor.description else []
                print(f"[DB2-ODBC] Column names: {columns}")
                
                # Fetch all rows
                rows = cursor.fetchall()
                print(f"[DB2-ODBC] Fetched {len(rows)} rows")
                
                # Convert to list of dictionaries
                results = []
                for row in rows:
                    row_dict = {}
                    for i, value in enumerate(row):
                        column_name = columns[i] if i < len(columns) else f"column_{i}"
                        
                        # Handle data types
                        if hasattr(value, 'isoformat'):  # datetime objects
                            row_dict[column_name] = value.isoformat()
                        elif value is None:
                            row_dict[column_name] = None
                        else:
                            row_dict[column_name] = value
                            
                    results.append(row_dict)
                
                return {
                    "status": "success",
                    "row_count": len(results),
                    "data": results,
                    "query": sql_query,
                    "connection_method": "odbc",
                    "driver_used": driver
                }
    
    def _execute_db2_jdbc(self, credentials: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Execute DB2 query using JDBC driver."""
        import jaydebeapi
        
        # Build connection parameters
        database = credentials.get("database")
        hostname = credentials.get("hostname") or credentials.get("host") or credentials.get("server")
        port = credentials.get("port", 50000)
        username = credentials.get("username") or credentials.get("user")
        password = credentials.get("password")
        
        print(f"[DB2-JDBC] Connecting to {hostname}:{port}/{database} as {username}")
        
        if not all([database, hostname, username, password]):
            return {
                "status": "error",
                "message": "Missing required DB2 credentials for JDBC connection"
            }
        
        # DB2 JDBC connection string
        jdbc_url = f"jdbc:db2://{hostname}:{port}/{database}"
        
        # Try to find DB2 JDBC driver
        import os
        jdbc_driver_paths = [
            credentials.get("jdbc_driver_path"),
            "db2jcc4.jar",
            "db2jcc.jar",
            os.path.join(os.environ.get("DB2_HOME", ""), "java", "db2jcc4.jar"),
            "/opt/ibm/db2/java/db2jcc4.jar",
            "C:\\Program Files\\IBM\\SQLLIB\\java\\db2jcc4.jar"
        ]
        
        jdbc_driver = None
        for path in jdbc_driver_paths:
            if path and os.path.exists(path):
                jdbc_driver = path
                break
        
        if not jdbc_driver:
            return {
                "status": "error",
                "message": f"DB2 JDBC driver not found. Please provide jdbc_driver_path in credentials or place db2jcc4.jar in working directory. Searched paths: {jdbc_driver_paths}"
            }
        
        print(f"[DB2-JDBC] Using JDBC driver: {jdbc_driver}")
        print(f"[DB2-JDBC] JDBC URL: {jdbc_url}")
        
        # Connect and execute
        conn = jaydebeapi.connect(
            "com.ibm.db2.jcc.DB2Driver",
            jdbc_url,
            [username, password],
            jdbc_driver
        )
        
        try:
            cursor = conn.cursor()
            print(f"[DB2-JDBC] Executing query: {sql_query[:100]}...")
            cursor.execute(sql_query)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            print(f"[DB2-JDBC] Column names: {columns}")
            
            # Fetch all rows
            rows = cursor.fetchall()
            print(f"[DB2-JDBC] Fetched {len(rows)} rows")
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                row_dict = {}
                for i, value in enumerate(row):
                    column_name = columns[i] if i < len(columns) else f"column_{i}"
                    
                    # Handle data types
                    if hasattr(value, 'isoformat'):  # datetime objects
                        row_dict[column_name] = value.isoformat()
                    elif value is None:
                        row_dict[column_name] = None
                    else:
                        row_dict[column_name] = value
                        
                results.append(row_dict)
            
            return {
                "status": "success",
                "row_count": len(results),
                "data": results,
                "query": sql_query,
                "connection_method": "jdbc",
                "jdbc_driver": jdbc_driver
            }
            
        finally:
            conn.close()

# Global instance
database_query_executor = DatabaseQueryExecutor()
