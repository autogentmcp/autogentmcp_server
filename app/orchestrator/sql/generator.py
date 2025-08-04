"""
SQL generation with validation and safety checks
"""

from typing import Dict, List, Any
from .validator import SQLValidator
from ..models import ExecutionContext
from app.llm import MultiModeLLMClient
from app.registry.client import get_enhanced_agent_details_for_llm
from app.database.sql_prompt_builder import sql_prompt_builder

class SQLGenerator:
    """Generates and validates SQL queries for data agents"""
    
    def __init__(self):
        self.llm_client = MultiModeLLMClient()
        self.validator = SQLValidator()
    
    async def generate_sql(self, agent_id: str, query: str, max_retries: int = 2) -> Dict[str, Any]:
        """Generate SQL for data agent using modular prompt builder with validation and retry loop"""
        
        print(f"[SQLGenerator] Generating SQL for agent {agent_id}")
        
        agent_details = get_enhanced_agent_details_for_llm(agent_id)
        
        # Build simplified schema structure with key details
        tables = agent_details.get("tables_with_columns", [])
        schema_tables = []
        
        for table in tables:  # Pass ALL tables to LLM for accurate query generation
            table_name = table.get('tableName')
            schema_name = table.get('schemaName', 'dbo')
            row_count = table.get('rowCount', 0)
            
            # Create simplified table structure with metadata
            table_schema = {
                "tableName": f"{schema_name}.{table_name}" if schema_name != "public" else table_name,
                "rowCount": row_count,
                "columns": []
            }
            
            columns = table.get("columns", [])  # Pass ALL columns for accurate schema representation
            for col in columns:
                col_name = col.get('columnName', '')
                data_type = col.get('dataType', '')
                is_pk = col.get('isPrimaryKey', False)
                is_fk = col.get('isForeignKey', False)
                ai_description = col.get('aiDescription', '')
                
                column_info = {
                    "name": col_name,
                    "type": data_type
                }
                
                # Add AI description if available for better LLM understanding
                if ai_description:
                    try:
                        import json
                        desc_data = json.loads(ai_description) if isinstance(ai_description, str) else ai_description
                        if isinstance(desc_data, dict):
                            purpose = desc_data.get('purpose', '')
                            if purpose:
                                column_info["aiDescription"] = purpose
                    except (json.JSONDecodeError, TypeError):
                        # If parsing fails, use raw description if it's a string
                        if isinstance(ai_description, str) and ai_description.strip():
                            column_info["aiDescription"] = ai_description.strip()
                
                # Add key information if relevant
                if is_pk or is_fk:
                    column_info["keys"] = []
                    if is_pk:
                        column_info["keys"].append("PK")
                    if is_fk:
                        column_info["keys"].append("FK")
                
                table_schema["columns"].append(column_info)
            
            schema_tables.append(table_schema)
        
        # Detect query type for optimized prompt
        query_type = sql_prompt_builder.detect_query_type(query)
        database_type = agent_details.get("database_type", "unknown")
        
        # Extract custom prompt from environment section
        environment_details = agent_details.get("environment", {})
        custom_prompt = environment_details.get("customPrompt", "")
        
        # Extract sample queries and business context from agent details
        sample_queries = agent_details.get("sample_queries", [])
        business_context = agent_details.get("business_context", "")
        
        # Retry loop for column validation failures
        for attempt in range(max_retries + 1):
            print(f"[SQLGenerator] Attempt {attempt + 1}/{max_retries + 1}")
            
            # Build optimized prompt with agent-specific guidelines
            # Add validation feedback for retry attempts
            validation_feedback = ""
            if attempt > 0:
                validation_feedback = f"\n\nIMPORTANT: Previous attempt failed column validation. Please carefully review the exact column names in the schema above and use ONLY those column names in your SQL query."
            
            prompt = sql_prompt_builder.build_prompt(
                query=query,
                schema=schema_tables,
                database_type=database_type,
                query_type=query_type,
                custom_prompt=custom_prompt,
                sample_queries=sample_queries,
                business_context=business_context,
                validation_feedback=validation_feedback
            )
            
            if attempt == 0:
                print(f"[SQLGenerator] Query Type: {query_type}, Database: {database_type}")
                if custom_prompt:
                    print(f"[SQLGenerator] Using custom prompt guidelines from environment")
                print(f"[SQLGenerator] Optimized Prompt Length: {len(prompt)} chars")
            else:
                print(f"[SQLGenerator] Retry attempt with validation feedback")
            
            response = self.llm_client.invoke_with_json_response(prompt, task_type="sql_generation")
            print(f"[SQLGenerator] DEBUG LLM response received: {response}")
            
            # Handle the structured response and validate SQL
            if response and response.get("status") == "ready":
                sql = response.get("query", "").strip()
                # Clean SQL and remove any markdown formatting
                sql = sql.replace("```sql", "").replace("```", "").strip()
                
                # CRITICAL: Validate SQL for security and column existence
                validation_result = self.validator.validate_sql_query(sql)
                
                # Additional validation: Check if all columns in SQL exist in schema
                column_validation = self._validate_columns_exist(sql, schema_tables)
                
                if not validation_result["is_valid"]:
                    print(f"[SQLGenerator] SQL validation failed: {validation_result['error']}")
                    if attempt == max_retries:
                        return {
                            "sql": None,
                            "status": "validation_failed",
                            "error": validation_result["error"],
                            "output_format": ["table"],
                            "chart_spec": {},
                            "reasoning": f"Generated SQL failed validation after {max_retries + 1} attempts: {validation_result['error']}",
                            "query_type": query_type,
                            "validation_warnings": validation_result.get("warnings", []),
                            "attempts_made": attempt + 1
                        }
                    continue  # Try again
                
                if not column_validation["is_valid"]:
                    print(f"[SQLGenerator] Column validation failed: {column_validation['error']}")
                    if attempt == max_retries:
                        return {
                            "sql": None,
                            "status": "validation_failed", 
                            "error": column_validation["error"],
                            "output_format": ["table"],
                            "chart_spec": {},
                            "reasoning": f"Generated SQL uses non-existent columns after {max_retries + 1} attempts: {column_validation['error']}",
                            "query_type": query_type,
                            "validation_warnings": column_validation.get("warnings", []),
                            "attempts_made": attempt + 1
                        }
                    
                    # Update validation feedback for next attempt
                    missing_columns = column_validation.get("missing_columns", [])
                    available_columns = column_validation.get("available_columns", [])
                    
                    validation_feedback = f"\n\nCOLUMN VALIDATION ERROR: The columns {missing_columns} do not exist in the schema."
                    if available_columns:
                        validation_feedback += f" Available columns are: {available_columns}. Please use ONLY these exact column names."
                    
                    continue  # Try again with feedback
                
                # Success! Both validations passed
                print(f"[SQLGenerator] SQL validation passed on attempt {attempt + 1}")
                
                # Log any warnings
                if validation_result.get("warnings"):
                    print(f"[SQLGenerator] SQL validation warnings: {validation_result['warnings']}")
                
                # Sanitize the SQL
                sanitized_sql = self.validator.sanitize_sql_query(sql)
                
                return {
                    "sql": sanitized_sql,
                    "status": "ready",
                    "output_format": response.get("output_format", ["table"]),
                    "chart_spec": response.get("chart_spec", {}),
                    "reasoning": response.get("reasoning", ""),
                    "tables_used": response.get("tables_used", []),
                    "query_type": query_type,
                    "validation_warnings": validation_result.get("warnings", []),
                    "attempts_made": attempt + 1
                }
            
            elif response and response.get("status") == "needs_clarification":
                # For clarification needs, don't retry - return immediately
                clarifications = response.get("clarification_needed", [])
                print(f"[SQLGenerator] Clarification needed: {clarifications}")
                return {
                    "sql": None,
                    "status": "needs_clarification",
                    "output_format": ["table"],
                    "chart_spec": {},
                    "clarification_needed": clarifications,
                    "reasoning": response.get("reasoning", "Need clarification"),
                    "query_type": query_type,
                    "attempts_made": attempt + 1
                }
            else:
                # LLM failed to generate proper response
                if attempt == max_retries:
                    reasoning = response.get("reasoning", "Unknown reason") if response else "Failed to generate response"
                    print(f"[SQLGenerator] Cannot proceed after {max_retries + 1} attempts: {reasoning}")
                    return {
                        "sql": None,
                        "status": "cannot_proceed",
                        "error": "Failed to generate valid SQL query",
                        "output_format": ["table"],
                        "chart_spec": {},
                        "reasoning": reasoning,
                        "query_type": query_type,
                        "attempts_made": attempt + 1
                    }
                continue  # Try again

    async def generate_sql_with_filtered_tables(self, agent_id: str, query: str, filtered_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate SQL using only the pre-selected relevant tables"""
        
        print(f"[SQLGenerator] Generating SQL with {len(filtered_tables)} filtered tables")
        
        # Build schema context from filtered tables only
        schema_tables = []
        for table in filtered_tables:
            table_name = table.get('tableName')
            schema_name = table.get('schemaName', 'dbo')
            row_count = table.get('rowCount', 0)
            
            # Create simplified table structure with metadata
            table_schema = {
                "tableName": f"{schema_name}.{table_name}" if schema_name != "public" else table_name,
                "rowCount": row_count,
                "columns": []
            }
            
            # Include ALL columns from the filtered tables for accurate SQL generation
            columns = table.get("columns", [])
            for col in columns:
                col_name = col.get('columnName', '')
                data_type = col.get('dataType', '')
                is_pk = col.get('isPrimaryKey', False)
                is_fk = col.get('isForeignKey', False)
                ai_description = col.get('aiDescription', '')
                
                column_info = {
                    "name": col_name,
                    "type": data_type,
                    "is_primary_key": is_pk,
                    "is_foreign_key": is_fk
                }
                
                # Add AI description if available for better LLM understanding
                if ai_description:
                    try:
                        import json
                        desc_data = json.loads(ai_description) if isinstance(ai_description, str) else ai_description
                        if isinstance(desc_data, dict):
                            purpose = desc_data.get('purpose', '')
                            if purpose:
                                column_info["aiDescription"] = purpose
                    except (json.JSONDecodeError, TypeError):
                        # If parsing fails, use raw description if it's a string
                        if isinstance(ai_description, str) and ai_description.strip():
                            column_info["aiDescription"] = ai_description.strip()
                
                table_schema["columns"].append(column_info)
            
            schema_tables.append(table_schema)
        
        # Get agent details for connection type and custom prompts
        agent_details = get_enhanced_agent_details_for_llm(agent_id)
        connection_type = agent_details.get("connection_type", "unknown")
        custom_prompt = agent_details.get("custom_prompt", "")
        agent_name = agent_details.get("name", agent_id)
        
        # Build the prompt using filtered schema
        prompt = sql_prompt_builder.build_prompt(
            query=query,
            schema=schema_tables,
            database_type=connection_type,
            custom_prompt=custom_prompt
        )
        
        # Use SQL generation task type for better routing
        response = self.llm_client.invoke_with_json_response(
            prompt=prompt,
            task_type="sql_generation",
            timeout=60
        )
        
        if not response:
            return {
                "sql": None,
                "status": "llm_error",
                "error": "Failed to get response from LLM",
                "output_format": ["table"],
                "chart_spec": {},
                "reasoning": "LLM did not respond",
                "filtered_table_count": len(filtered_tables)
            }
        
        sql_query = response.get("query")
        reasoning = response.get("thought", "")
        
        if not sql_query:
            return {
                "sql": None,
                "status": "no_sql_generated",
                "error": "LLM did not generate SQL query",
                "output_format": ["table"],
                "chart_spec": {},
                "reasoning": reasoning,
                "filtered_table_count": len(filtered_tables)
            }
        
        # Validate the generated SQL
        validation_result = self.validator.validate_sql_query(sql_query)
        
        if validation_result.get("is_valid"):
            print(f"[SQLGenerator] SQL validation passed for filtered query")
            return {
                "sql": sql_query,
                "status": "ready",
                "output_format": ["table"],
                "chart_spec": {},
                "reasoning": reasoning,
                "query_type": "optimized_filtered",
                "filtered_table_count": len(filtered_tables),
                "tables_used": [t.get("tableName") for t in filtered_tables]
            }
        else:
            print(f"[SQLGenerator] SQL validation failed: {validation_result.get('error')}")
            return {
                "sql": sql_query,
                "status": "validation_failed",
                "error": validation_result.get("error", "SQL validation failed"),
                "output_format": ["table"],
                "chart_spec": {},
                "reasoning": reasoning,
                "filtered_table_count": len(filtered_tables),
                "validation_warnings": validation_result.get("warnings", [])
            }
    
    def _validate_columns_exist(self, sql: str, schema_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate that all columns referenced in SQL exist in the provided schema
        """
        import re
        
        # Extract all column references from SQL
        # This is a simple pattern - matches word.word patterns and standalone words that could be columns
        column_patterns = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*|(?:SELECT|,)\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
        
        # Build set of valid columns from schema
        valid_columns = set()
        available_columns = []
        for table in schema_tables:
            table_name = table.get("tableName", "")
            columns = table.get("columns", [])
            for col in columns:
                col_name = col.get("name", "")
                if col_name:
                    valid_columns.add(col_name.lower())
                    available_columns.append(col_name)
                    # Also add table.column format
                    if table_name:
                        valid_columns.add(f"{table_name.lower()}.{col_name.lower()}")
        
        # Check for common problematic column names
        problematic_columns = []
        sql_lower = sql.lower()
        
        # Check for common timestamp column mistakes
        common_mistakes = [
            "created_ts", "updated_ts", "created_at", "updated_at", 
            "create_time", "update_time", "timestamp", "date_created"
        ]
        
        for mistake in common_mistakes:
            if mistake in sql_lower and mistake not in valid_columns:
                problematic_columns.append(mistake)
        
        if problematic_columns:
            available_time_cols = [col for col in available_columns if any(time_word in col.lower() for time_word in ["time", "date", "_ts", "_at", "open", "close"])]
            error_msg = f"SQL uses non-existent columns: {', '.join(problematic_columns)}. "
            if available_time_cols:
                error_msg += f"Available time/date columns: {', '.join(available_time_cols)}"
            else:
                error_msg += "No time/date columns found in schema."
            
            return {
                "is_valid": False,
                "error": error_msg,
                "warnings": [],
                "missing_columns": problematic_columns,
                "available_columns": available_time_cols if available_time_cols else available_columns
            }
        
        return {
            "is_valid": True,
            "error": None,
            "warnings": [],
            "missing_columns": [],
            "available_columns": available_columns
        }
