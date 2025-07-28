"""
SQL generation with validation and safety checks
"""

from typing import Dict, List, Any
from .validator import SQLValidator
from ..models import ExecutionContext
from app.multimode_llm_client import get_global_llm_client, TaskType
from app.registry import get_enhanced_agent_details_for_llm
from app.sql_prompt_builder import sql_prompt_builder

class SQLGenerator:
    """Generates and validates SQL queries for data agents"""
    
    def __init__(self):
        self.llm_client = get_global_llm_client()
        self.validator = SQLValidator()
    
    async def generate_sql(self, agent_id: str, query: str) -> Dict[str, Any]:
        """Generate SQL for data agent using modular prompt builder with validation"""
        
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
                
                column_info = {
                    "name": col_name,
                    "type": data_type
                }
                
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
        
        # Build optimized prompt with agent-specific guidelines
        prompt = sql_prompt_builder.build_prompt(
            query=query,
            schema=schema_tables,
            database_type=database_type,
            query_type=query_type,
            custom_prompt=custom_prompt
        )
        
        print(f"[SQLGenerator] Query Type: {query_type}, Database: {database_type}")
        if custom_prompt:
            print(f"[SQLGenerator] Using custom prompt guidelines from environment")
        print(f"[SQLGenerator] Optimized Prompt Length: {len(prompt)} chars")
        
        response = self.llm_client.invoke_with_json_response(prompt, task_type=TaskType.SQL_GENERATION)
        
        # Handle the structured response and validate SQL
        if response and response.get("status") == "ready":
            sql = response.get("query", "").strip()
            # Clean SQL and remove any markdown formatting
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            # CRITICAL: Validate SQL for security
            validation_result = self.validator.validate_sql_query(sql)
            
            if not validation_result["is_valid"]:
                print(f"[SQLGenerator] SQL validation failed: {validation_result['error']}")
                return {
                    "sql": None,
                    "status": "validation_failed",
                    "error": validation_result["error"],
                    "output_format": ["table"],
                    "chart_spec": {},
                    "reasoning": f"Generated SQL failed validation: {validation_result['error']}",
                    "query_type": query_type,
                    "validation_warnings": validation_result.get("warnings", [])
                }
            
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
                "validation_warnings": validation_result.get("warnings", [])
            }
            
        elif response and response.get("status") == "needs_clarification":
            # For now, return a simple query if clarification is needed
            clarifications = response.get("clarification_needed", [])
            print(f"[SQLGenerator] Clarification needed: {clarifications}")
            return {
                "sql": None,
                "status": "needs_clarification",
                "output_format": ["table"],
                "chart_spec": {},
                "clarification_needed": clarifications,
                "reasoning": response.get("reasoning", "Need clarification"),
                "query_type": query_type
            }
        else:
            # Cannot proceed - return error
            reasoning = response.get("reasoning", "Unknown reason") if response else "Failed to generate response"
            print(f"[SQLGenerator] Cannot proceed: {reasoning}")
            return {
                "sql": None,
                "status": "cannot_proceed",
                "error": "Failed to generate valid SQL query",
                "output_format": ["table"],
                "chart_spec": {},
                "reasoning": reasoning,
                "query_type": query_type
            }
