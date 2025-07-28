"""
Data agent execution with SQL validation and safety
"""

from typing import Dict, List, Any
from ..models import AgentResult
from ..sql.generator import SQLGenerator
from app.registry import get_enhanced_agent_details_for_llm
from app.database_query_executor import DatabaseQueryExecutor

class DataAgentExecutor:
    """Executes data agents with SQL validation and safety checks"""
    
    def __init__(self):
        self.sql_generator = SQLGenerator()
        self.db_executor = DatabaseQueryExecutor()
    
    async def execute_data_agent(self, agent_id: str, query: str) -> AgentResult:
        """Execute data agent with comprehensive validation"""
        
        print(f"[DataAgentExecutor] Executing data agent {agent_id}")
        
        try:
            # Get agent details
            agent_details = get_enhanced_agent_details_for_llm(agent_id)
            agent_name = agent_details.get("name", agent_id)
            
            # Generate and validate SQL
            sql_result = await self.sql_generator.generate_sql(agent_id, query)
            
            # Check if SQL generation was successful
            if sql_result.get("status") != "ready":
                error_msg = sql_result.get("error", f"SQL generation failed: {sql_result.get('status')}")
                print(f"[DataAgentExecutor] SQL generation failed: {error_msg}")
                
                return AgentResult(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    success=False,
                    error=error_msg,
                    metadata={
                        "sql_generation_status": sql_result.get("status"),
                        "reasoning": sql_result.get("reasoning", ""),
                        "clarification_needed": sql_result.get("clarification_needed", [])
                    },
                    visualization={
                        "output_format": ["table"],
                        "chart_spec": {},
                        "reasoning": sql_result.get("reasoning", "Query failed")
                    }
                )
            
            sql_query = sql_result.get("sql")
            if not sql_query:
                error_msg = "No SQL query generated"
                print(f"[DataAgentExecutor] {error_msg}")
                
                return AgentResult(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    success=False,
                    error=error_msg,
                    metadata={"sql_generation_status": "no_query"},
                    visualization={
                        "output_format": ["table"],
                        "chart_spec": {},
                        "reasoning": "No valid SQL query could be generated"
                    }
                )
            
            print(f"[DataAgentExecutor] Executing SQL query: {sql_query[:200]}...")
            
            # Execute the validated SQL query
            result = self.db_executor.execute_query(
                agent_details.get("vault_key"),
                agent_details.get("connection_type", "unknown"),
                sql_query
            )
            
            if result.get("status") == "success":
                data = result.get("data", [])
                row_count = len(data)
                
                print(f"[DataAgentExecutor] Query executed successfully: {row_count} rows")
                
                return AgentResult(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    success=True,
                    data=data,
                    row_count=row_count,
                    query=sql_query,
                    metadata={
                        "sql_generation_status": "ready",
                        "validation_warnings": sql_result.get("validation_warnings", []),
                        "query_type": sql_result.get("query_type", "unknown"),
                        "tables_used": sql_result.get("tables_used", [])
                    },
                    visualization={
                        "output_format": sql_result.get("output_format", ["table"]),
                        "chart_spec": sql_result.get("chart_spec", {}),
                        "reasoning": sql_result.get("reasoning", "")
                    }
                )
            else:
                error_msg = result.get("error", "Database execution failed")
                print(f"[DataAgentExecutor] Database execution failed: {error_msg}")
                
                return AgentResult(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    success=False,
                    error=error_msg,
                    query=sql_query,
                    metadata={
                        "sql_generation_status": "ready",
                        "database_error": True,
                        "validation_warnings": sql_result.get("validation_warnings", [])
                    },
                    visualization={
                        "output_format": ["table"],
                        "chart_spec": {},
                        "reasoning": sql_result.get("reasoning", "Query execution failed")
                    }
                )
        
        except Exception as e:
            error_msg = f"Data agent execution error: {str(e)}"
            print(f"[DataAgentExecutor] {error_msg}")
            
            return AgentResult(
                agent_id=agent_id,
                agent_name=agent_details.get("name", agent_id) if 'agent_details' in locals() else agent_id,
                success=False,
                error=error_msg,
                metadata={"exception": True}
            )
