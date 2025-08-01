"""
Data agent execution with SQL validation and safety
"""

from typing import Dict, List, Any
from ..models import AgentResult, ExecutionContext
from ..sql.generator import SQLGenerator
from app.registry import get_enhanced_agent_details_for_llm
from app.database_query_executor import DatabaseQueryExecutor
from app.multimode_llm_client import get_global_llm_client, TaskType

class DataAgentExecutor:
    """Executes data agents with SQL validation and safety checks"""
    
    def __init__(self):
        self.sql_generator = SQLGenerator()
        self.db_executor = DatabaseQueryExecutor()
    
    async def execute_data_agent(self, agent_id: str, query: str, context: ExecutionContext = None) -> AgentResult:
        """Execute data agent with comprehensive validation using two-phase approach"""
        
        print(f"[DataAgentExecutor] Executing data agent {agent_id}")
        
        try:
            # Get agent details
            agent_details = get_enhanced_agent_details_for_llm(agent_id)
            agent_name = agent_details.get("name", agent_id)
            
            # Phase 1: Table Selection
            print(f"[DataAgentExecutor] Phase 1: Analyzing query and selecting relevant tables")
            
            # Emit step for table selection if context is available
            if context and hasattr(context, 'workflow_streamer'):
                context.workflow_streamer.emit_step_started(
                    context.workflow_id, 
                    "table_selection", 
                    "📋 Analyzing relevant tables", 
                    context.session_id
                )
            
            relevant_tables = await self._select_relevant_tables(agent_id, query)
            
            if context and hasattr(context, 'workflow_streamer'):
                context.workflow_streamer.emit_step_completed(
                    context.workflow_id, 
                    "table_selection", 
                    context.session_id,
                    execution_time=1.0
                )
            
            if not relevant_tables:
                return AgentResult(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    success=False,
                    error="No relevant tables found for the query",
                    metadata={"phase": "table_selection"}
                )
            
            # Phase 2: SQL Generation with filtered tables
            print(f"[DataAgentExecutor] Phase 2: Generating SQL using {len(relevant_tables)} relevant tables")
            
            # Emit step for SQL generation
            if context and hasattr(context, 'workflow_streamer'):
                context.workflow_streamer.emit_step_started(
                    context.workflow_id, 
                    "sql_generation", 
                    "🔧 Generating SQL query", 
                    context.session_id
                )
            
            sql_result = await self.sql_generator.generate_sql_with_filtered_tables(agent_id, query, relevant_tables)
            
            if context and hasattr(context, 'workflow_streamer'):
                context.workflow_streamer.emit_step_completed(
                    context.workflow_id, 
                    "sql_generation", 
                    context.session_id,
                    execution_time=2.0
                )
            
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
                        "clarification_needed": sql_result.get("clarification_needed", []),
                        "relevant_tables": [t.get("tableName") for t in relevant_tables],
                        "phase": "sql_generation"
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
                print(f"[DataAgentExecutor] DEBUG sql_result visualization: {sql_result.get('output_format', 'MISSING')}, chart_spec: {sql_result.get('chart_spec', 'MISSING')}")
                
                # Ensure chart visualization for trend/chart queries if LLM didn't provide it
                output_format = sql_result.get("output_format", ["table"])
                chart_spec = sql_result.get("chart_spec", {})
                
                if ("chart" in query.lower() or "trend" in query.lower()) and output_format == ["table"]:
                    output_format = ["table", "line_chart"]
                    chart_spec = {
                        "type": "line_chart",
                        "x": "month" if "month" in str(data[0].keys()).lower() else list(data[0].keys())[0] if data else "x",
                        "y": "total_sales" if "total_sales" in str(data[0].keys()).lower() else list(data[0].keys())[-1] if data else "y",
                        "title": "Sales Trends Over Time"
                    }
                    print(f"[DataAgentExecutor] Enhanced chart spec for trend query: {output_format}, {chart_spec}")
                
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
                        "output_format": output_format,
                        "chart_spec": chart_spec,
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

    async def _select_relevant_tables(self, agent_id: str, query: str) -> List[Dict[str, Any]]:
        """Phase 1: Use LLM to select relevant tables for the query"""
        
        try:
            # Get all available tables
            agent_details = get_enhanced_agent_details_for_llm(agent_id)
            all_tables = agent_details.get("tables_with_columns", [])
            
            if not all_tables:
                print(f"[DataAgentExecutor] No tables available for agent {agent_id}")
                return []
            
            # Create a simplified table overview for LLM analysis
            table_overview = []
            for table in all_tables:
                table_name = table.get('tableName', '')
                schema_name = table.get('schemaName', 'dbo')
                full_table_name = f"{schema_name}.{table_name}" if schema_name != "public" else table_name
                
                # Get key column info
                columns = table.get("columns", [])
                key_columns = []
                for col in columns[:5]:  # Show first 5 columns as preview
                    col_name = col.get('columnName', '')
                    data_type = col.get('dataType', '')
                    key_columns.append(f"{col_name} ({data_type})")
                
                table_overview.append({
                    "table_name": full_table_name,
                    "description": table.get('description', 'No description'),
                    "row_count": table.get('rowCount', 0),
                    "sample_columns": ", ".join(key_columns),
                    "total_columns": len(columns)
                })
            
            # Create prompt for table selection
            table_selection_prompt = self._create_table_selection_prompt(query, table_overview)
            
            # Get LLM client and ask for table selection
            llm_client = get_global_llm_client()
            response = llm_client.invoke_with_json_response(
                prompt=table_selection_prompt,
                task_type=TaskType.AGENT_SELECTION,
                timeout=30
            )
            
            if not response:
                print(f"[DataAgentExecutor] LLM failed to select tables, using all tables")
                return all_tables
            
            # Extract selected table names
            selected_table_names = response.get("selected_tables", [])
            reasoning = response.get("reasoning", "No reasoning provided")
            
            print(f"[DataAgentExecutor] LLM selected {len(selected_table_names)} tables: {selected_table_names}")
            print(f"[DataAgentExecutor] Selection reasoning: {reasoning}")
            
            # Filter the original tables based on LLM selection
            relevant_tables = []
            for table in all_tables:
                table_name = table.get('tableName', '')
                schema_name = table.get('schemaName', 'dbo')
                full_table_name = f"{schema_name}.{table_name}" if schema_name != "public" else table_name
                
                if full_table_name in selected_table_names or table_name in selected_table_names:
                    relevant_tables.append(table)
            
            # Fallback: if no tables selected or selection failed, use all tables
            if not relevant_tables:
                print(f"[DataAgentExecutor] No matching tables found, using all tables as fallback")
                return all_tables
            
            return relevant_tables
            
        except Exception as e:
            print(f"[DataAgentExecutor] Table selection failed: {e}, using all tables")
            # Fallback to all tables if selection fails
            agent_details = get_enhanced_agent_details_for_llm(agent_id)
            return agent_details.get("tables_with_columns", [])
    
    def _create_table_selection_prompt(self, user_query: str, table_overview: List[Dict[str, Any]]) -> str:
        """Create prompt for LLM to select relevant tables"""
        
        tables_info = []
        for table in table_overview:
            tables_info.append(
                f"- **{table['table_name']}**: {table['description']} "
                f"({table['row_count']} rows, {table['total_columns']} columns) "
                f"Sample columns: {table['sample_columns']}"
            )
        
        tables_list = "\n".join(tables_info)
        
        return f"""You are a data analyst helping to identify which database tables are relevant for answering a user's question.

USER QUESTION: {user_query}

AVAILABLE TABLES:
{tables_list}

Analyze the user's question and identify which tables would be most relevant for answering it. Consider:
1. The type of data the user is asking about
2. The table names and descriptions
3. The sample columns shown
4. Potential relationships between tables

Respond with a JSON object in this exact format:
{{
    "reasoning": "Brief explanation of why these tables were selected",
    "selected_tables": ["table1", "table2", "table3"],
    "confidence": 0.8
}}

Important:
- Only include tables that are directly relevant to answering the question
- Use the exact table names from the list above
- If unsure, include tables that might be related rather than missing important data
- Minimum 1 table, maximum 5 tables for optimal performance"""
