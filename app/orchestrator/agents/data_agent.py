"""
Data agent execution with SQL validation and safety
"""

from typing import Dict, List, Any
from ..models import AgentResult, ExecutionContext
from ..sql.generator import SQLGenerator
from app.registry.client import get_enhanced_agent_details_for_llm
from app.database.database_query_executor import DatabaseQueryExecutor
from app.llm import MultiModeLLMClient

class DataAgentExecutor:
    """Executes data agents with SQL validation and safety checks"""
    
    def __init__(self):
        self.sql_generator = SQLGenerator()
        self.db_executor = DatabaseQueryExecutor()
        self.llm_client = MultiModeLLMClient()
        self.llm_client = MultiModeLLMClient()
    
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
                    context.session_id,
                    "table_selection", 
                    "table_selection",
                    "📋 Analyzing relevant tables"
                )
            
            relevant_tables = await self._select_relevant_tables(agent_id, query)
            
            if context and hasattr(context, 'workflow_streamer'):
                context.workflow_streamer.emit_step_completed(
                    context.workflow_id, 
                    context.session_id,
                    "table_selection", 
                    "table_selection",
                    1.0
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
                    context.session_id,
                    "sql_generation", 
                    "sql_generation",
                    "🔧 Generating SQL query"
                )
            
            sql_result = await self.sql_generator.generate_sql_with_filtered_tables(agent_id, query, relevant_tables)
            
            if context and hasattr(context, 'workflow_streamer'):
                context.workflow_streamer.emit_step_completed(
                    context.workflow_id, 
                    context.session_id,
                    "sql_generation", 
                    "sql_generation",
                    2.0
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
                        "execution_type": "data",
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
                    metadata={
                        "execution_type": "data",
                        "sql_generation_status": "no_query"
                    },
                    visualization={
                        "output_format": ["table"],
                        "chart_spec": {},
                        "reasoning": "No valid SQL query could be generated"
                    }
                )
            
            print(f"[DataAgentExecutor] Executing SQL query: {sql_query[:200]}...")
            
            # Emit SQL generated event with details
            if context and hasattr(context, 'workflow_streamer') and context.workflow_streamer:
                context.workflow_streamer.emit_sql_generated(
                    context.workflow_id,
                    context.session_id,
                    sql_query,
                    agent_details.get("connection_type", "unknown"),
                    sql_result.get("reasoning", "")
                )
            
            # Get connection config from registry for BigQuery project override
            connection_config = agent_details.get("connection_info", {})
            print(f"[DataAgentExecutor] Using connection config from registry: {connection_config}")
            
            # Emit query execution event
            if context and hasattr(context, 'workflow_streamer') and context.workflow_streamer:
                context.workflow_streamer.emit_query_execution(
                    context.workflow_id,
                    context.session_id,
                    agent_details.get("connection_type", "unknown"),
                    sql_query[:200]
                )
            
            # Execute the validated SQL query
            result = self.db_executor.execute_query(
                agent_details.get("vault_key"),
                agent_details.get("connection_type", "unknown"),
                sql_query,
                connection_config=connection_config
            )
            
            if result.get("status") == "success":
                data = result.get("data", [])
                row_count = len(data)
                
                print(f"[DataAgentExecutor] Query executed successfully: {row_count} rows")
                
                # Emit query results event
                if context and hasattr(context, 'workflow_streamer') and context.workflow_streamer:
                    context.workflow_streamer.emit_query_results(
                        context.workflow_id,
                        context.session_id,
                        agent_details.get("connection_type", "unknown"),
                        row_count
                    )
                
                print(f"[DataAgentExecutor] DEBUG sql_result visualization: {sql_result.get('output_format', 'MISSING')}, chart_spec: {sql_result.get('chart_spec', 'MISSING')}")
                
                # Enhanced chart visualization for trend/chart queries
                output_format = sql_result.get("output_format", ["table"])
                chart_spec = sql_result.get("chart_spec", {})
                
                # Intelligent chart enhancement based on query content and data structure
                if self._should_enhance_with_chart(query, data):
                    if output_format == ["table"]:
                        output_format = ["table", self._determine_best_chart_type(query, data)]
                    
                    if not chart_spec or not chart_spec.get("type"):
                        chart_spec = self._generate_enhanced_chart_spec(query, data, output_format)
                        print(f"[DataAgentExecutor] Enhanced chart spec for query: {chart_spec}")
                
                return AgentResult(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    success=True,
                    data=data,
                    row_count=row_count,
                    query=sql_query,
                    metadata={
                        "execution_type": "data",
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
                        "execution_type": "data",
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
                metadata={
                    "execution_type": "data",
                    "exception": True
                }
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
            response = self.llm_client.invoke_with_json_response(
                prompt=table_selection_prompt,
                task_type="agent_selection",
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

    def _should_enhance_with_chart(self, query: str, data: List[Dict[str, Any]]) -> bool:
        """Determine if query should have chart visualization"""
        if not data or not isinstance(data, list) or len(data) == 0:
            return False
            
        query_lower = query.lower()
        
        # Explicit chart requests
        if any(word in query_lower for word in ["chart", "graph", "plot", "visualize", "show trend"]):
            return True
            
        # Time-based queries
        if any(word in query_lower for word in ["trend", "over time", "monthly", "daily", "yearly", "growth", "change"]):
            return True
            
        # Comparison queries with multiple data points
        if len(data) > 1 and any(word in query_lower for word in ["compare", "top", "best", "worst", "ranking"]):
            return True
            
        # Aggregation queries that benefit from visualization
        if any(word in query_lower for word in ["total", "sum", "count", "average", "by category", "per"]):
            return True
            
        return False
    
    def _determine_best_chart_type(self, query: str, data: List[Dict[str, Any]]) -> str:
        """Determine the best chart type based on query and data structure"""
        query_lower = query.lower()
        
        if not data or len(data) == 0:
            return "bar_chart"
            
        first_record = data[0] if data else {}
        columns = list(first_record.keys()) if isinstance(first_record, dict) else []
        
        # Time series detection
        date_columns = [col for col in columns if any(date_word in col.lower() 
                      for date_word in ["date", "time", "month", "year", "day", "created", "updated"])]
        
        if date_columns or any(word in query_lower for word in ["trend", "over time", "timeline"]):
            return "line_chart"
            
        # Single metric
        if len(data) == 1 or any(word in query_lower for word in ["total", "sum", "count only"]):
            return "metric"
            
        # Distribution/percentage queries
        if any(word in query_lower for word in ["percentage", "share", "distribution", "breakdown"]):
            return "pie_chart"
            
        # Default to bar chart for comparisons
        return "bar_chart"
    
    def _generate_enhanced_chart_spec(self, query: str, data: List[Dict[str, Any]], output_format: List[str]) -> Dict[str, Any]:
        """Generate comprehensive chart specification"""
        if not data or len(data) == 0:
            return {}
            
        first_record = data[0] if data else {}
        columns = list(first_record.keys()) if isinstance(first_record, dict) else []
        
        chart_type = next((fmt for fmt in output_format if fmt != "table"), "bar_chart")
        
        # Smart column detection
        x_column = self._detect_x_axis_column(columns, query)
        y_column = self._detect_y_axis_column(columns, query, x_column)
        
        # Generate contextual title
        title = self._generate_chart_title(query, chart_type, x_column, y_column)
        
        # Determine data formatting
        data_format = self._detect_data_format(y_column, data)
        
        # Choose color scheme based on data type
        color_scheme = self._choose_color_scheme(query, chart_type)
        
        chart_spec = {
            "type": chart_type,
            "x": x_column,
            "y": y_column,
            "title": title,
            "x_label": x_column.replace("_", " ").title(),
            "y_label": y_column.replace("_", " ").title(),
            "color_scheme": color_scheme,
            "data_format": data_format,
            "show_values": True,
            "legend_position": "top" if len(data) > 10 else "right"
        }
        
        # Add aggregation type if applicable
        if any(word in y_column.lower() for word in ["total", "sum", "count"]):
            chart_spec["aggregation_type"] = "sum"
        elif "average" in y_column.lower() or "avg" in y_column.lower():
            chart_spec["aggregation_type"] = "avg"
        else:
            chart_spec["aggregation_type"] = "value"
            
        # Sort order based on chart type and data
        if chart_type in ["bar_chart", "horizontal_bar_chart"]:
            chart_spec["sort_order"] = "desc" if "top" in query.lower() else "asc"
        
        return chart_spec
    
    def _detect_x_axis_column(self, columns: List[str], query: str) -> str:
        """Detect the best column for X-axis"""
        if not columns:
            return "category"
            
        # Look for date/time columns first
        date_columns = [col for col in columns if any(date_word in col.lower() 
                      for date_word in ["date", "time", "month", "year", "day", "created", "updated"])]
        if date_columns:
            return date_columns[0]
            
        # Look for category columns
        category_columns = [col for col in columns if any(cat_word in col.lower() 
                          for cat_word in ["name", "category", "type", "department", "region", "product"])]
        if category_columns:
            return category_columns[0]
            
        # Default to first column
        return columns[0]
    
    def _detect_y_axis_column(self, columns: List[str], query: str, x_column: str) -> str:
        """Detect the best column for Y-axis"""
        if not columns:
            return "value"
            
        # Exclude x_column from consideration
        value_columns = [col for col in columns if col != x_column]
        
        if not value_columns:
            return "value"
            
        # Look for numeric/value columns
        numeric_indicators = ["total", "sum", "count", "amount", "value", "price", "cost", "revenue", "sales"]
        numeric_columns = [col for col in value_columns if any(num_word in col.lower() for num_word in numeric_indicators)]
        
        if numeric_columns:
            return numeric_columns[0]
            
        # Default to last column (often the calculated value)
        return value_columns[-1]
    
    def _generate_chart_title(self, query: str, chart_type: str, x_column: str, y_column: str) -> str:
        """Generate a contextual chart title"""
        query_words = query.lower().split()
        
        # Extract key business terms
        business_terms = []
        for word in query_words:
            if word in ["sales", "revenue", "customers", "orders", "products", "employees", "departments"]:
                business_terms.append(word.title())
        
        if business_terms:
            context = " ".join(business_terms)
        else:
            context = y_column.replace("_", " ").title()
            
        if chart_type == "line_chart":
            return f"{context} Trend Over Time"
        elif chart_type == "bar_chart":
            return f"{context} by {x_column.replace('_', ' ').title()}"
        elif chart_type == "pie_chart":
            return f"{context} Distribution"
        elif chart_type == "metric":
            return f"Total {context}"
        else:
            return f"{context} Analysis"
    
    def _detect_data_format(self, column: str, data: List[Dict[str, Any]]) -> str:
        """Detect appropriate data formatting"""
        column_lower = column.lower()
        
        if any(word in column_lower for word in ["price", "cost", "revenue", "sales", "amount", "value"]):
            return "currency"
        elif any(word in column_lower for word in ["percent", "rate", "ratio"]):
            return "percentage"
        elif any(word in column_lower for word in ["date", "time"]):
            return "date"
        else:
            return "number"
    
    def _choose_color_scheme(self, query: str, chart_type: str) -> str:
        """Choose appropriate color scheme"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["sales", "revenue", "profit", "growth"]):
            return "green"  # Positive/money themes
        elif any(word in query_lower for word in ["trend", "time", "timeline"]):
            return "blue"   # Time-based themes
        elif any(word in query_lower for word in ["warning", "alert", "problem"]):
            return "orange" # Warning themes
        elif chart_type == "pie_chart":
            return "rainbow" # Multi-color for distributions
        else:
            return "blue"   # Default professional blue
