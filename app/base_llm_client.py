"""
Base LLM client with common functionality for all LLM providers.
"""
import re
import json
from typing import Dict, Any, Optional, AsyncIterator
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Base class for all LLM clients with common prompt generation methods."""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    async def invoke_with_json_response(self, prompt: str, context: str = "", timeout: int = 600) -> Optional[Dict[str, Any]]:
        """Invoke with JSON response - must be implemented by subclass."""
        pass
    
    @abstractmethod
    async def invoke_with_text_response(self, prompt: str, context: str = "", allow_diagrams: bool = True) -> str:
        """Invoke with text response - must be implemented by subclass."""
        pass
    
    @abstractmethod
    async def stream_with_text_response(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        """Stream text response - must be implemented by subclass."""
        pass
    
    @abstractmethod
    async def invoke(self, prompt: str, context: str = "", timeout: int = 600):
        """General invoke method - must be implemented by subclass."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection - must be implemented by subclass."""
        pass
    
    # Common prompt generation methods
    def create_agent_selection_prompt(self, query: str, agents: Dict[str, Any], history: str = "") -> str:
        """Create a prompt for agent selection."""
        agent_list_str = '\n'.join([f"- {name}: {info['description']} (base: {info['base_domain']})" for name, info in agents.items()])
        
        return f"""/no_think
You are an expert agent selector. Maintain context across turns.

Conversation history:
{history}

Available agents:
{agent_list_str}

User query: {query}

Respond ONLY with a valid JSON object, with NO extra text, markdown, or explanation. The JSON must be on the first line of your response.

Example:
{{"agent": "<agent_name>", "reason": "<short explanation>"}}

Now, respond with your selection:
"""
    
    def create_tool_selection_prompt(self, query: str, agent_name: str, agent_info: Dict[str, Any]) -> str:
        """Create a prompt for tool selection."""
        tools = agent_info['tools']
        tool_list_str = '\n'.join([f"- {t['name']}: {t['description']}" for t in tools])
        
        return f"""/no_think
You are an expert tool selector. The selected agent is: {agent_name} ({agent_info['description']}).

Base domain: {agent_info['base_domain']}

Available tools for this agent:
{tool_list_str}

User query: {query}

If a tool requires path parameters (e.g., /users/{{id}}), extract the value from the user query and substitute it into the endpoint.

Respond ONLY with a valid JSON object, with NO extra text, markdown, or explanation.
The "tool" value MUST be copied exactly from the tool names in the list above.
The JSON must be on the first line of your response.

Example:
{{"tool": "<tool_name>", "reason": "<short explanation>", "resolved_endpoint": "https://example.com/users/123", "query_params": {{}}, "body_params": {{}}, "headers": {{}}}}

Now, respond with your selection:
"""
    
    def create_final_answer_prompt(self, query: str, call_result: Any) -> str:
        """Create a prompt for final answer formatting."""
        return f"""/no_think
You are an expert assistant.

User query: {query}

Raw result from the service: {call_result}

Instructions:
- If the user is asking for a list (e.g., 'Give me list of users'), format the result as a markdown table.
- If the answer includes a diagram, output it as a markdown code block with the correct language tag (e.g., ```mermaid for Mermaid diagrams, ```plantuml for PlantUML diagrams).
- Do NOT use <think> blocks for diagrams.
- Otherwise, summarize or present the result in the most appropriate and helpful way.
- Do not add extra commentary or markdown unless formatting a table or diagram.
"""

    def create_data_answer_prompt(self, query: str, sql_query: str, query_result: Dict[str, Any]) -> str:
        """Create a dynamic prompt for formatting data query results based on query type."""
        
        # Analyze query type to provide appropriate formatting instructions
        query_lower = query.lower()
        
        # Determine query type and provide specific instructions
        if any(word in query_lower for word in ['inventory', 'stock', 'reorder', 'supplier']):
            specific_instructions = """
- Show specific product names, current stock levels, reorder points, and supplier details
- Format as a table with columns: Product Name, Current Stock, Reorder Level, Store/Location, Supplier
- Highlight items that need immediate attention (low stock, out of stock)
- Include actionable recommendations for specific products"""
            
        elif any(word in query_lower for word in ['sales', 'revenue', 'profit', 'performance']):
            specific_instructions = """
- Show specific sales figures, product names, dates, and performance metrics
- Format as a table with relevant columns: Product/Item, Sales Amount, Quantity Sold, Date/Period
- Include actual numbers and percentages, not generic summaries
- Highlight top performers and trends with specific data points"""
            
        elif any(word in query_lower for word in ['customer', 'client', 'user']):
            specific_instructions = """
- Show specific customer names, IDs, contact information, and relevant metrics
- Format as a table with columns: Customer Name, ID, Location, Relevant Metrics
- Include actual customer details and behavioral data
- Provide actionable insights about specific customers"""
            
        elif any(word in query_lower for word in ['employee', 'staff', 'worker']):
            specific_instructions = """
- Show specific employee names, roles, departments, and relevant metrics
- Format as a table with columns: Employee Name, Role, Department, Relevant Data
- Include actual employee information and performance metrics
- Provide specific insights about workforce data"""
            
        elif any(word in query_lower for word in ['order', 'purchase', 'transaction']):
            specific_instructions = """
- Show specific order details, dates, amounts, and customer information
- Format as a table with columns: Order ID, Date, Customer, Amount, Status
- Include actual transaction data and order specifics
- Provide insights about specific orders and purchasing patterns"""
            
        else:
            # Generic instructions for other types of queries
            specific_instructions = """
- Show the actual data with specific details and names from the query results
- Format as a table if multiple rows with relevant columns from the data
- Include specific values, names, IDs, and metrics from the actual results
- Provide actionable insights based on the specific data returned"""
        
        return f"""/no_think
You are an expert data analyst assistant.

User query: {query}

SQL Query executed: {sql_query}

Query result: {query_result}

General Instructions:
- ALWAYS show the actual data with specific details from the query results
- Do NOT provide generic summaries - show the concrete data the user requested
- If the result contains multiple rows, format as a markdown table
- If there's an error in the query result, explain it clearly
- For empty results, explain that no data was found matching the criteria
- Focus on actionable, specific information rather than high-level insights
- Do not include the raw SQL query in your response unless specifically asked

Query-Specific Instructions:
{specific_instructions}

CRITICAL: Show actual data with specific names, values, and details - not generic summaries like "100 items" or "various locations"
"""

    def create_sql_generation_prompt(self, user_query: str, schema_context: str = None, connection_type: str = "unknown", tables_data: list = None, agent_name: str = None, custom_prompt: str = "") -> str:
        """Create a prompt for SQL query generation."""
        
        # Use structured data if available, otherwise use provided schema context
        if tables_data:
            schema_context = self._build_schema_context_from_structured_data(tables_data)
        
        # Get database-specific syntax guidance
        syntax_guidance = self._get_database_syntax_guidance(connection_type)

        # Generate location filtering guidance based on agent name
        location_guidance = self._get_location_filtering_guidance(agent_name, user_query)
        
        # Format custom prompt section if provided
        custom_prompt_section = ""
        if custom_prompt and custom_prompt.strip():
            custom_prompt_section = f"\nAGENT-SPECIFIC GUIDELINES:\n{custom_prompt.strip()}\n"
        
        return f"""You are an expert SQL developer with advanced reasoning capabilities. Generate a SQL query to answer the user's request.

USER REQUEST: {user_query}

DATABASE TYPE: {connection_type}

{syntax_guidance}

{location_guidance}
{custom_prompt_section}
SCHEMA DEFINITION:
{schema_context}

Think through this step by step:
1. Analyze what the user is asking for
2. Identify which tables and columns from the schema are needed
3. Consider any joins, filters, or aggregations required
4. Ensure the syntax matches the database type

STRICT RULES:
- Use ONLY tables and columns that exist in the schema above
- Do NOT invent or assume any table or column names
- Use exact table and column names as shown in the schema
- Use schema-qualified table names (e.g., schema.table_name)
- If the requested tables don't exist, create a query using available tables that best answers the user's intent

CRITICAL: You MUST respond with EXACTLY this JSON format, no other format is acceptable:

{{
    "thought": "Brief analysis of the request and which tables/columns from the schema I'll use",
    "query": "The SQL query using only schema tables and columns"
}}

If tables mentioned in the user query don't exist:
- Look for similar tables in the schema that could answer the query
- Use your best judgment to map the user's intent to available tables
- Provide a working SQL query using existing tables
- Explain your mapping decision in the "thought" field

Do NOT respond with status, reasoning, output_format, chart_spec, or any other fields.
Do NOT say the tables don't exist - instead find the best alternative from available tables.

Respond with ONLY the JSON:"""

    def create_structured_json_prompt(self, base_prompt: str, json_format: str, examples: str = "") -> str:
        """Create a robust prompt that enforces JSON-only output."""
        prompt = f"""CRITICAL INSTRUCTIONS:
- Respond ONLY with valid JSON
- Do NOT use <think> blocks, explanations, or any text outside the JSON
- Do NOT include markdown code blocks or formatting
- Your entire response must be parseable JSON

{base_prompt}

REQUIRED JSON FORMAT:
{json_format}

{examples}

Remember: Respond with ONLY the JSON object, nothing else."""
        print(f"[BaseLLMClient] Created structured JSON prompt:\n{prompt}")
        return prompt

    def _get_location_filtering_guidance(self, agent_name: str, user_query: str) -> str:
        """Generate flexible location context guidance for the LLM to make intelligent decisions."""
        if not agent_name:
            return ""
        
        return f"""
AGENT CONTEXT AWARENESS:
🤖 AGENT: {agent_name}
📍 CONTEXT: Consider whether this agent name suggests a specific geographic scope or region
💭 INTELLIGENT DECISION: Analyze if the agent's scope already filters data by location/region
⚖️ BALANCE: Decide whether additional location filters in the SQL would be redundant or necessary

Guidelines for your consideration:
- If the agent name contains geographic indicators (country, region, city), the data may already be filtered
- If the user mentions the same location as suggested by the agent name, additional filtering might be redundant
- If the user mentions a different location than the agent's apparent scope, location filtering may be appropriate
- Use your judgment based on the schema, agent context, and user intent
- Focus on generating the most effective query for the user's specific request
"""

    def _get_database_syntax_guidance(self, connection_type: str) -> str:
        """Get database-specific SQL syntax guidance."""
        connection_type_lower = connection_type.lower() if connection_type else ""
        
        if "mssql" in connection_type_lower or "sql server" in connection_type_lower or "sqlserver" in connection_type_lower:
            return """
DATABASE-SPECIFIC SYNTAX REQUIREMENTS:
⚠️ SQL SERVER: Use TOP instead of LIMIT
- Correct: SELECT TOP 10 * FROM table_name
- Incorrect: SELECT * FROM table_name LIMIT 10
- Use square brackets for reserved words: [order], [user], etc.
- Use GETDATE() for current timestamp
- String concatenation: + operator or CONCAT() function
"""
        elif "postgres" in connection_type_lower or "postgresql" in connection_type_lower:
            return """
DATABASE-SPECIFIC SYNTAX REQUIREMENTS:
✅ POSTGRESQL: Use LIMIT for row limiting
- Correct: SELECT * FROM table_name LIMIT 10
- Use double quotes for case-sensitive identifiers
- Use NOW() for current timestamp
- String concatenation: || operator or CONCAT() function
"""
        elif "mysql" in connection_type_lower:
            return """
DATABASE-SPECIFIC SYNTAX REQUIREMENTS:
✅ MYSQL: Use LIMIT for row limiting
- Correct: SELECT * FROM table_name LIMIT 10
- Use backticks for reserved words: `order`, `user`, etc.
- Use NOW() for current timestamp
- String concatenation: CONCAT() function
"""
        else:
            return """
DATABASE-SPECIFIC SYNTAX REQUIREMENTS:
⚠️ UNKNOWN DATABASE TYPE: Use standard SQL practices
- For row limiting, check if database supports LIMIT or TOP
- Be careful with reserved words and use appropriate quoting
"""

    def _build_schema_context_from_structured_data(self, tables_data: list) -> str:
        """Build a clean schema context from structured table data."""
        context_parts = []
        
        for table in tables_data:
            table_name = table.get("tableName", "")
            schema_name = table.get("schemaName", "public")
            description = table.get("description", "")
            row_count = table.get("rowCount", 0)
            columns = table.get("columns", [])
            
            if table_name:
                full_table_name = f"{schema_name}.{table_name}" if schema_name != "public" else table_name
                context_parts.append(f"Table: {full_table_name}")
                
                if description:
                    # Truncate long descriptions
                    desc_preview = description[:200] + "..." if len(description) > 200 else description
                    context_parts.append(f"  Description: {desc_preview}")
                
                context_parts.append(f"  Row Count: {row_count}")
                context_parts.append("  Columns:")
                
                for col in columns:
                    col_name = col.get("columnName", "")
                    data_type = col.get("dataType", "")
                    nullable = "NULL" if col.get("isNullable", True) else "NOT NULL"
                    
                    if col.get("isPrimaryKey", False):
                        context_parts.append(f"    - {col_name}: {data_type} PRIMARY KEY")
                    elif col.get("isForeignKey", False):
                        ref_table = col.get("referencedTable", "")
                        context_parts.append(f"    - {col_name}: {data_type} FK -> {ref_table}")
                    else:
                        context_parts.append(f"    - {col_name}: {data_type} {nullable}")
                
                context_parts.append("")
        
        return "\n".join(context_parts)

    # Helper methods for diagram processing
    def _extract_and_replace_diagrams(self, text: str) -> str:
        """Extract diagrams from reasoning blocks and convert to markdown code blocks."""
        def repl(match):
            content = match.group(1)
            # Try to detect mermaid or plantuml
            if 'mermaid' in content.lower():
                return f"```mermaid\n{content}\n```"
            elif '@startuml' in content.lower():
                return f"```plantuml\n{content}\n```"
            elif 'graph' in content.lower() and ('-->' in content or '->' in content):
                return f"```mermaid\n{content}\n```"
            else:
                return content  # fallback: just insert as-is
        
        # Replace <reasoning>...</reasoning> and <think>...</think> with code blocks if diagram detected
        text = re.sub(r'<reasoning>([\s\S]*?)</reasoning>', repl, text, flags=re.IGNORECASE)
        text = re.sub(r'<think>([\s\S]*?)</think>', repl, text, flags=re.IGNORECASE)
        return text

    # Synchronous wrapper methods for compatibility
    def invoke_with_json_response_sync(self, prompt: str, context: str = "", timeout: int = 600) -> Optional[Dict[str, Any]]:
        """Synchronous version of invoke_with_json_response"""
        import asyncio
        import concurrent.futures
        
        # If we're in an async context, use thread executor
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._sync_invoke_json, prompt, context, timeout)
                return future.result(timeout=timeout)
        except RuntimeError:
            # No running loop, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.invoke_with_json_response(prompt, context, timeout))
            finally:
                loop.close()
    
    def _sync_invoke_json(self, prompt: str, context: str = "", timeout: int = 600) -> Optional[Dict[str, Any]]:
        """Helper method that runs async method in new loop"""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.invoke_with_json_response(prompt, context, timeout))
        finally:
            loop.close()
    
    def invoke_with_text_response_sync(self, prompt: str, context: str = "", allow_diagrams: bool = True) -> str:
        """Synchronous version of invoke_with_text_response"""
        import asyncio
        import concurrent.futures
        
        # If we're in an async context, use thread executor
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._sync_invoke_text, prompt, context, allow_diagrams)
                return future.result(timeout=300)
        except RuntimeError:
            # No running loop, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.invoke_with_text_response(prompt, context, allow_diagrams))
            finally:
                loop.close()
    
    def _sync_invoke_text(self, prompt: str, context: str = "", allow_diagrams: bool = True) -> str:
        """Helper method that runs async method in new loop"""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.invoke_with_text_response(prompt, context, allow_diagrams))
        finally:
            loop.close()
    
    def invoke_sync(self, prompt: str, context: str = "", timeout: int = 600):
        """Synchronous version of invoke"""
        import asyncio
        import concurrent.futures
        
        # If we're in an async context, use thread executor
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._sync_invoke, prompt, context, timeout)
                return future.result(timeout=timeout)
        except RuntimeError:
            # No running loop, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.invoke(prompt, context, timeout))
            finally:
                loop.close()
    
    def _sync_invoke(self, prompt: str, context: str = "", timeout: int = 600):
        """Helper method that runs async method in new loop"""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.invoke(prompt, context, timeout))
        finally:
            loop.close()
    
    def test_connection_sync(self) -> Dict[str, Any]:
        """Synchronous version of test_connection"""
        import asyncio
        import concurrent.futures
        
        # If we're in an async context, use thread executor
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._sync_test_connection)
                return future.result(timeout=30)
        except RuntimeError:
            # No running loop, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.test_connection())
            finally:
                loop.close()
    
    def _sync_test_connection(self) -> Dict[str, Any]:
        """Helper method that runs async method in new loop"""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.test_connection())
        finally:
            loop.close()
