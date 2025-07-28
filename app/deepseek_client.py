"""
DeepSeek LLM client for high-performance reasoning tasks.
"""
import re
import json
import os
from typing import Dict, Any, Optional, AsyncIterator
from openai import AsyncOpenAI

class DeepSeekClient:
    """DeepSeek client for advanced reasoning and coding tasks."""
    
    def __init__(self, 
                 model: str = "deepseek-chat", 
                 api_key: Optional[str] = None,
                 think: bool = False,
                 base_url: str = "https://api.deepseek.com"):
        """
        Initialize DeepSeek client.
        
        Args:
            model: DeepSeek model name (deepseek-reasoner, deepseek-coder, deepseek-chat)
            api_key: DeepSeek API key (will use DEEPSEEK_API_KEY env var if not provided)
            base_url: DeepSeek API base URL
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        
        print(f"[DeepSeekClient] Initialized with model: {model}")
    
    async def stream_with_text_response(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        """
        Stream response from DeepSeek with text output.
        
        Args:
            prompt: The main prompt
            context: Additional context to include
            
        Yields:
            String chunks of the response as they arrive
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[DeepSeekClient] Starting streaming call")
        
        try:
            messages = [{"role": "user", "content": full_prompt}]
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                think=False,  # Disable thinking blocks for streaming
                temperature=0.1,
                max_tokens=4000
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"[DeepSeekClient] Streaming call failed: {e}")
            raise Exception(f"DeepSeek streaming call failed: {str(e)}")
    
    async def invoke(self, prompt: str, context: str = "", timeout: int = 600):
        """
        General invoke method for compatibility.
        Returns a response object with a content attribute.
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[DeepSeekClient] General invoke with timeout: {timeout}s")
        
        try:
            messages = [{"role": "user", "content": full_prompt}]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=4000
            )
            
            # Create a response object similar to langchain's format
            class DeepSeekResponse:
                def __init__(self, content: str):
                    self.content = content
            
            return DeepSeekResponse(response.choices[0].message.content)
            
        except Exception as e:
            print(f"[DeepSeekClient] General invoke failed: {e}")
            raise Exception(f"DeepSeek call failed: {str(e)}")

    async def invoke_with_json_response(self, prompt: str, context: str = "", timeout: int = 600) -> Optional[Dict[str, Any]]:
        """
        Invoke DeepSeek with a prompt expecting JSON response.
        
        Args:
            prompt: The main prompt
            context: Additional context to include
            timeout: Timeout in seconds
            
        Returns:
            Parsed JSON response or None if parsing fails
        """
        import time
        import asyncio
        
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[DeepSeekClient] Invoking with JSON response expected, timeout: {timeout}s")
        
        try:
            start_time = time.time()
            messages = [{"role": "user", "content": full_prompt}]
            
            # Add timeout to prevent hanging
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4000
                ),
                timeout=min(timeout, 120)  # Max 2 minutes to prevent very long hangs
            )
            
            elapsed = time.time() - start_time
            print(f"[DeepSeekClient] Response received in {elapsed:.1f}s")
            
            llm_response = response.choices[0].message.content
            
        except Exception as e:
            print(f"[DeepSeekClient] Invocation failed: {e}")
            raise Exception(f"DeepSeek call failed after timeout or error: {str(e)}")
        
        # Truncate very long responses for logging
        response_preview = llm_response[:500] + "..." if len(llm_response) > 500 else llm_response
        print(f"[DeepSeekClient] Response preview:\n{response_preview}")
        
        # Clean response by removing reasoning blocks
        clean_response = re.sub(r'<think>[\s\S]*?</think>', '', llm_response, flags=re.IGNORECASE).strip()
        clean_response = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', clean_response, flags=re.IGNORECASE).strip()
        
        # Remove any leading/trailing non-JSON content before the first {
        json_start = clean_response.find('{')
        json_end = clean_response.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            clean_response = clean_response[json_start:json_end + 1]
        
        print(f"[DeepSeekClient] Cleaned response for JSON parsing:\n{clean_response[:300]}...")
        
        try:
            parsed = json.loads(clean_response)
            print(f"[DeepSeekClient] Successfully parsed JSON response")
            return parsed
        except json.JSONDecodeError as e:
            print(f"[DeepSeekClient] JSON parsing error: {e}")
            print(f"[DeepSeekClient] Problematic content around error: {clean_response[max(0, e.pos-50):e.pos+50]}")
            
            # Try to extract JSON from the response with more aggressive pattern
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested JSON
                r'\{[\s\S]*\}',  # Any content between first { and last }
            ]
            
            for pattern in json_patterns:
                json_matches = re.findall(pattern, clean_response, re.DOTALL)
                for match in json_matches:
                    try:
                        parsed = json.loads(match)
                        print(f"[DeepSeekClient] Successfully extracted and parsed JSON with pattern")
                        return parsed
                    except json.JSONDecodeError:
                        continue
                        
            print(f"[DeepSeekClient] No valid JSON found in response")
            return None
    
    async def invoke_with_text_response(self, prompt: str, context: str = "", allow_diagrams: bool = True) -> str:
        """
        Invoke DeepSeek with a prompt expecting text response.
        
        Args:
            prompt: The main prompt
            context: Additional context to include
            allow_diagrams: Whether to preserve diagrams in the response
            
        Returns:
            Processed text response
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[DeepSeekClient] Text Prompt:\n{full_prompt}")
        
        try:
            messages = [{"role": "user", "content": full_prompt}]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=4000
            )
            
            llm_response = response.choices[0].message.content
            print(f"[DeepSeekClient] Raw Text Response:\n{llm_response}")
            
            if not llm_response or llm_response.strip() == "":
                print("[DeepSeekClient] Warning: Empty response from DeepSeek")
                return "No response generated"
            
            if allow_diagrams:
                # Extract diagrams from reasoning blocks and convert to markdown
                processed_response = self._extract_and_replace_diagrams(llm_response)
            else:
                # Remove reasoning blocks entirely
                processed_response = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', llm_response, flags=re.IGNORECASE).strip()
                processed_response = re.sub(r'<think>[\s\S]*?</think>', '', processed_response, flags=re.IGNORECASE).strip()
            
            print(f"[DeepSeekClient] Processed Text Response:\n{processed_response}")
            return processed_response if processed_response else "No response generated"
            
        except Exception as e:
            print(f"[DeepSeekClient] Error invoking for text response: {e}")
            return f"Error generating response: {str(e)}"
    
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
    
    def create_agent_selection_prompt(self, query: str, agents: Dict[str, Any], history: str = "") -> str:
        """Create a prompt for agent selection."""
        agent_list_str = '\n'.join([f"- {name}: {info['description']} (base: {info['base_domain']})" for name, info in agents.items()])
        
        return f"""You are an expert agent selector with advanced reasoning capabilities. Maintain context across turns.

Conversation history:
{history}

Available agents:
{agent_list_str}

User query: {query}

Think through which agent would be most appropriate for this task. Consider the agent's capabilities, domain expertise, and the specific requirements of the user's query.

Respond ONLY with a valid JSON object, with NO extra text, markdown, or explanation. The JSON must be on the first line of your response.

Example:
{{"agent": "<agent_name>", "reason": "<short explanation>"}}

Now, respond with your selection:
"""
    
    def create_tool_selection_prompt(self, query: str, agent_name: str, agent_info: Dict[str, Any]) -> str:
        """Create a prompt for tool selection."""
        tools = agent_info['tools']
        tool_list_str = '\n'.join([f"- {t['name']}: {t['description']}" for t in tools])
        
        return f"""You are an expert tool selector with advanced reasoning capabilities. The selected agent is: {agent_name} ({agent_info['description']}).

Base domain: {agent_info['base_domain']}

Available tools for this agent:
{tool_list_str}

User query: {query}

Analyze the user query and determine which tool would be most effective. If a tool requires path parameters (e.g., /users/{{id}}), extract the value from the user query and substitute it into the endpoint.

Respond ONLY with a valid JSON object, with NO extra text, markdown, or explanation.
The "tool" value MUST be copied exactly from the tool names in the list above.
The JSON must be on the first line of your response.

Example:
{{"tool": "<tool_name>", "reason": "<short explanation>", "resolved_endpoint": "https://example.com/users/123", "query_params": {{}}, "body_params": {{}}, "headers": {{}}}}

Now, respond with your selection:
"""
    
    def create_final_answer_prompt(self, query: str, call_result: Any) -> str:
        """Create a prompt for final answer formatting."""
        return f"""You are an expert assistant with advanced reasoning capabilities.

User query: {query}

Raw result from the service: {call_result}

Instructions:
- If the user is asking for a list (e.g., 'Give me list of users'), format the result as a markdown table.
- If the answer includes a diagram, output it as a markdown code block with the correct language tag (e.g., ```mermaid for Mermaid diagrams, ```plantuml for PlantUML diagrams).
- Do NOT use <reasoning> or <think> blocks for diagrams.
- Otherwise, summarize or present the result in the most appropriate and helpful way.
- Do not add extra commentary or markdown unless formatting a table or diagram.
"""

    def create_data_answer_prompt(self, query: str, sql_query: str, query_result: Dict[str, Any]) -> str:
        """Create a dynamic prompt for formatting data query results with visualization recommendations."""
        
        # Analyze query type to provide appropriate formatting instructions
        query_lower = query.lower()
        
        # Determine query type and provide specific instructions
        if any(word in query_lower for word in ['inventory', 'stock', 'reorder', 'supplier']):
            specific_instructions = """
- Show specific product names, current stock levels, reorder points, and supplier details
- Format as a table with columns: Product Name, Current Stock, Reorder Level, Store/Location, Supplier
- Highlight items that need immediate attention (low stock, out of stock)
- Include actionable recommendations for specific products
- Recommend bar charts for stock level comparisons"""
            
        elif any(word in query_lower for word in ['sales', 'revenue', 'profit', 'performance']):
            specific_instructions = """
- Show specific sales figures, product names, dates, and performance metrics
- Format as a table with relevant columns: Product/Item, Sales Amount, Quantity Sold, Date/Period
- Include actual numbers and percentages, not generic summaries
- Highlight top performers and trends with specific data points
- Recommend line charts for trends, bar charts for comparisons"""
            
        elif any(word in query_lower for word in ['customer', 'client', 'user']):
            specific_instructions = """
- Show specific customer names, IDs, contact information, and relevant metrics
- Format as a table with columns: Customer Name, ID, Location, Relevant Metrics
- Include actual customer details and behavioral data
- Provide actionable insights about specific customers
- Recommend bar charts for customer metrics, pie charts for distributions"""
            
        elif any(word in query_lower for word in ['employee', 'staff', 'worker']):
            specific_instructions = """
- Show specific employee names, roles, departments, and relevant metrics
- Format as a table with columns: Employee Name, Role, Department, Relevant Data
- Include actual employee information and performance metrics
- Provide specific insights about workforce data
- Recommend pie charts for department distribution, bar charts for performance"""
            
        elif any(word in query_lower for word in ['order', 'purchase', 'transaction']):
            specific_instructions = """
- Show specific order details, dates, amounts, and customer information
- Format as a table with columns: Order ID, Date, Customer, Amount, Status
- Include actual transaction data and order specifics
- Provide insights about specific orders and purchasing patterns
- Recommend line charts for order trends, bar charts for volume comparisons"""
            
        else:
            # Generic instructions for other types of queries
            specific_instructions = """
- Show the actual data with specific details and names from the query results
- Format as a table if multiple rows with relevant columns from the data
- Include specific values, names, IDs, and metrics from the actual results
- Provide actionable insights based on the specific data returned
- Recommend appropriate chart types based on data structure"""
        
        return f"""You are an expert data analyst assistant with advanced reasoning capabilities and visualization expertise.

User query: {query}

SQL Query executed: {sql_query}

Query result: {query_result}

## Response Requirements:

Provide a comprehensive analysis with the following structure:

{{
  "analysis": "Your detailed analysis text with data insights",
  "data_table": "Markdown table with the actual data",
  "key_insights": ["List of specific insights from the data"],
  "output_format": ["list of recommended formats"],
  "chart_spec": {{
    "type": "chart_type",
    "x": "x_axis_field",
    "y": "y_axis_field", 
    "title": "Chart Title",
    "description": "Why this chart is recommended"
  }}
}}

## Chart Type Guidelines:
- **line_chart**: For trends over time, temporal data
- **bar_chart**: For comparisons between categories
- **pie_chart**: For parts of a whole, distributions
- **scatter_plot**: For correlations between variables
- **table**: Always include for detailed data

## Data Analysis Instructions:
- ALWAYS show the actual data with specific details from the query results
- Do NOT provide generic summaries - show the concrete data the user requested
- If the result contains multiple rows, format as a markdown table
- If there's an error in the query result, explain it clearly
- For empty results, explain that no data was found matching the criteria
- Focus on actionable, specific information rather than high-level insights

Query-Specific Instructions:
{specific_instructions}

## Critical Requirements:
1. Show actual data with specific names, values, and details
2. Recommend appropriate visualization formats in "output_format"
3. Provide detailed chart specifications if visualization is recommended
4. Include both analysis text and structured data table
5. Extract meaningful insights from the actual data patterns

CRITICAL: Show actual data with specific names, values, and details - not generic summaries like "100 items" or "various locations"
"""

    async def test_connection(self) -> Dict[str, Any]:
        """Test DeepSeek connection with a simple prompt."""
        try:
            test_prompt = "Say 'Hello, DeepSeek connection is working!' and nothing else."
            response = await self.invoke_with_text_response(test_prompt)
            
            return {
                "success": bool(response),
                "response": response,
                "model": self.model,
                "base_url": self.base_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
                "base_url": self.base_url
            }

    def create_sql_generation_prompt(self, user_query: str, schema_context: str = None, connection_type: str = "unknown", tables_data: list = None, agent_name: str = None, custom_prompt: str = "") -> str:
        """Create a prompt for SQL query generation with DeepSeek's reasoning capabilities."""
        
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

    def create_structured_json_prompt(self, base_prompt: str, json_format: str, examples: str = "") -> str:
        """Create a robust prompt that enforces JSON-only output."""
        prompt = f"""CRITICAL INSTRUCTIONS:
- Respond ONLY with valid JSON
- Do NOT use <reasoning> blocks, explanations, or any text outside the JSON
- Do NOT include markdown code blocks or formatting
- Your entire response must be parseable JSON

{base_prompt}

REQUIRED JSON FORMAT:
{json_format}

{examples}

Remember: Respond with ONLY the JSON object, nothing else."""
        print(f"[DeepSeekClient] Created structured JSON prompt:\n{prompt}")
        return prompt
    
    # Sync versions for compatibility with multi-mode client
    
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
