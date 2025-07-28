"""
LLM client wrapper for consistent LLM interactions.
"""
from langchain_ollama import ChatOllama
import re
import json
from typing import Dict, Any, Optional

class LLMClient:
    """Wrapper for LLM interactions with consistent prompting and parsing."""
    
    def __init__(self, model: str = "qwen2.5:32b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.ollama = ChatOllama(
            model=model, 
            base_url=base_url, 
            keep_alive="10m",
            think=False,  # Disable <think> blocks globally
            timeout=600,  # 60 second timeout
            streaming=True  # Enable streaming by default
        )
    
    async def stream_with_text_response(self, prompt: str, context: str = ""):
        """
        Stream response from LLM with text output.
        
        Args:
            prompt: The main prompt
            context: Additional context to include
            
        Yields:
            String chunks of the response as they arrive
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[LLMClient] Starting streaming LLM call")
        
        try:
            # Use astream for streaming response
            async for chunk in self.ollama.astream(full_prompt):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
        except Exception as e:
            print(f"[LLMClient] Streaming LLM call failed: {e}")
            raise Exception(f"Streaming LLM call failed: {str(e)}")
    
    def invoke(self, prompt: str, context: str = "", timeout: int = 600):
        """
        General invoke method for compatibility.
        Returns a response object with a content attribute.
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[LLMClient] General invoke with timeout: {timeout}s")
        
        try:
            response = self.ollama.invoke(full_prompt, think=False)
            return response
        except Exception as e:
            print(f"[LLMClient] General invoke failed: {e}")
            raise Exception(f"LLM call failed: {str(e)}")

    def invoke_with_json_response(self, prompt: str, context: str = "", timeout: int = 600) -> Optional[Dict[str, Any]]:
        """
        Invoke LLM with a prompt expecting JSON response.
        
        Args:
            prompt: The main prompt
            context: Additional context to include
            timeout: Timeout in seconds (default 600 = 10 minutes)
            
        Returns:
            Parsed JSON response or None if parsing fails
        """
        import time
        
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[LLMClient] Invoking LLM with timeout: {timeout}s")
        
        try:
            start_time = time.time()
            response = self.ollama.invoke(full_prompt, think=False)
            elapsed = time.time() - start_time
            print(f"[LLMClient] LLM response received in {elapsed:.1f}s")
            
        except Exception as e:
            print(f"[LLMClient] LLM invocation failed: {e}")
            raise Exception(f"LLM call failed after timeout or error: {str(e)}")
        
        llm_response = response.content if hasattr(response, 'content') else str(response)
        
        # Truncate very long responses for logging
        response_preview = llm_response[:500] + "..." if len(llm_response) > 500 else llm_response
        print(f"[LLMClient] Response preview:\n{response_preview}")
        
        # Clean response by removing <think> blocks more aggressively
        clean_response = re.sub(r'<think>[\s\S]*?</think>', '', llm_response, flags=re.IGNORECASE).strip()
        
        # Also remove any remaining think-like patterns
        clean_response = re.sub(r'</?think[^>]*>', '', clean_response, flags=re.IGNORECASE)
        
        # Remove JavaScript-style comments that break JSON parsing
        # Remove single-line comments (// comment) but not URLs (http://)
        clean_response = re.sub(r'(?<!:)//[^\r\n]*', '', clean_response)
        
        # Remove multi-line comments (/* comment */)
        clean_response = re.sub(r'/\*[\s\S]*?\*/', '', clean_response)
        
        # Remove trailing commas that might break JSON
        clean_response = re.sub(r',\s*([}\]])', r'\1', clean_response)
        
        # Remove any leading/trailing non-JSON content before the first {
        json_start = clean_response.find('{')
        json_end = clean_response.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            clean_response = clean_response[json_start:json_end + 1]
        
        print(f"[LLMClient] Cleaned response for JSON parsing:\n{clean_response[:300]}...")
        
        try:
            parsed = json.loads(clean_response)
            print(f"[LLMClient] Successfully parsed JSON response")
            return parsed
        except json.JSONDecodeError as e:
            print(f"[LLMClient] JSON parsing error: {e}")
            print(f"[LLMClient] Problematic content around error: {clean_response[max(0, e.pos-50):e.pos+50]}")
            
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
                        print(f"[LLMClient] Successfully extracted and parsed JSON with pattern")
                        return parsed
                    except json.JSONDecodeError:
                        continue
                        
            print(f"[LLMClient] No valid JSON found in response")
            return None
    
    def invoke_with_text_response(self, prompt: str, context: str = "", allow_diagrams: bool = True) -> str:
        """
        Invoke LLM with a prompt expecting text response.
        
        Args:
            prompt: The main prompt
            context: Additional context to include
            allow_diagrams: Whether to preserve diagrams in the response
            
        Returns:
            Processed text response
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[LLMClient] Text Prompt:\n{full_prompt}")
        
        try:
            response = self.ollama.invoke(full_prompt, think=False)
            llm_response = response.content if hasattr(response, 'content') else str(response)
            print(f"[LLMClient] Raw Text Response:\n{llm_response}")
            
            if not llm_response or llm_response.strip() == "":
                print("[LLMClient] Warning: Empty response from LLM")
                return "No response generated"
            
            if allow_diagrams:
                # Extract diagrams from <think> blocks and convert to markdown
                processed_response = self._extract_and_replace_diagrams(llm_response)
            else:
                # Remove <think> blocks entirely
                processed_response = re.sub(r'<think>[\s\S]*?</think>', '', llm_response, flags=re.IGNORECASE).strip()
            
            print(f"[LLMClient] Processed Text Response:\n{processed_response}")
            return processed_response if processed_response else "No response generated"
            
        except Exception as e:
            print(f"[LLMClient] Error invoking LLM for text response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _extract_and_replace_diagrams(self, text: str) -> str:
        """Extract diagrams from <think> blocks and convert to markdown code blocks."""
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
        
        # Replace <think>...</think> with code blocks if diagram detected
        return re.sub(r'<think>([\s\S]*?)</think>', repl, text, flags=re.IGNORECASE)
    
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

    def test_connection(self) -> Dict[str, Any]:
        """Test LLM connection with a simple prompt."""
        try:
            test_prompt = "Say 'Hello, LLM connection is working!' and nothing else."
            response = self.invoke_with_text_response(test_prompt)
            
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

    def _extract_column_reference_from_structured_data(self, tables_data: list) -> str:
        """Extract and format exact column names from structured table data."""
        try:
            column_reference = "📋 EXACT COLUMN NAMES (use these EXACT names only):\n\n"
            
            print(f"[DEBUG] LLMClient: Processing {len(tables_data)} tables for column reference")
            
            for table in tables_data:
                table_name = table.get("tableName", "")
                schema_name = table.get("schemaName", "public")
                columns = table.get("columns", [])
                
                print(f"[DEBUG] LLMClient: Table '{table_name}' has {len(columns)} columns")
                
                if table_name and columns:
                    full_table_name = f"{schema_name}.{table_name}" if schema_name != "public" else table_name
                    column_reference += f"🏷️  {full_table_name}:\n"
                    
                    for col in columns:
                        col_name = col.get("columnName", "")
                        if col_name:
                            column_reference += f"     • {col_name}\n"
                            print(f"[DEBUG] LLMClient:   Added column: {col_name}")
                    
                    column_reference += "\n"
            
            # Add generic reminders
            column_reference += "⚠️  CRITICAL REMINDERS:\n"
            column_reference += "   • Use ONLY the column names listed above\n"
            column_reference += "   • DO NOT assume or invent column names\n"
            column_reference += "   • If you don't see a column name you expect, it doesn't exist\n"
            column_reference += "   • Use schema-qualified table names (e.g., schema.table_name)\n"
            column_reference += "   • Check the exact column names from the list above before writing queries\n"
            
            print(f"[DEBUG] LLMClient: Generated column reference ({len(column_reference)} chars):")
            print("=" * 50)
            print(column_reference)
            print("=" * 50)
            
            return column_reference
            
        except Exception as e:
            print(f"[LLMClient] Error processing structured table data: {e}")
            return """🔍 Use ONLY the exact column names shown in the schema.
⚠️  CRITICAL: 
  • DO NOT assume standard column names
  • DO NOT use variations of column names  
  • Use exact column names from schema only
⚠️  Verify every column name against the schema!"""

    def _extract_column_reference(self, schema_context: str) -> str:
        """Extract and format exact column names from schema context for strict validation."""
        try:
            column_reference = "📋 EXACT COLUMN NAMES (use these EXACT names only):\n\n"
            
            # Parse the schema format: "-- Table: schema.tablename" followed by "-- Key Columns:"
            lines = schema_context.split('\n')
            current_table = None
            in_columns_section = False
            
            for line in lines:
                line = line.strip()
                
                # Detect table definition
                if line.startswith('-- Table:'):
                    table_full_name = line.replace('-- Table:', '').strip()
                    if '.' in table_full_name:
                        current_table = table_full_name.split('.')[-1]  # Get just the table name
                    else:
                        current_table = table_full_name
                    column_reference += f"🏷️  {current_table}:\n"
                    in_columns_section = False
                
                # Detect columns section
                elif line.startswith('-- Key Columns:'):
                    in_columns_section = True
                
                # Extract column names from column definitions
                elif in_columns_section and line.startswith('--   '):
                    # Format: "--   column_name: datatype [PRIMARY KEY|FK -> table|NULL|NOT NULL]"
                    col_line = line.replace('--   ', '').strip()
                    if ':' in col_line:
                        col_name = col_line.split(':')[0].strip()
                        if col_name and col_name not in ['', 'Description', 'Row Count']:
                            column_reference += f"     • {col_name}\n"
            
            # Add generic reminders without hard-coding table names
            column_reference += "\n⚠️  CRITICAL REMINDERS:\n"
            column_reference += "   • Use ONLY the column names listed above\n"
            column_reference += "   • DO NOT assume or invent column names\n"
            column_reference += "   • If you don't see a column name you expect, it doesn't exist\n"
            column_reference += "   • Check the exact column names from the schema before writing queries\n"
            
            return column_reference
            
        except Exception as e:
            print(f"[LLMClient] Error parsing schema for column reference: {e}")
            # Fallback if parsing fails
            return """🔍 Use ONLY the exact column names shown in the schema above.
⚠️  CRITICAL: 
  • DO NOT assume standard column names
  • DO NOT use variations of column names  
  • Use exact column names from schema only
⚠️  Verify every column name against the schema!"""

    def _get_database_syntax_guidance(self, connection_type: str) -> str:
        """Get database-specific SQL syntax guidance."""
        connection_type_lower = connection_type.lower() if connection_type else ""
        
        print(f"[LLMClient._get_database_syntax_guidance] 🔍 CONNECTION TYPE DEBUG:")
        print(f"[LLMClient._get_database_syntax_guidance] Input connection_type: '{connection_type}'")
        print(f"[LLMClient._get_database_syntax_guidance] Lowercase: '{connection_type_lower}'")
        
        if "mssql" in connection_type_lower or "sql server" in connection_type_lower or "sqlserver" in connection_type_lower:
            guidance = """
DATABASE-SPECIFIC SYNTAX REQUIREMENTS:
⚠️ SQL SERVER: Use TOP instead of LIMIT
- Correct: SELECT TOP 10 * FROM table_name
- Incorrect: SELECT * FROM table_name LIMIT 10
- Use square brackets for reserved words: [order], [user], etc.
- Use GETDATE() for current timestamp
- String concatenation: + operator or CONCAT() function
"""
            print(f"[LLMClient._get_database_syntax_guidance] ✅ DETECTED SQL SERVER - returning TOP guidance")
            return guidance
            
        elif "postgres" in connection_type_lower or "postgresql" in connection_type_lower:
            guidance = """
DATABASE-SPECIFIC SYNTAX REQUIREMENTS:
✅ POSTGRESQL: Use LIMIT for row limiting
- Correct: SELECT * FROM table_name LIMIT 10
- Use double quotes for case-sensitive identifiers
- Use NOW() for current timestamp
- String concatenation: || operator or CONCAT() function
"""
            print(f"[LLMClient._get_database_syntax_guidance] ✅ DETECTED POSTGRESQL - returning LIMIT guidance")
            return guidance
            
        elif "mysql" in connection_type_lower:
            guidance = """
DATABASE-SPECIFIC SYNTAX REQUIREMENTS:
✅ MYSQL: Use LIMIT for row limiting
- Correct: SELECT * FROM table_name LIMIT 10
- Use backticks for reserved words: `order`, `user`, etc.
- Use NOW() for current timestamp
- String concatenation: CONCAT() function
"""
            print(f"[LLMClient._get_database_syntax_guidance] ✅ DETECTED MYSQL - returning LIMIT guidance")
            return guidance
        else:
            guidance = """
DATABASE-SPECIFIC SYNTAX REQUIREMENTS:
⚠️ UNKNOWN DATABASE TYPE: Use standard SQL practices
- For row limiting, check if database supports LIMIT or TOP
- Be careful with reserved words and use appropriate quoting
"""
            print(f"[LLMClient._get_database_syntax_guidance] ⚠️ UNKNOWN DATABASE TYPE - returning generic guidance")
            return guidance

    def create_sql_generation_prompt(self, user_query: str, schema_context: str = None, connection_type: str = "unknown", tables_data: list = None, agent_name: str = None) -> str:
        """Create a simple prompt for SQL query generation with structured JSON output."""
        
        # Use structured data if available, otherwise use provided schema context
        if tables_data:
            schema_context = self._build_schema_context_from_structured_data(tables_data)
        
        # Get database-specific syntax guidance
        syntax_guidance = self._get_database_syntax_guidance(connection_type)
        
        return f"""You are an expert SQL developer. Generate a SQL query to answer the user's request.

USER REQUEST: {user_query}

DATABASE TYPE: {connection_type}

{syntax_guidance}

SCHEMA DEFINITION:
{schema_context}

STRICT RULES:
- Use ONLY tables and columns that exist in the schema above
- Do NOT invent or assume any table or column names
- Use exact table and column names as shown in the schema
- Use schema-qualified table names (e.g., schema.table_name)

REQUIRED JSON FORMAT:
{{
    "thought": "Brief analysis of the request and which tables/columns from the schema I'll use",
    "query": "The SQL query using only schema tables and columns"
}}

Respond with ONLY the JSON:"""

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
- Do NOT use <think> blocks, explanations, or any text outside the JSON
- Do NOT include markdown code blocks or formatting
- Your entire response must be parseable JSON

{base_prompt}

REQUIRED JSON FORMAT:
{json_format}

{examples}

Remember: Respond with ONLY the JSON object, nothing else."""
        print(f"[LLMClient] Created structured JSON prompt:\n{prompt}")
        return prompt

# Global LLM client instance
llm_client = LLMClient()
