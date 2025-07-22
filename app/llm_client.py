"""
LLM client wrapper for consistent LLM interactions.
"""
from langchain_ollama import ChatOllama
import re
import json
from typing import Dict, Any, Optional

class LLMClient:
    """Wrapper for LLM interactions with consistent prompting and parsing."""
    
    def __init__(self, model: str = "qwen3:14b", base_url: str = "http://localhost:11434"):
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
        """Create a prompt for formatting data query results."""
        return f"""/no_think
You are an expert data analyst assistant.

User query: {query}

SQL Query executed: {sql_query}

Query result: {query_result}

Instructions:
- Present the data in a clear, user-friendly format
- If the result contains multiple rows, format as a markdown table
- Include relevant insights or patterns in the data if applicable
- If there's an error in the query result, explain it clearly
- For empty results, explain that no data was found matching the criteria
- Do not include the raw SQL query in your response unless specifically asked
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
                    column_reference += f"🏷️  {table_name}:\n"
                    
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

    def create_sql_generation_prompt(self, user_query: str, schema_context: str = None, connection_type: str = "unknown", tables_data: list = None) -> str:
        """Create an enhanced prompt for SQL query generation with structured JSON output."""
        
        # Use structured data if available, otherwise fall back to text parsing
        if tables_data:
            column_reference = self._extract_column_reference_from_structured_data(tables_data)
            # Create a simple schema context from structured data
            schema_context = self._build_schema_context_from_structured_data(tables_data)
        else:
            column_reference = self._extract_column_reference(schema_context or "")
        
        return f"""You are an expert SQL developer. Analyze the user request and generate a SQL query with your reasoning.

⚠️ CRITICAL: You MUST use ONLY the exact column names from the schema below. DO NOT assume or invent any column names.

USER REQUEST: {user_query}

DATABASE TYPE: {connection_type}

EXACT SCHEMA DEFINITION:
{schema_context}

STRICT COLUMN NAME REFERENCE:
{column_reference}

MANDATORY VALIDATION PROCESS:
1. Identify which tables you need for the query
2. For each table, find the exact column names in the schema above  
3. Verify that every column in your SQL query exists in the schema
4. Do NOT use assumed column names like "product_name", "customer_name", etc.
5. Use ONLY the exact column names shown in the schema

REQUIRED JSON FORMAT:
{{
    "thought": "Step 1: Tables needed: [list]. Step 2: Exact columns from schema: [list each column with table.column format]. Step 3: Verified all columns exist in schema.",
    "query": "The exact SQL query using ONLY the column names from the schema above"
}}

VALIDATION CHECKLIST:
- ✅ All column names are from the schema above
- ✅ No assumed or invented column names used
- ✅ Table names match the schema exactly
- ✅ Column names match the schema exactly

Respond with ONLY your JSON - validate every column against the schema:"""

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
        
        return prompt

# Global LLM client instance
llm_client = LLMClient()
