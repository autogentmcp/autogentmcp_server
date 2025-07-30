"""
DeepSeek LLM client for high-performance reasoning tasks.
"""
import re
import json
import os
from typing import Dict, Any, Optional, AsyncIterator
from openai import AsyncOpenAI
from .base_llm_client import BaseLLMClient

class DeepSeekClient(BaseLLMClient):
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
        super().__init__(model)
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

    async def invoke_with_json_response(self, prompt: str, context: str = "", timeout: int = 600) -> Optional[Dict[str, Any]]:
        """DeepSeek does not implement JSON response - use multimode client instead."""
        raise NotImplementedError("DeepSeek client is specialized for SQL generation only. Use MultiModeLLMClient for JSON responses.")

    async def invoke_with_text_response(self, prompt: str, context: str = "", allow_diagrams: bool = True) -> str:
        """DeepSeek does not implement text response - use multimode client instead."""
        raise NotImplementedError("DeepSeek client is specialized for SQL generation only. Use MultiModeLLMClient for text responses.")

    async def stream_with_text_response(self, prompt: str, context: str = ""):
        """DeepSeek does not implement streaming - use multimode client instead."""
        raise NotImplementedError("DeepSeek client is specialized for SQL generation only. Use MultiModeLLMClient for streaming.")

    async def test_connection(self) -> Dict[str, Any]:
        """Test DeepSeek connection."""
        return {
            "success": True,
            "model": self.model,
            "note": "DeepSeek client is specialized for SQL generation only"
        }
