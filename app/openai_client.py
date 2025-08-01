"""
OpenAI LLM client with certificate support for enterprise environments.
"""
import re
import json
import os
import ssl
import httpx
from typing import Dict, Any, Optional, AsyncIterator
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .base_llm_client import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    
    def __init__(self, 
                 model: str = "gpt-4o-mini", 
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 cert_file: Optional[str] = None,
                 cert_key: Optional[str] = None,
                 ca_bundle: Optional[str] = None,
                 verify_ssl: bool = True):
        """
        Initialize OpenAI client with certificate support.
        
        Args:
            model: OpenAI model name (gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.)
            api_key: OpenAI API key (will use OPENAI_API_KEY env var if not provided)
            base_url: OpenAI API base URL
            cert_file: Path to client certificate file (PEM format)
            cert_key: Path to private key file (PEM format)
            ca_bundle: Path to CA bundle file for custom certificate authorities
            verify_ssl: Whether to verify SSL certificates
        """
        super().__init__(model)
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Create HTTP client with certificate support
        http_client = None
        if cert_file or cert_key or ca_bundle or not verify_ssl:
            print(f"[OpenAIClient] Configuring custom SSL settings")
            
            # Create SSL context
            ssl_context = ssl.create_default_context()
            
            # Configure SSL verification
            if not verify_ssl:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                print(f"[OpenAIClient] SSL verification disabled")
            
            # Load custom CA bundle
            if ca_bundle and os.path.exists(ca_bundle):
                ssl_context.load_verify_locations(ca_bundle)
                print(f"[OpenAIClient] Loaded CA bundle from: {ca_bundle}")
            
            # Load client certificate
            if cert_file and cert_key:
                if os.path.exists(cert_file) and os.path.exists(cert_key):
                    ssl_context.load_cert_chain(cert_file, cert_key)
                    print(f"[OpenAIClient] Loaded client certificate: {cert_file}")
                else:
                    print(f"[OpenAIClient] Warning: Certificate files not found - cert: {cert_file}, key: {cert_key}")
            elif cert_file:
                if os.path.exists(cert_file):
                    ssl_context.load_cert_chain(cert_file)
                    print(f"[OpenAIClient] Loaded client certificate: {cert_file}")
                else:
                    print(f"[OpenAIClient] Warning: Certificate file not found: {cert_file}")
            
            # Create HTTP client with custom SSL context
            http_client = httpx.AsyncClient(
                verify=ssl_context,
                timeout=httpx.Timeout(60.0)
            )
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            http_client=http_client
        )
        
        print(f"[OpenAIClient] Initialized with model: {model}")
    
    async def stream_with_text_response(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        """
        Stream response from OpenAI with text output.
        
        Args:
            prompt: The main prompt
            context: Additional context to include
            
        Yields:
            String chunks of the response as they arrive
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[OpenAIClient] Starting streaming call")
        
        try:
            messages = [{"role": "user", "content": full_prompt}]
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.1,
                max_tokens=4000
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"[OpenAIClient] Streaming call failed: {e}")
            raise Exception(f"OpenAI streaming call failed: {str(e)}")
    
    async def invoke(self, prompt: str, context: str = "", timeout: int = 600):
        """
        General invoke method for compatibility.
        Returns a response object with a content attribute.
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[OpenAIClient] General invoke with timeout: {timeout}s")
        
        try:
            messages = [{"role": "user", "content": full_prompt}]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=4000
            )
            
            # Create a response object similar to langchain's format
            class OpenAIResponse:
                def __init__(self, content: str):
                    self.content = content
            
            return OpenAIResponse(response.choices[0].message.content)
            
        except Exception as e:
            print(f"[OpenAIClient] General invoke failed: {e}")
            raise Exception(f"OpenAI call failed: {str(e)}")

    async def invoke_with_json_response(self, prompt: str, context: str = "", timeout: int = 600) -> Optional[Dict[str, Any]]:
        """
        Invoke OpenAI with a prompt expecting JSON response.
        
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
        print(f"[OpenAIClient] Invoking with JSON response expected, timeout: {timeout}s")
        
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
            print(f"[OpenAIClient] Response received in {elapsed:.1f}s")
            
            llm_response = response.choices[0].message.content
            
        except Exception as e:
            print(f"[OpenAIClient] Invocation failed: {e}")
            raise Exception(f"OpenAI call failed after timeout or error: {str(e)}")
        
        # Truncate very long responses for logging
        response_preview = llm_response[:500] + "..." if len(llm_response) > 500 else llm_response
        print(f"[OpenAIClient] Response preview:\n{response_preview}")
        
        # Clean response by removing any thinking or reasoning blocks
        clean_response = re.sub(r'<think>[\s\S]*?</think>', '', llm_response, flags=re.IGNORECASE).strip()
        clean_response = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', clean_response, flags=re.IGNORECASE).strip()
        
        # Remove any leading/trailing non-JSON content before the first {
        json_start = clean_response.find('{')
        json_end = clean_response.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            clean_response = clean_response[json_start:json_end + 1]
        
        print(f"[OpenAIClient] Cleaned response for JSON parsing:\n{clean_response[:300]}...")
        
        try:
            parsed = json.loads(clean_response)
            print(f"[OpenAIClient] Successfully parsed JSON response")
            return parsed
        except json.JSONDecodeError as e:
            print(f"[OpenAIClient] JSON parsing error: {e}")
            print(f"[OpenAIClient] Problematic content around error: {clean_response[max(0, e.pos-50):e.pos+50]}")
            
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
                        print(f"[OpenAIClient] Successfully extracted and parsed JSON with pattern")
                        return parsed
                    except json.JSONDecodeError:
                        continue
                        
            print(f"[OpenAIClient] No valid JSON found in response")
            return None
    
    async def invoke_with_text_response(self, prompt: str, context: str = "", allow_diagrams: bool = True) -> str:
        """
        Invoke OpenAI with a prompt expecting text response.
        
        Args:
            prompt: The main prompt
            context: Additional context to include
            allow_diagrams: Whether to preserve diagrams in the response
            
        Returns:
            Processed text response
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[OpenAIClient] Invoking with text response expected")
        
        try:
            messages = [{"role": "user", "content": full_prompt}]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=4000
            )
            
            llm_response = response.choices[0].message.content
            
            # Clean and process response if needed
            if allow_diagrams:
                processed_response = self._extract_and_replace_diagrams(llm_response)
            else:
                processed_response = llm_response
            
            print(f"[OpenAIClient] Text response generated successfully")
            return processed_response
            
        except Exception as e:
            print(f"[OpenAIClient] Text invocation failed: {e}")
            raise Exception(f"OpenAI text call failed: {str(e)}")
    
    # Synchronous versions for compatibility
    
    def invoke_with_json_response_sync(self, prompt: str, context: str = "", timeout: int = 600) -> Optional[Dict[str, Any]]:
        """Synchronous version of invoke_with_json_response."""
        import asyncio
        try:
            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, we need to create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.invoke_with_json_response(prompt, context, timeout))
                    return future.result(timeout=timeout)
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                return asyncio.run(self.invoke_with_json_response(prompt, context, timeout))
        except Exception as e:
            print(f"[OpenAIClient] Sync JSON call failed: {e}")
            return None
    
    def invoke_with_text_response_sync(self, prompt: str, context: str = "", allow_diagrams: bool = True) -> str:
        """Synchronous version of invoke_with_text_response."""
        import asyncio
        try:
            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, we need to create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.invoke_with_text_response(prompt, context, allow_diagrams))
                    return future.result(timeout=600)
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                return asyncio.run(self.invoke_with_text_response(prompt, context, allow_diagrams))
        except Exception as e:
            print(f"[OpenAIClient] Sync text call failed: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    def _extract_and_replace_diagrams(self, text: str) -> str:
        """Extract diagrams from thinking blocks and convert to markdown code blocks."""
        # Pattern to match ASCII art diagrams within thinking blocks
        diagram_pattern = r'<think>(.*?)(```[\s\S]*?```|[\s\S]*?(?:\||\+|\-){3,}[\s\S]*?)</think>'
        
        def extract_diagram(match):
            thinking_content = match.group(1)
            diagram_content = match.group(2)
            
            # If it's already a code block, return it as-is
            if diagram_content.startswith('```'):
                return f"\n{diagram_content}\n"
            
            # Check if it looks like a diagram (contains drawing characters)
            drawing_chars = ['|', '+', '-', '/', '\\', '┌', '┐', '└', '┘', '─', '│', '├', '┤', '┬', '┴', '┼']
            if any(char in diagram_content for char in drawing_chars):
                return f"\n```\n{diagram_content.strip()}\n```\n"
            
            return ""  # Remove if it's not a diagram
        
        # Replace diagram patterns
        processed_text = re.sub(diagram_pattern, extract_diagram, text, flags=re.DOTALL)
        
        # Remove any remaining <think> blocks
        processed_text = re.sub(r'<think>[\s\S]*?</think>', '', processed_text, flags=re.IGNORECASE)
        
        return processed_text.strip()

    async def test_connection(self) -> Dict[str, Any]:
        """Test OpenAI connection with a simple prompt."""
        try:
            test_prompt = "Say 'Hello' and nothing else."
            response = await self.invoke_with_text_response(test_prompt)
            
            return {
                "success": bool(response),
                "response": response,
                "model": self.model,
                "api_key_present": bool(self.api_key)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
                "api_key_present": bool(self.api_key)
            }
