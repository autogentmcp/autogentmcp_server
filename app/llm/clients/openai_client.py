"""
Secure OpenAI LLM client with enterprise security features.
Implements secure configuration pattern with fixed environment variables,
proxy support, custom headers, and SSL/TLS certificate handling.
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

from .base_client import BaseLLMClient

# Fixed environment variable names (security-critical)
OPENAI_ENV_VARS = {
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "OPENAI_PROXY_URL": "OPENAI_PROXY_URL",
    "OPENAI_BASE_URL": "OPENAI_BASE_URL",
    "OPENAI_CUSTOM_HEADERS": "OPENAI_CUSTOM_HEADERS",
    "OPENAI_CA_BUNDLE": "OPENAI_CA_BUNDLE",
    "OPENAI_CLIENT_CERT": "OPENAI_CLIENT_CERT",
    "OPENAI_CLIENT_KEY": "OPENAI_CLIENT_KEY",
    "OPENAI_REJECT_UNAUTHORIZED": "OPENAI_REJECT_UNAUTHORIZED"
}

class OpenAIClient(BaseLLMClient):
    
    def __init__(self, 
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 # Legacy parameters kept for backward compatibility
                 cert_file: Optional[str] = None,
                 cert_key: Optional[str] = None,
                 ca_bundle: Optional[str] = None,
                 verify_ssl: Optional[bool] = None):
        """
        Initialize secure OpenAI client with enterprise features.
        
        Args:
            model: OpenAI model name (defaults to OPENAI_MODEL env var or gpt-4o-mini)
            api_key: OpenAI API key (defaults to secure env var lookup)
            base_url: Base URL (defaults to secure env var lookup with proxy priority)
            
        Legacy compatibility args (will be overridden by secure config):
            cert_file: Deprecated - use OPENAI_CLIENT_CERT env var
            cert_key: Deprecated - use OPENAI_CLIENT_KEY env var  
            ca_bundle: Deprecated - use OPENAI_CA_BUNDLE env var
            verify_ssl: Deprecated - use OPENAI_REJECT_UNAUTHORIZED env var
        """
        # Get secure configuration
        config = self._get_secure_config()
        
        # Use secure config values, falling back to parameters for backward compatibility
        final_model = model or config["model"]
        final_api_key = api_key or config["api_key"]
        final_base_url = base_url or config["base_url"]
        
        super().__init__(final_model)
        self.base_url = final_base_url
        self.api_key = final_api_key
        self.custom_headers = config["custom_headers"]
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Create HTTP client with secure SSL configuration and proxy support
        http_client = self._create_secure_http_client(config, cert_file, cert_key, ca_bundle, verify_ssl)
        
        # Prepare client configuration
        client_config = {
            "api_key": self.api_key,
            "base_url": final_base_url
        }
        
        # Add custom headers if configured
        if self.custom_headers:
            client_config["default_headers"] = self.custom_headers
            print(f"[OpenAIClient] Applied {len(self.custom_headers)} custom headers")
        
        # Add HTTP client if customization (SSL/proxy) is needed
        if http_client:
            client_config["http_client"] = http_client
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(**client_config)
        
        print(f"[OpenAIClient] Initialized securely with model: {final_model}")
        print(f"[OpenAIClient] Base URL: {final_base_url}")
        if config["is_proxy_used"]:
            print(f"[OpenAIClient] Using proxy: {config['proxy_url']} (corporate environment detected)")
        else:
            print(f"[OpenAIClient] Direct connection to OpenAI API")
    
    def _get_secure_config(self) -> Dict[str, Any]:
        """Get secure OpenAI configuration using fixed environment variables."""
        # API Key (required)
        api_key = os.getenv(OPENAI_ENV_VARS["OPENAI_API_KEY"], "")
        
        # URL Configuration - Proxy and Base URL are separate concepts
        proxy_url = os.getenv(OPENAI_ENV_VARS["OPENAI_PROXY_URL"])
        base_url = os.getenv(OPENAI_ENV_VARS["OPENAI_BASE_URL"])
        
        # Base URL should always be OpenAI's API unless explicitly overridden
        final_base_url = base_url or "https://api.openai.com/v1"
        
        # Proxy is handled separately in HTTP client configuration
        is_proxy_used = bool(proxy_url)
        
        # Model configuration - only use OpenAI-specific model, don't fall back to general LLM_MODEL
        model = os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
        
        # Parse custom headers
        custom_headers = self._parse_custom_headers(
            os.getenv(OPENAI_ENV_VARS["OPENAI_CUSTOM_HEADERS"])
        )
        
        # SSL Configuration
        ssl_config = self._build_ssl_config()
        
        return {
            "api_key": api_key,
            "base_url": final_base_url,
            "proxy_url": proxy_url,
            "model": model,
            "custom_headers": custom_headers,
            "ssl_config": ssl_config,
            "is_proxy_used": is_proxy_used
        }
    
    def _parse_custom_headers(self, env_var_value: Optional[str]) -> Dict[str, str]:
        """Parse custom headers from environment variable using secure format."""
        if not env_var_value or not env_var_value.strip():
            return {}
        
        try:
            # Try JSON format first (recommended)
            if env_var_value.strip().startswith('{'):
                headers = json.loads(env_var_value)
                print(f"[OpenAIClient] Parsed {len(headers)} custom headers from JSON format")
                return headers
            
            # Fallback: Parse delimited format "key1:value1;key2:value2"
            headers = {}
            pairs = env_var_value.split(';')
            
            for pair in pairs:
                parts = pair.split(':', 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    headers[parts[0].strip()] = parts[1].strip()
            
            if headers:
                print(f"[OpenAIClient] Parsed {len(headers)} custom headers from delimited format")
            return headers
            
        except Exception as e:
            print(f"[OpenAIClient] Warning: Failed to parse custom headers: {e}")
            return {}
    
    def _build_ssl_config(self) -> Dict[str, Any]:
        """Build SSL configuration from secure environment variables."""
        ssl_config = {}
        
        # Reject unauthorized certificates setting
        reject_unauthorized = os.getenv(
            OPENAI_ENV_VARS["OPENAI_REJECT_UNAUTHORIZED"], "true"
        ).lower() != "false"
        
        ssl_config["verify"] = reject_unauthorized
        
        # SSL certificate files
        ca_bundle_path = os.getenv(OPENAI_ENV_VARS["OPENAI_CA_BUNDLE"])
        client_cert_path = os.getenv(OPENAI_ENV_VARS["OPENAI_CLIENT_CERT"])
        client_key_path = os.getenv(OPENAI_ENV_VARS["OPENAI_CLIENT_KEY"])
        
        if ca_bundle_path:
            ssl_config["ca_bundle"] = ca_bundle_path
        
        if client_cert_path and client_key_path:
            ssl_config["cert"] = (client_cert_path, client_key_path)
        elif client_cert_path:
            ssl_config["cert"] = client_cert_path
        
        return ssl_config
    
    def _create_secure_http_client(self, config: Dict[str, Any], 
                                   legacy_cert_file: Optional[str] = None,
                                   legacy_cert_key: Optional[str] = None,
                                   legacy_ca_bundle: Optional[str] = None,
                                   legacy_verify_ssl: Optional[bool] = None) -> Optional[httpx.AsyncClient]:
        """Create HTTP client with secure SSL configuration and proxy support."""
        ssl_config = config["ssl_config"]
        proxy_url = config.get("proxy_url")
        
        # Check if any customization is needed
        needs_custom_client = (
            proxy_url or
            not ssl_config["verify"] or
            ssl_config.get("ca_bundle") or
            ssl_config.get("cert") or
            legacy_cert_file or legacy_cert_key or legacy_ca_bundle or 
            (legacy_verify_ssl is not None and not legacy_verify_ssl)
        )
        
        if not needs_custom_client:
            return None  # Use default HTTP client
        
        print(f"[OpenAIClient] Configuring custom HTTP client")
        
        # Create SSL context
        ssl_context = ssl.create_default_context()
        
        # Configure SSL verification (secure config takes priority)
        verify_ssl = ssl_config["verify"]
        if legacy_verify_ssl is not None:
            print(f"[OpenAIClient] Warning: Using legacy verify_ssl parameter. Consider using OPENAI_REJECT_UNAUTHORIZED env var.")
            verify_ssl = legacy_verify_ssl
        
        if not verify_ssl:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            print(f"[OpenAIClient] SSL verification disabled")
        
        # Load custom CA bundle (secure config takes priority)
        ca_bundle = ssl_config.get("ca_bundle") or legacy_ca_bundle
        if ca_bundle and os.path.exists(ca_bundle):
            ssl_context.load_verify_locations(ca_bundle)
            print(f"[OpenAIClient] Loaded CA bundle from: {ca_bundle}")
        elif ca_bundle:
            print(f"[OpenAIClient] Warning: CA bundle file not found: {ca_bundle}")
        
        # Load client certificate (secure config takes priority)
        cert_info = ssl_config.get("cert")
        cert_file = legacy_cert_file
        cert_key = legacy_cert_key
        
        if cert_info:
            if isinstance(cert_info, tuple):
                cert_file, cert_key = cert_info
            else:
                cert_file = cert_info
                cert_key = None
        
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
        
        # Configure HTTP client with proxy and SSL
        client_config = {
            "verify": ssl_context,
            "timeout": httpx.Timeout(60.0)
        }
        
        # Add proxy configuration if provided
        if proxy_url:
            client_config["proxies"] = {
                "http://": proxy_url,
                "https://": proxy_url
            }
            print(f"[OpenAIClient] Configured HTTP/HTTPS proxy: {proxy_url}")
        
        return httpx.AsyncClient(**client_config)
    
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
