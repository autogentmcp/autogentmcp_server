"""
Ollama client wrapper for consistent LLM interactions using langchain-ollama.
"""
import re
import json
import time
import os
from typing import Dict, Any, Optional, AsyncGenerator
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .base_client import BaseLLMClient

class OllamaClient(BaseLLMClient):
    """Wrapper for Ollama LLM interactions with consistent prompting and parsing."""
    
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        # Use provided model or default from environment
        final_model = model or os.getenv("OLLAMA_DEFAULT_MODEL") or os.getenv("LLM_MODEL", "qwen2.5:32b")
        final_base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434")
        
        super().__init__(final_model)
        self.base_url = final_base_url
        self.model = final_model
        
        print(f"[OllamaClient] Initialized with model: {final_model}, base_url: {final_base_url}")
    
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
        print(f"[OllamaClient] Starting streaming LLM call")
        
        try:
            # Use astream for streaming response
            async for chunk in self.ollama.astream(full_prompt):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
        except Exception as e:
            print(f"[OllamaClient] Streaming LLM call failed: {e}")
            raise Exception(f"Streaming LLM call failed: {str(e)}")
    
    async def invoke(self, prompt: str, context: str = "", timeout: int = 600):
        """
        General invoke method for compatibility.
        Returns a response object with a content attribute.
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[OllamaClient] General invoke with timeout: {timeout}s")
        
        try:
            response = self.ollama.invoke(full_prompt, think=False)
            return response
        except Exception as e:
            print(f"[OllamaClient] General invoke failed: {e}")
            raise Exception(f"LLM call failed: {str(e)}")

    async def invoke_with_json_response(self, prompt: str, context: str = "", timeout: int = 600) -> Optional[Dict[str, Any]]:
        """
        Invoke LLM with a prompt expecting JSON response.
        
        Args:
            prompt: The main prompt
            context: Additional context to include
            timeout: Timeout in seconds (default 600 = 10 minutes)
            
        Returns:
            Parsed JSON response or None if parsing fails
        """
        
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[OllamaClient] Invoking LLM with timeout: {timeout}s")
        
        try:
            start_time = time.time()
            response = self.ollama.invoke(full_prompt, think=False)
            elapsed = time.time() - start_time
            print(f"[OllamaClient] LLM response received in {elapsed:.1f}s")
            
        except Exception as e:
            print(f"[OllamaClient] LLM invocation failed: {e}")
            raise Exception(f"LLM call failed after timeout or error: {str(e)}")
        
        llm_response = response.content if hasattr(response, 'content') else str(response)
        
        # Truncate very long responses for logging
        response_preview = llm_response[:500] + "..." if len(llm_response) > 500 else llm_response
        print(f"[OllamaClient] Response preview:\n{response_preview}")
        
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
        
        print(f"[OllamaClient] Cleaned response for JSON parsing:\n{clean_response[:300]}...")
        
        try:
            parsed = json.loads(clean_response)
            print(f"[OllamaClient] Successfully parsed JSON response")
            return parsed
        except json.JSONDecodeError as e:
            print(f"[OllamaClient] JSON parsing error: {e}")
            print(f"[OllamaClient] Problematic content around error: {clean_response[max(0, e.pos-50):e.pos+50]}")
            
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
                        print(f"[OllamaClient] Successfully extracted and parsed JSON with pattern")
                        return parsed
                    except json.JSONDecodeError:
                        continue
                        
            print(f"[OllamaClient] No valid JSON found in response")
            return None
    
    async def invoke_with_text_response(self, prompt: str, context: str = "", allow_diagrams: bool = True) -> str:
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
        print(f"[OllamaClient] Text Prompt:\n{full_prompt}")
        
        try:
            response = self.ollama.invoke(full_prompt, think=False)
            llm_response = response.content if hasattr(response, 'content') else str(response)
            print(f"[OllamaClient] Raw Text Response:\n{llm_response}")
            
            if not llm_response or llm_response.strip() == "":
                print("[OllamaClient] Warning: Empty response from LLM")
                return "No response generated"
            
            if allow_diagrams:
                # Extract diagrams from <think> blocks and convert to markdown
                processed_response = self._extract_and_replace_diagrams(llm_response)
            else:
                # Remove <think> blocks entirely
                processed_response = re.sub(r'<think>[\s\S]*?</think>', '', llm_response, flags=re.IGNORECASE).strip()
            
            print(f"[OllamaClient] Processed Text Response:\n{processed_response}")
            return processed_response if processed_response else "No response generated"
            
        except Exception as e:
            print(f"[OllamaClient] Error invoking LLM for text response: {e}")
            return f"Error generating response: {str(e)}"

    async def test_connection(self) -> Dict[str, Any]:
        """Test LLM connection with a simple prompt."""
        try:
            test_prompt = "Say 'Hello, LLM connection is working!' and nothing else."
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
    
    async def generate_response(self, messages: list, **kwargs) -> str:
        """Generate response from messages using Ollama."""
        try:
            # Convert messages to a single prompt if needed
            if isinstance(messages, list) and len(messages) > 0:
                if isinstance(messages[0], dict) and 'content' in messages[0]:
                    # Extract content from message format
                    prompt = messages[-1]['content']  # Use the last message as prompt
                else:
                    prompt = str(messages[0])
            else:
                prompt = str(messages)
            
            # Use the existing text response method
            return await self.invoke_with_text_response(prompt, "")
            
        except Exception as e:
            print(f"[OllamaClient] Generate response failed: {e}")
            return f"Error generating response: {str(e)}"


# Global Ollama client instance
ollama_client = OllamaClient()
