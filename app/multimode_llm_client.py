"""
Multi-mode LLM Client for using different LLM providers for different tasks.
This allows routing specific tasks to models that excel at them.
"""

import os
import time
import json
import re
from typing import Dict, Any, Optional, AsyncGenerator
from enum import Enum

from app.llm_client import LLMClient
from app.deepseek_client import DeepSeekClient


class TaskType(Enum):
    """Different types of tasks that can be routed to different LLM providers"""
    AGENT_SELECTION = "agent_selection"
    INTENT_ANALYSIS = "intent_analysis" 
    TOOL_SELECTION = "tool_selection"
    SQL_GENERATION = "sql_generation"
    FINAL_ANSWER = "final_answer"
    DATA_ANSWER = "data_answer"
    CONVERSATION = "conversation"
    GENERAL = "general"


class MultiModeLLMClient:
    """
    Multi-mode LLM client that routes different tasks to different LLM providers.
    
    Example configuration:
    - Agent routing, intent analysis: Ollama (fast, local)
    - SQL generation: DeepSeek (specialized for coding)
    - Final answers: Either based on complexity
    """
    
    def __init__(self):
        # Load configuration from environment first
        self.task_routing = self._load_task_routing_config()
        
        # Only initialize clients that are actually used
        providers_needed = set(self.task_routing.values())
        
        self.ollama_client = None
        self.deepseek_client = None
        
        if "ollama" in providers_needed:
            print("[MultiModeLLMClient] Initializing Ollama client...")
            self.ollama_client = LLMClient()
        
        if "deepseek" in providers_needed:
            print("[MultiModeLLMClient] Initializing DeepSeek client...")
            # Check if API key is available before initializing
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key:
                self.deepseek_client = DeepSeekClient()
            else:
                print("[MultiModeLLMClient] Warning: DeepSeek configured but no API key found. Falling back to Ollama.")
                # Update routing to use Ollama for tasks that were supposed to use DeepSeek
                for task_type, provider in self.task_routing.items():
                    if provider == "deepseek":
                        self.task_routing[task_type] = "ollama"
                # Initialize Ollama if not already done
                if not self.ollama_client:
                    self.ollama_client = LLMClient()
        
        # Test connections on startup
        self._test_connections()
    
    def _load_task_routing_config(self) -> Dict[TaskType, str]:
        """Load task routing configuration from environment variables"""
        
        # Default routing configuration
        default_routing = {
            TaskType.AGENT_SELECTION: "ollama",
            TaskType.INTENT_ANALYSIS: "ollama", 
            TaskType.TOOL_SELECTION: "ollama",
            TaskType.SQL_GENERATION: "deepseek",
            TaskType.FINAL_ANSWER: "ollama",
            TaskType.DATA_ANSWER: "ollama",
            TaskType.CONVERSATION: "ollama",
            TaskType.GENERAL: "ollama"
        }
        
        # Override with environment variables if set
        routing = {}
        for task_type in TaskType:
            env_key = f"LLM_ROUTING_{task_type.value.upper()}"
            provider = os.getenv(env_key, default_routing[task_type])
            routing[task_type] = provider
            
        print(f"[MultiModeLLMClient] Task routing configuration:")
        for task_type, provider in routing.items():
            print(f"  {task_type.value}: {provider}")
            
        return routing
    
    def _test_connections(self):
        """Test connections to all configured providers"""
        print(f"[MultiModeLLMClient] Testing provider connections...")
        
        # Test Ollama if initialized
        if self.ollama_client:
            try:
                result = self.ollama_client.test_connection()
                print(f"  Ollama: {'✓' if result['success'] else '✗'} {result.get('model', 'N/A')}")
            except Exception as e:
                print(f"  Ollama: ✗ Connection failed: {e}")
        
        # Test DeepSeek if initialized
        if self.deepseek_client:
            try:
                result = self.deepseek_client.test_connection_sync()
                print(f"  DeepSeek: {'✓' if result['success'] else '✗'} {result.get('model', 'N/A')}")
            except Exception as e:
                print(f"  DeepSeek: ✗ Connection failed: {e}")
    
    def _get_client_for_task(self, task_type: TaskType):
        """Get the appropriate LLM client for a specific task type"""
        provider = self.task_routing.get(task_type, "ollama")
        
        if provider == "deepseek" and self.deepseek_client:
            return self.deepseek_client
        else:
            # Default to Ollama client
            if not self.ollama_client:
                print("[MultiModeLLMClient] Warning: Ollama client not initialized, initializing now...")
                self.ollama_client = LLMClient()
            return self.ollama_client
    
    # Main interface methods that route to appropriate providers
    
    def invoke_with_json_response(self, prompt: str, context: str = "", 
                                task_type: TaskType = TaskType.GENERAL,
                                timeout: int = 600) -> Optional[Dict[str, Any]]:
        """
        Invoke LLM with JSON response, routing based on task type.
        
        Args:
            prompt: The main prompt
            context: Additional context
            task_type: Type of task to determine which provider to use
            timeout: Timeout in seconds
        """
        client = self._get_client_for_task(task_type)
        provider_name = self.task_routing.get(task_type, "ollama")
        
        print(f"[MultiModeLLMClient] Routing {task_type.value} to {provider_name}")
        
        # Determine which client we actually got back
        if client == self.deepseek_client and self.deepseek_client is not None:
            print(f"[MultiModeLLMClient] Using DeepSeek client for {task_type.value}")
            return client.invoke_with_json_response_sync(prompt, context, timeout)
        else:
            print(f"[MultiModeLLMClient] Using Ollama client for {task_type.value}")
            return client.invoke_with_json_response(prompt, context, timeout)
    
    def invoke_with_text_response(self, prompt: str, context: str = "",
                                task_type: TaskType = TaskType.GENERAL,
                                allow_diagrams: bool = True) -> str:
        """
        Invoke LLM with text response, routing based on task type.
        
        Args:
            prompt: The main prompt
            context: Additional context
            task_type: Type of task to determine which provider to use
            allow_diagrams: Whether to preserve diagrams in response
        """
        client = self._get_client_for_task(task_type)
        provider_name = self.task_routing.get(task_type, "ollama")
        
        print(f"[MultiModeLLMClient] Routing {task_type.value} to {provider_name}")
        
        # Determine which client we actually got back
        if client == self.deepseek_client and self.deepseek_client is not None:
            print(f"[MultiModeLLMClient] Using DeepSeek client for {task_type.value}")
            return client.invoke_with_text_response_sync(prompt, context, allow_diagrams)
        else:
            print(f"[MultiModeLLMClient] Using Ollama client for {task_type.value}")
            return client.invoke_with_text_response(prompt, context, allow_diagrams)
    
    async def stream_with_text_response(self, prompt: str, context: str = "",
                                      task_type: TaskType = TaskType.GENERAL) -> AsyncGenerator[str, None]:
        """
        Stream response from LLM, routing based on task type.
        
        Args:
            prompt: The main prompt
            context: Additional context
            task_type: Type of task to determine which provider to use
        """
        client = self._get_client_for_task(task_type)
        provider_name = self.task_routing.get(task_type, "ollama")
        
        print(f"[MultiModeLLMClient] Streaming {task_type.value} to {provider_name}")
        
        async for chunk in client.stream_with_text_response(prompt, context):
            yield chunk
    
    def invoke(self, prompt: str, context: str = "", timeout: int = 600,
              task_type: TaskType = TaskType.GENERAL):
        """
        General invoke method for compatibility, routing based on task type.
        """
        client = self._get_client_for_task(task_type)
        provider_name = self.task_routing.get(task_type, "ollama")
        
        print(f"[MultiModeLLMClient] General invoke {task_type.value} to {provider_name}")
        
        # Determine which client we actually got back
        if client == self.deepseek_client and self.deepseek_client is not None:
            print(f"[MultiModeLLMClient] Using DeepSeek client for {task_type.value}")
            return client.invoke_sync(prompt, context, timeout)
        else:
            print(f"[MultiModeLLMClient] Using Ollama client for {task_type.value}")
            return client.invoke(prompt, context, timeout)
    
    # Task-specific convenience methods
    
    def analyze_intent(self, prompt: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Analyze user intent - typically fast local model"""
        return self.invoke_with_json_response(
            prompt, context, 
            task_type=TaskType.INTENT_ANALYSIS
        )
    
    def select_agent(self, prompt: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Select appropriate agent - typically fast local model"""
        return self.invoke_with_json_response(
            prompt, context,
            task_type=TaskType.AGENT_SELECTION
        )
    
    def generate_sql(self, prompt: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Generate SQL query - typically specialized coding model"""
        return self.invoke_with_json_response(
            prompt, context,
            task_type=TaskType.SQL_GENERATION,
            timeout=300  # SQL generation might need more time
        )
    
    def format_final_answer(self, prompt: str, context: str = "") -> str:
        """Format final answer for user"""
        return self.invoke_with_text_response(
            prompt, context,
            task_type=TaskType.FINAL_ANSWER
        )
    
    def format_data_answer(self, prompt: str, context: str = "") -> str:
        """Format data query results"""
        return self.invoke_with_text_response(
            prompt, context,
            task_type=TaskType.DATA_ANSWER
        )
    
    # Utility methods
    
    def get_provider_for_task(self, task_type: TaskType) -> str:
        """Get which provider handles a specific task type"""
        return self.task_routing.get(task_type, "ollama")
    
    def get_task_routing_summary(self) -> Dict[str, str]:
        """Get a summary of current task routing configuration"""
        return {task.value: provider for task, provider in self.task_routing.items()}
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connections to all providers and return status"""
        results = {}
        
        # Test each unique provider
        unique_providers = set(self.task_routing.values())
        
        for provider in unique_providers:
            try:
                if provider == "ollama" and self.ollama_client:
                    result = self.ollama_client.test_connection()
                elif provider == "deepseek" and self.deepseek_client:
                    result = self.deepseek_client.test_connection_sync()
                else:
                    result = {"success": False, "error": f"Provider {provider} not available"}
                
                results[provider] = result
            except Exception as e:
                results[provider] = {"success": False, "error": str(e)}
        
        # Overall status
        all_success = all(r.get("success", False) for r in results.values())
        
        return {
            "success": all_success,
            "providers": results,
            "routing": self.get_task_routing_summary()
        }


# Global singleton instance to avoid re-initialization
_global_llm_client = None

def get_global_llm_client() -> MultiModeLLMClient:
    """Get the global MultiModeLLMClient instance, creating it if needed"""
    global _global_llm_client
    if _global_llm_client is None:
        _global_llm_client = MultiModeLLMClient()
    return _global_llm_client
