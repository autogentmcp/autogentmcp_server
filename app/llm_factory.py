"""
Unified LLM client factory supporting multiple providers (Ollama, DeepSeek).
"""
import os
from typing import Union, Dict, Any, Optional
from .llm_client import LLMClient
from .deepseek_client import DeepSeekClient

class LLMClientFactory:
    """Factory for creating LLM clients based on configuration."""
    
    @staticmethod
    def create_client(provider: str = "ollama", **kwargs) -> Union[LLMClient, DeepSeekClient]:
        """
        Create an LLM client based on the provider.
        
        Args:
            provider: The LLM provider ('ollama' or 'deepseek')
            **kwargs: Provider-specific configuration
            
        Returns:
            Configured LLM client instance
        """
        provider = provider.lower()
        
        if provider == "ollama":
            model = kwargs.get("model", "qwen2.5:32b")
            base_url = kwargs.get("base_url", "http://localhost:11434")
            return LLMClient(model=model, base_url=base_url)
            
        elif provider == "deepseek":
            model = kwargs.get("model", "deepseek-reasoner")
            api_key = kwargs.get("api_key")
            base_url = kwargs.get("base_url", "https://api.deepseek.com")
            return DeepSeekClient(model=model, api_key=api_key, base_url=base_url)
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: ollama, deepseek")
    
    @staticmethod
    def create_from_env() -> Union[LLMClient, DeepSeekClient]:
        """
        Create an LLM client based on environment variables.
        
        Environment Variables:
            LLM_PROVIDER: 'ollama' or 'deepseek' (default: 'ollama')
            LLM_MODEL: Model name for the provider
            LLM_BASE_URL: Base URL for the provider
            DEEPSEEK_API_KEY: Required for DeepSeek provider
            
        Returns:
            Configured LLM client instance
        """
        provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        
        if provider == "ollama":
            model = os.getenv("LLM_MODEL", "qwen2.5:32b")
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
            return LLMClient(model=model, base_url=base_url)
            
        elif provider == "deepseek":
            model = os.getenv("LLM_MODEL", "deepseek-reasoner")
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
            return DeepSeekClient(model=model, api_key=api_key, base_url=base_url)
            
        else:
            raise ValueError(f"Unsupported LLM provider in environment: {provider}")

# Global client instance - will be initialized based on environment
llm_client = None

def get_llm_client() -> Union[LLMClient, DeepSeekClient]:
    """Get the global LLM client instance, creating it if necessary."""
    global llm_client
    if llm_client is None:
        llm_client = LLMClientFactory.create_from_env()
    return llm_client

def set_llm_client(client: Union[LLMClient, DeepSeekClient]):
    """Set the global LLM client instance."""
    global llm_client
    llm_client = client

# Backwards compatibility - create default client
try:
    llm_client = LLMClientFactory.create_from_env()
except Exception as e:
    print(f"[LLMClientFactory] Warning: Could not initialize default client: {e}")
    print("[LLMClientFactory] Falling back to Ollama client")
    llm_client = LLMClient()
