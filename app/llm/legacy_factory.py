"""
Legacy LLM factory - redirects to new organized structure.
For new code, use app.llm.factory.LLMClientFactory instead.
"""
import os
import warnings
from typing import Union, Dict, Any, Optional

# Import from new organized structure
from .factory import LLMClientFactory as NewLLMClientFactory
from .clients.ollama_client import OllamaLLMClient as LLMClient
from .clients.deepseek_client import DeepSeekLLMClient as DeepSeekClient
from .clients.openai_client import OpenAILLMClient as OpenAIClient

# Deprecation warning
warnings.warn(
    "app.llm_factory is deprecated. Use app.llm.factory.LLMClientFactory instead.",
    DeprecationWarning,
    stacklevel=2
)

class LLMClientFactory:
    """Factory for creating LLM clients based on configuration."""
    
    @staticmethod
    def create_client(provider: str = "ollama", **kwargs) -> Union[LLMClient, DeepSeekClient, OpenAIClient]:
        """
        Create an LLM client based on the provider.
        
        Args:
            provider: The LLM provider ('ollama', 'deepseek', or 'openai')
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
            
        elif provider == "openai":
            model = kwargs.get("model", "gpt-4o-mini")
            api_key = kwargs.get("api_key")
            base_url = kwargs.get("base_url", "https://api.openai.com/v1")
            cert_file = kwargs.get("cert_file")
            cert_key = kwargs.get("cert_key")
            ca_bundle = kwargs.get("ca_bundle")
            verify_ssl = kwargs.get("verify_ssl", True)
            
            if not api_key:
                raise ValueError("OpenAI API key is required. Pass api_key parameter.")
            
            return OpenAIClient(
                model=model, 
                api_key=api_key, 
                base_url=base_url,
                cert_file=cert_file,
                cert_key=cert_key,
                ca_bundle=ca_bundle,
                verify_ssl=verify_ssl
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: ollama, deepseek, openai")
    
    @staticmethod
    def create_from_env() -> Union[LLMClient, DeepSeekClient, OpenAIClient]:
        """
        Create an LLM client based on environment variables.
        
        Environment Variables:
            LLM_PROVIDER: 'ollama', 'deepseek', or 'openai' (default: 'ollama')
            LLM_MODEL: Model name for the provider
            LLM_BASE_URL: Base URL for the provider
            DEEPSEEK_API_KEY: Required for DeepSeek provider
            OPENAI_API_KEY: Required for OpenAI provider
            OPENAI_CERT_FILE: Client certificate file (optional)
            OPENAI_CERT_KEY: Private key file (optional)
            OPENAI_CA_BUNDLE: CA bundle file (optional)
            OPENAI_VERIFY_SSL: SSL verification (default: true)
            
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
            
        elif provider == "openai":
            model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
            cert_file = os.getenv("OPENAI_CERT_FILE")
            cert_key = os.getenv("OPENAI_CERT_KEY")
            ca_bundle = os.getenv("OPENAI_CA_BUNDLE")
            verify_ssl = os.getenv("OPENAI_VERIFY_SSL", "true").lower() == "true"
            
            return OpenAIClient(
                model=model, 
                api_key=api_key, 
                base_url=base_url,
                cert_file=cert_file,
                cert_key=cert_key,
                ca_bundle=ca_bundle,
                verify_ssl=verify_ssl
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider in environment: {provider}")

# Global client instance - will be initialized based on environment
llm_client = None

def get_llm_client() -> Union[LLMClient, DeepSeekClient, OpenAIClient]:
    """Get the global LLM client instance, creating it if necessary."""
    global llm_client
    if llm_client is None:
        llm_client = LLMClientFactory.create_from_env()
    return llm_client

def set_llm_client(client: Union[LLMClient, DeepSeekClient, OpenAIClient]):
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
