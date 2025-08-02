"""
LLM Client Factory for dynamic client instantiation
"""
from typing import Dict, Any, Optional
from .clients.openai_client import OpenAIClient
from .clients.deepseek_client import DeepSeekClient
from .clients.ollama_client import OllamaClient
from .clients.base_client import BaseLLMClient


class LLMClientFactory:
    """Factory for creating LLM client instances"""
    
    _client_registry = {
        'openai': OpenAIClient,
        'deepseek': DeepSeekClient,
        'ollama': OllamaClient
    }
    
    @classmethod
    def create_client(
        cls, 
        provider: str, 
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseLLMClient:
        """
        Create an LLM client instance for the specified provider
        
        Args:
            provider: The LLM provider name (openai, deepseek, ollama)
            config: Optional configuration dictionary (merged with kwargs)
            **kwargs: Additional keyword arguments for client initialization
            
        Returns:
            BaseLLMClient: Configured LLM client instance
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider not in cls._client_registry:
            available_providers = ', '.join(cls._client_registry.keys())
            raise ValueError(f"Unsupported provider: {provider}. Available: {available_providers}")
        
        # Merge config into kwargs if provided, but don't pass 'config' parameter
        final_kwargs = dict(kwargs)
        if config:
            final_kwargs.update(config)
        
        client_class = cls._client_registry[provider]
        return client_class(**final_kwargs)
    
    @classmethod
    def is_provider_supported(cls, provider: str) -> bool:
        """
        Check if a provider is supported by the factory
        
        Args:
            provider: The LLM provider name
            
        Returns:
            bool: True if provider is supported, False otherwise
        """
        return provider in cls._client_registry
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available LLM providers"""
        return list(cls._client_registry.keys())
    
    @classmethod
    def register_client(cls, provider: str, client_class: type):
        """Register a new LLM client provider"""
        cls._client_registry[provider] = client_class
