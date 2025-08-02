"""
Multi-modal LLM Client with dynamic provider routing and selective provider enabling
"""
import os
import json
import logging
from typing import Dict, Any, Optional, AsyncIterator, List
from .factory import LLMClientFactory
from .clients.base_client import BaseLLMClient

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available

logger = logging.getLogger(__name__)


class MultiModeLLMClient:
    """Multi-provider LLM client with dynamic routing and selective provider enabling"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MultiMode LLM Client with selective provider enabling
        
        Args:
            config_path: Path to LLM configuration file (defaults to llm_config.json)
        """
        self.clients = {}
        self.config_path = config_path or "llm_config.json"
        self.llm_config = self._load_llm_config()
        self._initialize_enabled_clients()
    
    def _load_llm_config(self) -> Dict[str, Any]:
        """Load LLM configuration with selective provider enabling"""
        # Default least-permissive configuration - only OpenAI enabled
        default_config = {
            "enabled_providers": ["openai"],  # Only explicitly enabled providers
            "provider_configs": {
                "openai": {
                    "enabled": True,
                    "default_model": "gpt-4o-mini", 
                    "required_env_vars": ["OPENAI_API_KEY"],
                    "description": "OpenAI GPT models"
                },
                "deepseek": {
                    "enabled": False,  # Disabled by default
                    "default_model": "deepseek-chat",
                    "required_env_vars": ["DEEPSEEK_API_KEY"],
                    "description": "DeepSeek models"
                },
                "ollama": {
                    "enabled": False,  # Disabled by default
                    "default_model": "qwen2.5:32b",
                    "required_env_vars": [],
                    "description": "Local Ollama models"
                }
            },
            "task_routing": {
                "code_generation": {"provider": "openai", "model": "gpt-4o-mini"},
                "data_analysis": {"provider": "openai", "model": "gpt-4o-mini"},
                "general_chat": {"provider": "openai", "model": "gpt-4o-mini"},
                "tool_selection": {"provider": "openai", "model": "gpt-4o-mini"},
                "agent_selection": {"provider": "openai", "model": "gpt-4o-mini"}
            },
            "fallback_provider": "openai",
            "allow_disabled_fallback": False
        }
        
        # Try to load from config file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults, prioritizing file config
                    default_config.update(file_config)
                    logger.info(f"Loaded LLM config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load LLM config from {self.config_path}: {e}")
                logger.info("Using default least-permissive configuration")
        else:
            logger.info(f"LLM config file {self.config_path} not found, using default least-permissive configuration")
        
        return default_config
    
    def _initialize_enabled_clients(self):
        """Initialize only explicitly enabled LLM clients"""
        enabled_providers = self.llm_config.get("enabled_providers", [])
        provider_configs = self.llm_config.get("provider_configs", {})
        
        logger.info(f"Initializing LLM clients for enabled providers: {enabled_providers}")
        
        for provider in enabled_providers:
            provider_config = provider_configs.get(provider, {})
            
            # Double-check that provider is explicitly enabled
            if not provider_config.get("enabled", False):
                logger.warning(f"Provider {provider} is in enabled_providers but marked enabled=false, skipping")
                continue
            
            # Check required environment variables
            required_env_vars = provider_config.get("required_env_vars", [])
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.error(f"Cannot initialize {provider}: missing required environment variables: {missing_vars}")
                continue
            
            # Check if provider is supported by factory
            if not LLMClientFactory.is_provider_supported(provider):
                logger.error(f"Provider {provider} is not supported by LLMClientFactory")
                continue
            
            try:
                client = LLMClientFactory.create_client(provider)
                self.clients[provider] = client
                description = provider_config.get("description", provider)
                logger.info(f"✅ Initialized {provider} client: {description}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {provider} client: {e}")
        
        # Validate we have at least one working client
        if not self.clients:
            raise RuntimeError("No LLM clients successfully initialized. Check your configuration and environment variables.")
        
        logger.info(f"Successfully initialized {len(self.clients)} LLM client(s): {list(self.clients.keys())}")
    
    def get_client_for_task(self, task_type: str = None) -> tuple[BaseLLMClient, str]:
        """
        Get the appropriate client and model for a given task with fallback support
        
        Args:
            task_type: Type of task (optional)
            
        Returns:
            tuple: (client_instance, model_name)
        """
        task_routing = self.llm_config.get("task_routing", {})
        fallback_provider = self.llm_config.get("fallback_provider", "openai")
        allow_disabled_fallback = self.llm_config.get("allow_disabled_fallback", False)
        
        # Determine provider and model based on task
        if task_type and task_type in task_routing:
            task_config = task_routing[task_type]
            provider = task_config["provider"]
            model = task_config["model"]
        else:
            # Use fallback provider
            provider = fallback_provider
            provider_configs = self.llm_config.get("provider_configs", {})
            model = provider_configs.get(provider, {}).get("default_model", "gpt-4o-mini")
        
        # Check if requested provider is available
        if provider in self.clients:
            return self.clients[provider], model
        
        # Provider not available, try fallback
        logger.warning(f"Requested provider {provider} not available for task {task_type}")
        
        if not allow_disabled_fallback:
            if fallback_provider in self.clients:
                provider_configs = self.llm_config.get("provider_configs", {})
                fallback_model = provider_configs.get(fallback_provider, {}).get("default_model", "gpt-4o-mini")
                logger.info(f"Using fallback provider {fallback_provider} instead")
                return self.clients[fallback_provider], fallback_model
        
        # If no fallback allowed or fallback not available, use any available client
        if self.clients:
            available_provider = next(iter(self.clients.keys()))
            provider_configs = self.llm_config.get("provider_configs", {})
            available_model = provider_configs.get(available_provider, {}).get("default_model", "gpt-4o-mini")
            logger.warning(f"Using any available provider {available_provider} for task {task_type}")
            return self.clients[available_provider], available_model
        
        raise ValueError(f"No LLM providers available for task {task_type}")
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        task_type: str = None,
        **kwargs
    ) -> str:
        """
        Generate a response using the appropriate LLM client
        
        Args:
            messages: List of message dictionaries
            task_type: Type of task for routing
            **kwargs: Additional arguments for the LLM client
            
        Returns:
            str: Generated response
        """
        client, model = self.get_client_for_task(task_type)
        
        # Override model if not specified in kwargs
        if 'model' not in kwargs:
            kwargs['model'] = model
        
        return await client.generate_response(messages, **kwargs)
    
    async def generate_streaming_response(
        self, 
        messages: List[Dict[str, str]], 
        task_type: str = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response using the appropriate LLM client
        
        Args:
            messages: List of message dictionaries
            task_type: Type of task for routing
            **kwargs: Additional arguments for the LLM client
            
        Yields:
            str: Response chunks
        """
        client, model = self.get_client_for_task(task_type)
        
        # Override model if not specified in kwargs
        if 'model' not in kwargs:
            kwargs['model'] = model
        
        async for chunk in client.generate_streaming_response(messages, **kwargs):
            yield chunk
    
    def invoke_with_json_response(self, prompt: str, context: str = "", task_type: str = None, timeout: int = 600) -> Optional[Dict[str, Any]]:
        """
        Synchronous JSON response using the appropriate LLM client
        
        Args:
            prompt: The prompt text
            context: Additional context
            task_type: Type of task for routing
            timeout: Request timeout
            
        Returns:
            Optional[Dict[str, Any]]: Parsed JSON response
        """
        import asyncio
        
        async def _async_invoke():
            client, model = self.get_client_for_task(task_type)
            return await client.invoke_with_json_response(prompt, context, timeout)
        
        try:
            # Handle both sync and async contexts
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create a new event loop in a thread
                    import concurrent.futures
                    import threading
                    
                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(_async_invoke())
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        return future.result(timeout=timeout)
                else:
                    # Event loop exists but not running
                    return loop.run_until_complete(_async_invoke())
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(_async_invoke())
        except Exception as e:
            logger.error(f"Error in invoke_with_json_response: {e}")
            return None

    async def invoke_with_json_response_async(self, prompt: str, context: str = "", task_type: str = None, timeout: int = 600) -> Optional[Dict[str, Any]]:
        """
        Async JSON response using the appropriate LLM client
        
        Args:
            prompt: The prompt text
            context: Additional context
            task_type: Type of task for routing
            timeout: Request timeout
            
        Returns:
            Optional[Dict[str, Any]]: Parsed JSON response
        """
        try:
            client, model = self.get_client_for_task(task_type)
            return await client.invoke_with_json_response(prompt, context, timeout)
        except Exception as e:
            logger.error(f"Error in invoke_with_json_response_async: {e}")
            return None
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about available providers and their configurations"""
        info = {
            "available_providers": list(self.clients.keys()),
            "enabled_providers": self.llm_config.get("enabled_providers", []),
            "fallback_provider": self.llm_config.get("fallback_provider"),
            "provider_configs": self.llm_config.get("provider_configs", {}),
            "task_routing": self.llm_config.get("task_routing", {}),
            "total_initialized_clients": len(self.clients)
        }
        return info
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of currently enabled and initialized providers"""
        return list(self.clients.keys())
    
    def is_provider_enabled(self, provider: str) -> bool:
        """Check if a specific provider is enabled and initialized"""
        return provider in self.clients
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all configured providers"""
        provider_configs = self.llm_config.get("provider_configs", {})
        status = {}
        
        for provider, config in provider_configs.items():
            required_vars = config.get("required_env_vars", [])
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            status[provider] = {
                "configured": True,
                "enabled_in_config": config.get("enabled", False),
                "initialized": provider in self.clients,
                "description": config.get("description", ""),
                "default_model": config.get("default_model", ""),
                "required_env_vars": required_vars,
                "missing_env_vars": missing_vars,
                "ready": provider in self.clients
            }
        
        return status
        return info
