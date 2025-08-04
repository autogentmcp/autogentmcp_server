"""
Base LLM client with common functionality for all LLM providers.
"""
import re
import json
from typing import Dict, Any, Optional, AsyncIterator
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Base class for all LLM clients with common prompt generation methods."""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    async def invoke_with_json_response(self, prompt: str, context: str = "", timeout: int = 600) -> Optional[Dict[str, Any]]:
        """Invoke with JSON response - must be implemented by subclass."""
        pass
    
    @abstractmethod
    async def invoke_with_text_response(self, prompt: str, context: str = "", allow_diagrams: bool = True) -> str:
        """Invoke with text response - must be implemented by subclass."""
        pass
    
    @abstractmethod
    async def stream_with_text_response(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        """Stream text response - must be implemented by subclass."""
        pass
    
    @abstractmethod
    async def invoke(self, prompt: str, context: str = "", timeout: int = 600):
        """General invoke method - must be implemented by subclass."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection - must be implemented by subclass."""
        pass
    
    @abstractmethod
    async def generate_response(self, messages: list, **kwargs) -> str:
        """Generate response from messages - must be implemented by subclass."""
        pass
    
    # Common prompt generation methods can be added here
    # For now, keeping it simple and focused
