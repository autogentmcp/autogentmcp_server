"""
Response models for the MCP Server API.
"""
from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class BaseResponse(BaseModel):
    """Base response model."""
    status: str
    message: Optional[str] = None


class HealthResponse(BaseResponse):
    """Health check response."""
    vault: Optional[Dict[str, Any]] = None


class QueryResponse(BaseResponse):
    """Query execution response."""
    session_id: str
    result: Optional[Dict[str, Any]] = None
    endpoint_info: Optional[Dict[str, Any]] = None


class SessionResponse(BaseResponse):
    """Session creation response."""
    user_id: Optional[str] = None
    session_id: str
    conversation_id: Optional[str] = None
    fingerprint_quality: Optional[float] = None


class ConversationResponse(BaseResponse):
    """Conversation management response."""
    session_id: str
    new_conversation_id: Optional[str] = None
    conversation_title: Optional[str] = None
    conversations: Optional[List[Dict[str, Any]]] = None
    total_conversations: Optional[int] = None


class ErrorResponse(BaseResponse):
    """Error response model."""
    error: str
    status: str = "error"
