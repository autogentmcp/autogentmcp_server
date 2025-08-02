"""
Request models for the MCP Server API.
"""
from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class QueryRequest(BaseModel):
    """Basic query request model."""
    query: str
    session_id: str = "default"


class StreamingQueryRequest(BaseModel):
    """Streaming query request model."""
    query: str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = []
    include_analysis: bool = True
    max_steps: int = 5
    request_id: Optional[str] = None


class BrowserFingerprint(BaseModel):
    """Browser fingerprint data for session creation."""
    user_agent: Optional[str] = None
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    platform: Optional[str] = None
    viewport_size: Optional[str] = None
    color_depth: Optional[int] = None
    device_memory: Optional[float] = None
    hardware_concurrency: Optional[int] = None
    connection_type: Optional[str] = None
    cookie_enabled: Optional[bool] = None
    do_not_track: Optional[str] = None
    canvas_fingerprint: Optional[str] = None
    webgl_fingerprint: Optional[str] = None


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""
    browser_fingerprint: Optional[BrowserFingerprint] = None


class AuthCredential(BaseModel):
    """Authentication credential model."""
    key: str
    value: str
