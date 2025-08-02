"""
Session controller - handles session management endpoints.
"""
from fastapi import APIRouter
from app.services.session_service import session_service
from app.models.requests import SessionCreateRequest
from app.models.responses import SessionResponse, ConversationResponse

router = APIRouter(tags=["session"])


@router.post("/enhanced/session/create", response_model=SessionResponse)
def create_session(request: SessionCreateRequest):
    """Create a new session with browser fingerprint."""
    result = session_service.create_session_from_fingerprint(
        fingerprint=request.browser_fingerprint
    )
    return result


@router.post("/enhanced/session/{session_id}/new_conversation", response_model=ConversationResponse)
def create_new_conversation(session_id: str):
    """Create a new conversation within an existing session."""
    result = session_service.create_new_conversation(session_id)
    return result


@router.get("/enhanced/session/{session_id}/conversations", response_model=ConversationResponse)
def get_user_conversations(session_id: str):
    """Get all conversations for a user session."""
    result = session_service.get_user_conversations(session_id)
    return result
