"""
Session service - handles session management without enhanced orchestrator.
"""
from typing import Dict, Any, Optional, List
from app.models.requests import BrowserFingerprint
import uuid
import time
import hashlib


class SessionService:
    """Simplified session service without enhanced orchestrator dependencies."""
    
    def __init__(self):
        # Simple in-memory storage for now
        # In production, this would use a proper database
        self.sessions = {}
        self.conversations = {}
    
    def create_session_from_fingerprint(
        self, 
        fingerprint: Optional[BrowserFingerprint] = None
    ) -> Dict[str, Any]:
        """
        Create a new session using browser fingerprint.
        
        Args:
            fingerprint: Optional browser fingerprint data
            
        Returns:
            Session creation result
        """
        try:
            # Generate session ID
            session_id = str(uuid.uuid4())
            user_id = "anonymous"
            
            # Calculate fingerprint quality if provided
            fingerprint_quality = 0.5
            if fingerprint:
                fingerprint_quality = self._calculate_fingerprint_quality(fingerprint)
                # Use fingerprint to generate more consistent user_id
                fingerprint_str = str(fingerprint.dict())
                user_id = hashlib.md5(fingerprint_str.encode()).hexdigest()[:12]
            
            # Store session
            self.sessions[session_id] = {
                "user_id": user_id,
                "session_id": session_id,
                "created_at": time.time(),
                "fingerprint": fingerprint.dict() if fingerprint else {},
                "fingerprint_quality": fingerprint_quality
            }
            
            return {
                "status": "success",
                "user_id": user_id,
                "session_id": session_id,
                "conversation_id": session_id,  # For compatibility
                "fingerprint_quality": fingerprint_quality,
                "message": "Session created successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to create session"
            }
    
    def create_new_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Create a new conversation within an existing session.
        
        Args:
            session_id: The session ID
            
        Returns:
            Conversation creation result
        """
        try:
            if session_id not in self.sessions:
                return {
                    "status": "error",
                    "message": "Session not found"
                }
            
            conversation_id = str(uuid.uuid4())
            
            # Store conversation
            if session_id not in self.conversations:
                self.conversations[session_id] = []
            
            self.conversations[session_id].append({
                "conversation_id": conversation_id,
                "title": "New Chat",
                "created_at": time.time(),
                "last_activity": time.time()
            })
            
            return {
                "status": "success",
                "session_id": session_id,
                "new_conversation_id": conversation_id,
                "conversation_title": "New Chat",
                "message": "New conversation created"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to create new conversation"
            }
    
    def get_user_conversations(self, session_id: str) -> Dict[str, Any]:
        """
        Get all conversations for a user session.
        
        Args:
            session_id: The session ID
            
        Returns:
            List of conversations
        """
        try:
            conversations = self.conversations.get(session_id, [])
            
            return {
                "status": "success",
                "session_id": session_id,
                "conversations": conversations,
                "total_conversations": len(conversations)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to get conversations"
            }
    
    def _calculate_fingerprint_quality(self, fingerprint: BrowserFingerprint) -> float:
        """Calculate fingerprint quality score."""
        score = 0.0
        total_fields = 13  # Total number of fingerprint fields
        
        if fingerprint.user_agent:
            score += 1
        if fingerprint.screen_resolution:
            score += 1
        if fingerprint.timezone:
            score += 1
        if fingerprint.language:
            score += 1
        if fingerprint.platform:
            score += 1
        if fingerprint.viewport_size:
            score += 1
        if fingerprint.color_depth:
            score += 1
        if fingerprint.device_memory:
            score += 1
        if fingerprint.hardware_concurrency:
            score += 1
        if fingerprint.connection_type:
            score += 1
        if fingerprint.cookie_enabled is not None:
            score += 1
        if fingerprint.canvas_fingerprint:
            score += 1
        if fingerprint.webgl_fingerprint:
            score += 1
        
        return score / total_fields


# Global service instance
session_service = SessionService()
