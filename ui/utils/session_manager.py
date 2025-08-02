"""
Session management utilities for MCP Chat Interface
"""

import time
import uuid
import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional

from ..config import SESSION_CONFIG


class SessionManager:
    """Manages session state and lifecycle"""
    
    def __init__(self):
        self.auto_load_recent = SESSION_CONFIG.get("auto_load_recent", True)
        self.max_history_length = SESSION_CONFIG.get("max_history_length", 10)
    
    def _is_streamlit_context(self) -> bool:
        """Check if we're running in a Streamlit context."""
        try:
            # Try to access session_state to see if we're in Streamlit context
            _ = st.session_state
            return True
        except Exception:
            return False
    
    def initialize_session_state(self):
        """Initialize all session state variables."""
        
        # Session management
        if 'mcp_session_initialized' not in st.session_state:
            st.session_state.mcp_session_initialized = False
        
        if 'mcp_session_data' not in st.session_state:
            st.session_state.mcp_session_data = None
        
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
        
        if 'current_conversation_id' not in st.session_state:
            st.session_state.current_conversation_id = None
        
        # Conversation management
        if 'conversations' not in st.session_state:
            st.session_state.conversations = []
        
        if 'current_conversation_history' not in st.session_state:
            st.session_state.current_conversation_history = []
        
        # UI state
        if 'show_debug' not in st.session_state:
            st.session_state.show_debug = False
        
        if 'last_query_time' not in st.session_state:
            st.session_state.last_query_time = None
        
        # Processing state management
        if 'is_processing' not in st.session_state:
            st.session_state.is_processing = False
        
        if 'stop_processing' not in st.session_state:
            st.session_state.stop_processing = False
        
        if 'current_request_id' not in st.session_state:
            st.session_state.current_request_id = None
        
        # Auto-load conversation flag
        if 'conversation_auto_loaded' not in st.session_state:
            st.session_state.conversation_auto_loaded = False
    
    def get_or_create_session(self) -> Dict[str, Any]:
        """Get or create a session using browser storage."""
        try:
            if not st.session_state.get('mcp_session_initialized', False):
                # Try to get session from URL params first
                session_id = st.query_params.get("session_id")
                
                if not session_id:
                    # Generate a new session ID and store it in URL params
                    session_id = f"sess_{int(time.time())}_{str(uuid.uuid4())[:8]}"
                    st.query_params["session_id"] = session_id
                
                # Create session data
                session_data = {
                    "session_id": session_id,
                    "user_id": "anonymous",
                    "created_at": datetime.now().isoformat(),
                    "source": "browser_storage"
                }
                
                st.session_state.current_session_id = session_id
                st.session_state.user_id = "anonymous"
                st.session_state.mcp_session_initialized = True
                return session_data
            else:
                return {
                    "session_id": st.session_state.current_session_id,
                    "user_id": st.session_state.get("user_id", "anonymous"),
                    "is_existing": True
                }
        except Exception:
            # Return default session data if not in Streamlit context
            return {
                "session_id": f"default_{int(time.time())}",
                "user_id": "anonymous",
                "created_at": datetime.now().isoformat(),
                "source": "fallback"
            }
    
    def reset_session(self):
        """Reset the current session and reinitialize."""
        try:
            st.session_state.mcp_session_initialized = False
            st.session_state.current_session_id = None
            st.session_state.current_conversation_id = None
            st.session_state.conversations = []
            st.session_state.current_conversation_history = []
            st.session_state.conversation_auto_loaded = False
            st.session_state.is_processing = False
            st.session_state.stop_processing = False
        except Exception:
            pass  # Ignore if not in Streamlit context
    
    def get_conversation_context(self) -> list:
        """Get recent conversation history for context."""
        try:
            if not st.session_state.current_conversation_history:
                return []
            
            # Get recent history (last N messages)
            recent_history = st.session_state.current_conversation_history[-self.max_history_length:]
            
            return [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp", "")
                }
                for msg in recent_history
            ]
        except Exception:
            return []  # Return empty if not in Streamlit context
    
    def add_message_to_history(self, role: str, content: str, metadata: Dict = None):
        """Add a message to the current conversation history."""
        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            st.session_state.current_conversation_history.append(message)
        except Exception:
            pass  # Ignore if not in Streamlit context
    
    def is_processing(self) -> bool:
        """Check if currently processing a request."""
        try:
            return st.session_state.get('is_processing', False)
        except Exception:
            return False
    
    def set_processing(self, processing: bool):
        """Set processing state."""
        try:
            st.session_state.is_processing = processing
            if not processing:
                st.session_state.stop_processing = False
        except Exception:
            pass  # Ignore if not in Streamlit context
    
    def stop_current_processing(self):
        """Stop current processing."""
        try:
            st.session_state.stop_processing = True
            st.session_state.is_processing = False
        except Exception:
            pass  # Ignore if not in Streamlit context
    
    def should_stop_processing(self) -> bool:
        """Check if processing should be stopped."""
        try:
            return st.session_state.get('stop_processing', False)
        except Exception:
            return False
