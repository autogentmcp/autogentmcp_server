"""
Sidebar component for conversation management
"""

import streamlit as st
from typing import List, Dict, Any
from datetime import datetime

from ..styles.styles import SIDEBAR_STYLES
from ..config import UI_MESSAGES


class Sidebar:
    """Sidebar component for conversation management"""
    
    def __init__(self, conversation_storage, session_manager):
        self.conversation_storage = conversation_storage
        self.session_manager = session_manager
    
    def render(self):
        """Render the complete sidebar"""
        with st.sidebar:
            # Apply sidebar styles
            st.markdown(SIDEBAR_STYLES, unsafe_allow_html=True)
            
            # Sidebar title
            st.markdown('<div class="sidebar-title">Chats</div>', unsafe_allow_html=True)
            
            # Main sidebar content
            if st.session_state.current_session_id:
                self._render_conversation_management()
                self._render_conversation_actions()
                self._render_settings()
            else:
                st.info("No active session")
    
    def _render_conversation_management(self):
        """Render conversation list and management"""
        conversations = self._load_conversations()
        st.session_state.conversations = conversations
        
        # New Chat button
        if st.button(
            UI_MESSAGES["new_chat"], 
            key="new_chat_main", 
            use_container_width=True
        ):
            conversation_id = self._create_new_conversation()
            if conversation_id:
                st.success("New conversation created!")
                st.rerun()
        
        # Conversation list
        if conversations:
            for conv in conversations:
                self._render_conversation_item(conv)
        else:
            st.info(UI_MESSAGES["no_conversations"])
    
    def _render_conversation_item(self, conv: Dict[str, Any]):
        """Render a single conversation item"""
        conv_id = conv.get("conversation_id", "")
        title = conv.get("title", "New Chat")
        last_activity = conv.get("last_activity", "")
        
        # Format last activity
        time_str = self._format_time(last_activity)
        
        # Check if this is the current conversation
        is_current = conv_id == st.session_state.current_conversation_id
        
        # Create conversation item with columns
        col1, col2 = st.columns([5, 1])
        
        with col1:
            button_type = "primary" if is_current else "secondary"
            if st.button(
                title,
                key=f"conv_{conv_id}",
                type=button_type,
                help=f"Last activity: {time_str}",
                use_container_width=True
            ):
                self._switch_to_conversation(conv_id)
        
        with col2:
            if st.button("🗑", key=f"del_{conv_id}", help="Delete conversation"):
                self._delete_conversation(conv_id)
    
    def _render_conversation_actions(self):
        """Render conversation-specific actions"""
        if (st.session_state.current_conversation_id and 
            len(st.session_state.current_conversation_history) >= 4):
            
            st.markdown("---")
            if st.button("📝 Summarize Chat", use_container_width=True):
                summary = self.conversation_storage.generate_conversation_summary(
                    st.session_state.current_conversation_id
                )
                if summary:
                    st.success(f"💡 {summary}")
                else:
                    st.info("No summary generated for this conversation.")
    
    def _render_settings(self):
        """Render settings section"""
        st.markdown("---")
        
        if st.session_state.current_session_id:
            with st.expander("⚙️ Settings"):
                st.session_state.show_debug = st.checkbox(
                    "Debug Mode", 
                    value=st.session_state.show_debug
                )
                
                if st.button("🔄 Reset Session"):
                    self.session_manager.reset_session()
                    st.rerun()
    
    def _load_conversations(self) -> List[Dict[str, Any]]:
        """Load all conversations for the current session"""
        if not st.session_state.current_session_id:
            return []
        
        try:
            # Ensure session exists in database
            self.conversation_storage.create_session(
                st.session_state.current_session_id, 
                st.session_state.get("user_id", "anonymous")
            )
            
            conversations = self.conversation_storage.get_conversations(
                st.session_state.current_session_id
            )
            return conversations
        except Exception as e:
            st.error(f"Error loading conversations: {e}")
            return []
    
    def _create_new_conversation(self) -> str:
        """Create a new conversation"""
        if not st.session_state.current_session_id:
            st.error("No active session")
            return None
        
        try:
            print(f"[Sidebar] Creating new conversation for session: {st.session_state.current_session_id}")
            
            # Create conversation in storage
            conversation_id = self.conversation_storage.create_conversation(
                st.session_state.current_session_id
            )
            
            print(f"[Sidebar] Created conversation: {conversation_id}")
            
            # Update session state
            st.session_state.current_conversation_id = conversation_id
            st.session_state.current_conversation_history = []
            
            # Refresh conversations list
            st.session_state.conversations = self._load_conversations()
            
            return conversation_id
                
        except Exception as e:
            print(f"[Sidebar] Conversation creation exception: {e}")
            st.error(f"Error creating conversation: {e}")
            return None
    
    def _switch_to_conversation(self, conv_id: str):
        """Switch to a specific conversation"""
        print(f"[Sidebar] Switching to conversation: {conv_id}")
        st.session_state.current_conversation_id = conv_id
        
        # Load conversation history
        messages = self.conversation_storage.get_conversation_messages(conv_id)
        st.session_state.current_conversation_history = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["created_at"],
                "metadata": msg.get("metadata", {})
            }
            for msg in messages
        ]
        
        print(f"[Sidebar] Loaded {len(st.session_state.current_conversation_history)} messages")
        
        # Reset auto-load flag
        st.session_state.conversation_auto_loaded = True
        st.rerun()
    
    def _delete_conversation(self, conv_id: str):
        """Delete a conversation"""
        self.conversation_storage.delete_conversation(conv_id)
        st.success("Deleted!")
        
        # If this was the current conversation, clear it
        if conv_id == st.session_state.current_conversation_id:
            st.session_state.current_conversation_id = None
            st.session_state.current_conversation_history = []
        
        # Refresh conversations list
        st.session_state.conversations = self._load_conversations()
        st.rerun()
    
    def _format_time(self, last_activity: str) -> str:
        """Format last activity time"""
        try:
            if last_activity:
                dt = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                return dt.strftime("%m/%d")
            else:
                return "Today"
        except:
            return "Today"
