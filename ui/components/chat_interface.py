"""
Main chat interface component
"""

import streamlit as st
from typing import Dict, Any
from datetime import datetime

from ..styles.styles import MAIN_APP_STYLES
from ..config import UI_MESSAGES
from .progress import ProgressDisplay, WorkflowProgressRenderer
from .agent_selector import AgentSelector


class ChatInterface:
    """Main chat interface component"""
    
    def __init__(self, session_manager, conversation_storage, api_service):
        self.session_manager = session_manager
        self.conversation_storage = conversation_storage
        self.api_service = api_service
        self.progress_display = ProgressDisplay()
        self.workflow_renderer = WorkflowProgressRenderer()
        self.agent_selector = AgentSelector()
    
    def render(self):
        """Render the complete chat interface"""
        # Apply main app styles
        st.markdown(MAIN_APP_STYLES, unsafe_allow_html=True)
        
        # Start chat content wrapper
        st.markdown('<div class="chat-content">', unsafe_allow_html=True)
        
        # Custom title
        st.markdown('<div class="chat-title">🚀 Enhanced MCP Assistant</div>', unsafe_allow_html=True)
        
        # Initialize session
        session_data = self.session_manager.get_or_create_session()
        
        # Auto-load recent conversation
        self._auto_load_recent_conversation()
        
        # Debug information
        self._render_debug_info(session_data)
        
        # Main chat interface
        has_messages = bool(st.session_state.current_conversation_history)
        
        if has_messages:
            self._render_existing_conversation()
        else:
            self._render_new_conversation()
        
        # Close chat content wrapper
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _auto_load_recent_conversation(self):
        """Auto-load the most recent conversation if needed"""
        if (st.session_state.current_session_id and 
            not st.session_state.current_conversation_id and 
            not st.session_state.conversation_auto_loaded):
            
            print(f"[ChatInterface] Auto-loading conversation check")
            
            conversations = self._load_conversations()
            if conversations:
                recent_conv = conversations[0]
                conv_id = recent_conv.get("conversation_id", "")
                
                if conv_id:
                    print(f"[ChatInterface] Auto-loading conversation: {conv_id}")
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
                    
                    print(f"[ChatInterface] Auto-loaded {len(st.session_state.current_conversation_history)} messages")
            
            st.session_state.conversation_auto_loaded = True
    
    def _render_debug_info(self, session_data: Dict[str, Any]):
        """Render debug information if enabled"""
        if st.session_state.show_debug:
            with st.expander("🔍 Debug Information", expanded=False):
                st.json({
                    "session_data": session_data,
                    "current_conversation_id": st.session_state.current_conversation_id,
                    "conversations_count": len(st.session_state.conversations),
                    "last_query_time": st.session_state.last_query_time,
                    "history_length": len(st.session_state.current_conversation_history),
                    "conversation_auto_loaded": st.session_state.conversation_auto_loaded
                })
    
    def _render_existing_conversation(self):
        """Render interface for existing conversation with history"""
        # Display conversation history
        for message in st.session_state.current_conversation_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Show workflow progress for assistant messages
                    if "metadata" in message:
                        metadata = message["metadata"]
                        workflow_events = metadata.get("workflow_events", [])
                        if workflow_events and not st.session_state.is_processing:
                            workflow_result = {
                                "workflow_id": metadata.get("workflow_id"),
                                "agents_used": metadata.get("agents_used", []),
                                "total_execution_time": metadata.get("execution_time", 0),
                                "total_steps": len([
                                    e for e in workflow_events 
                                    if isinstance(e, dict) and e.get("type") == "step_completed"
                                ])
                            }
                            self.workflow_renderer.render_workflow_progress(
                                workflow_events, workflow_result, False
                            )
                    
                    # Show debug info if enabled
                    if "metadata" in message and st.session_state.show_debug:
                        metadata = message["metadata"]
                        if metadata:
                            with st.expander("📊 Response Details"):
                                st.json(metadata)
        
        # Chat input
        self._render_chat_input()
    
    def _render_new_conversation(self):
        """Render interface for new conversation"""
        # Welcome message
        st.markdown(
            f"""
            <div class="welcome-container">
                <div class="welcome-title">{UI_MESSAGES['welcome']['title']}</div>
                <div class="welcome-subtitle">{UI_MESSAGES['welcome']['subtitle']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Chat input
        self._render_chat_input()
    
    def _render_chat_input(self):
        """Render chat input area with processing state handling"""
        if self.session_manager.is_processing():
            user_input = st.chat_input(
                UI_MESSAGES["processing"],
                disabled=True
            )
            # Show stop button
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button(UI_MESSAGES["stop_processing"], type="secondary", use_container_width=True):
                    self.session_manager.stop_current_processing()
                    st.rerun()
        else:
            user_input = st.chat_input(UI_MESSAGES["input_placeholder"])
        
        # Handle user input
        if user_input:
            self._handle_user_input(user_input)
    
    def _handle_user_input(self, user_input: str):
        """Handle user input and generate response"""
        print(f"[ChatInterface] ========== NEW QUERY STARTED ==========")
        print(f"[ChatInterface] Query: {user_input}")
        
        # Prevent multiple simultaneous queries
        if self.session_manager.is_processing():
            st.warning("⚠️ Please wait for the current query to complete before sending another.")
            return
        
        # Set processing state
        self.session_manager.set_processing(True)
        
        # Ensure we have a conversation
        if not st.session_state.current_conversation_id:
            conversation_id = self._create_new_conversation()
            if not conversation_id:
                st.error("Failed to create conversation. Please try again.")
                self.session_manager.set_processing(False)
                return
        
        # Add user message to history
        self.session_manager.add_message_to_history("user", user_input)
        
        # Store in database
        self.conversation_storage.add_message(
            st.session_state.current_conversation_id,
            "user",
            user_input
        )
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate and display response
        self._generate_response(user_input)
    
    def _generate_response(self, user_input: str):
        """Generate response using the API service"""
        with st.chat_message("assistant"):
            # Create progress placeholder
            progress_placeholder = st.empty()
            
            # Clear progress display
            self.progress_display.clear()
            
            # Show initial loading
            self.progress_display.render_loading_spinner(
                progress_placeholder, 
                "🚀 Starting AI analysis..."
            )
            
            # Set up progress callback
            def progress_callback(event_type: str, event_data: Dict[str, Any]):
                if not self.session_manager.should_stop_processing():
                    self.progress_display.update_progress(
                        progress_placeholder, event_type, event_data
                    )
            
            # Send query with progress updates
            response = self.api_service.send_query_with_streaming(
                user_input,
                st.session_state.current_session_id,
                st.session_state.current_conversation_id,
                self.session_manager.get_conversation_context(),
                progress_callback
            )
            
            # Clear progress display
            progress_placeholder.empty()
            
            # Process response
            self._process_response(response, user_input)
    
    def _process_response(self, response: Dict[str, Any], user_input: str):
        """Process and display the response"""
        self.session_manager.set_processing(False)
        
        print(f"[ChatInterface] Got response: {response.get('status', 'no_status')}")
        
        if response.get("status") == "success":
            # Handle different response formats
            if "enhanced_result" in response:
                workflow_result = response.get("enhanced_result", {})
            else:
                workflow_result = response
            
            # Extract response text
            response_text = (
                workflow_result.get("greeting", "") or 
                workflow_result.get("final_answer", "") or 
                workflow_result.get("message", "") or 
                "I apologize, but I couldn't generate a proper response."
            )
            
            print(f"[ChatInterface] Displaying response: {response_text[:100]}...")
            
            # Check for agent selection scenario
            if (workflow_result.get("status") == "need_more_info" or 
                workflow_result.get("type") == "clarification_needed"):
                
                # Show agent selection UI
                if not self.agent_selector.render_agent_selection_ui(workflow_result):
                    st.write(response_text)
            else:
                # Regular response
                st.write(response_text)
            
            # Store assistant response
            metadata = {
                "workflow_events": workflow_result.get("workflow_events", []),
                "execution_time": workflow_result.get("total_execution_time", 0),
                "agents_used": workflow_result.get("agents_used", []),
                "workflow_id": workflow_result.get("workflow_id")
            }
            
            self.session_manager.add_message_to_history("assistant", response_text, metadata)
            
            # Store in database
            self.conversation_storage.add_message(
                st.session_state.current_conversation_id,
                "assistant",
                response_text,
                metadata
            )
            
            # Show workflow progress
            workflow_events = workflow_result.get("workflow_events", [])
            if workflow_events:
                self.workflow_renderer.render_workflow_progress(
                    workflow_events, workflow_result, True
                )
            
            # Enhanced visualization for data results
            execution_results = workflow_result.get("results", [])
            if execution_results:
                self._render_data_results(execution_results)
        
        else:
            # Error handling
            error_message = response.get("message", "An error occurred while processing your request.")
            st.error(f"❌ {error_message}")
            
            if st.session_state.show_debug:
                st.json(response)
    
    def _render_data_results(self, results: list):
        """Render data visualization results"""
        # This would import and use the enhanced_visualization module
        # For now, just show a placeholder
        if results:
            with st.expander("📊 Query Results", expanded=True):
                for i, result in enumerate(results):
                    st.subheader(f"Result {i+1}")
                    if isinstance(result, dict):
                        if result.get("data"):
                            st.dataframe(result["data"])
                        else:
                            st.json(result)
                    else:
                        st.write(result)
    
    def _create_new_conversation(self) -> str:
        """Create a new conversation"""
        if not st.session_state.current_session_id:
            return None
        
        try:
            conversation_id = self.conversation_storage.create_conversation(
                st.session_state.current_session_id
            )
            
            st.session_state.current_conversation_id = conversation_id
            st.session_state.current_conversation_history = []
            
            return conversation_id
        except Exception as e:
            print(f"[ChatInterface] Error creating conversation: {e}")
            return None
    
    def _load_conversations(self):
        """Load conversations for the current session"""
        if not st.session_state.current_session_id:
            return []
        
        try:
            self.conversation_storage.create_session(
                st.session_state.current_session_id, 
                st.session_state.get("user_id", "anonymous")
            )
            
            return self.conversation_storage.get_conversations(
                st.session_state.current_session_id
            )
        except Exception as e:
            st.error(f"Error loading conversations: {e}")
            return []
