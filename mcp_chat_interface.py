"""
Enhanced MCP Chat Interface for Streamlit
A ChatGPT-like interface with conversation management, browser fingerprinting, and enhanced orchestration

Dependencies:
- streamlit
- requests
- sseclient-py (optional, for real-time progress streaming)

Install missing dependencies:
pip install sseclient-py
"""

import streamlit as st
import requests
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

# Try to import sseclient for streaming support
try:
    import sseclient
    has_sse_support = True
except ImportError:
    has_sse_support = False

# Import our session helper and conversation storage
from streamlit_session_helper import StreamlitSessionManager
from conversation_storage import ConversationStorage
from enhanced_visualization import render_enhanced_results, render_data_visualization

# Configuration
BACKEND_URL = "http://localhost:8001"
PAGE_TITLE = "🚀 Enhanced MCP Assistant"
PAGE_ICON = "🤖"

# Initialize session manager and conversation storage
session_manager = StreamlitSessionManager(BACKEND_URL)
conversation_storage = ConversationStorage()

def init_streamlit_config():
    """Initialize Streamlit page configuration."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )

def initialize_session_state():
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
    
    # Auto-load conversation flag to ensure it happens only once
    if 'conversation_auto_loaded' not in st.session_state:
        st.session_state.conversation_auto_loaded = False

def get_or_create_session() -> Dict[str, Any]:
    """Get or create a session using browser storage."""
    if not st.session_state.get('mcp_session_initialized', False):
        # Try to get session from browser storage first
        session_id = st.query_params.get("session_id")
        
        if not session_id:
            # Generate a new session ID and store it in URL params for persistence
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

def auto_load_recent_conversation():
    """Auto-load the most recent conversation if no conversation is currently loaded."""
    if (st.session_state.current_session_id and 
        not st.session_state.current_conversation_id and 
        not st.session_state.conversation_auto_loaded):
        
        print(f"[DEBUG] Auto-loading conversation check - Session: {st.session_state.current_session_id}")
        
        conversations = load_conversations()
        if conversations:
            # Get the most recent conversation (first in list)
            recent_conv = conversations[0]
            conv_id = recent_conv.get("conversation_id", "")
            
            if conv_id:
                print(f"[DEBUG] Auto-loading most recent conversation: {conv_id}")
                st.session_state.current_conversation_id = conv_id
                
                # Load conversation history from SQLite
                messages = conversation_storage.get_conversation_messages(conv_id)
                st.session_state.current_conversation_history = [
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg["created_at"],
                        "metadata": msg.get("metadata", {})
                    }
                    for msg in messages
                ]
                
                print(f"[DEBUG] Auto-loaded {len(st.session_state.current_conversation_history)} messages")
                # Set flag to prevent re-loading
                st.session_state.conversation_auto_loaded = True
                return True
        
        # Mark as attempted even if no conversations found
        st.session_state.conversation_auto_loaded = True
    
    return False

def load_conversations() -> List[Dict[str, Any]]:
    """Load all conversations for the current session from SQLite."""
    if not st.session_state.current_session_id:
        return []
    
    try:
        # Ensure session exists in database
        conversation_storage.create_session(
            st.session_state.current_session_id, 
            st.session_state.get("user_id", "anonymous")
        )
        
        conversations = conversation_storage.get_conversations(st.session_state.current_session_id)
        return conversations
    except Exception as e:
        st.error(f"Error loading conversations: {e}")
        return []

def create_new_conversation() -> Optional[str]:
    """Create a new conversation using SQLite storage."""
    if not st.session_state.current_session_id:
        st.error("No active session")
        return None
    
    try:
        print(f"[DEBUG] Creating new conversation for session: {st.session_state.current_session_id}")
        
        # Create conversation in SQLite
        conversation_id = conversation_storage.create_conversation(st.session_state.current_session_id)
        
        print(f"[DEBUG] Created conversation: {conversation_id}")
        
        # Update session state
        st.session_state.current_conversation_id = conversation_id
        st.session_state.current_conversation_history = []
        
        # Refresh conversations list
        st.session_state.conversations = load_conversations()
        
        return conversation_id
            
    except Exception as e:
        print(f"[DEBUG] Conversation creation exception: {e}")
        st.error(f"Error creating conversation: {e}")
        return None

def send_query_with_streaming(query: str, conversation_id: str = None, progress_container=None) -> Dict[str, Any]:
    """Send a query to the enhanced orchestrator with real-time streaming progress."""
    print(f"[DEBUG] *** STREAMING FUNCTION CALLED *** Query: {query[:50]}...")
    try:
        # Set processing state
        st.session_state.is_processing = True
        st.session_state.stop_processing = False
        request_id = str(uuid.uuid4())
        st.session_state.current_request_id = request_id
        
        # Check if sseclient is available
        if not has_sse_support:
            print("sseclient not available, falling back to regular query")
            return send_query_fallback(query, conversation_id)
        
        # Prepare conversation history for context
        conversation_history = []
        if conversation_id and st.session_state.current_conversation_history:
            # Include recent conversation history for context (last 10 messages)
            recent_history = st.session_state.current_conversation_history[-10:]
            conversation_history = [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp", "")
                }
                for msg in recent_history
            ]
        
        payload = {
            "query": query,
            "session_id": st.session_state.current_session_id,
            "conversation_id": conversation_id,
            "conversation_history": conversation_history,
            "include_analysis": True,
            "max_steps": 5,
            "request_id": request_id
        }
        
        print(f"[DEBUG] Streaming query with {len(conversation_history)} history messages")
        
        # Use the streaming endpoint
        response = requests.post(
            f"{BACKEND_URL}/orchestration/enhanced/stream",
            json=payload,
            timeout=120,
            stream=True,
            headers={'Accept': 'text/event-stream'}
        )
        
        if response.status_code != 200:
            st.session_state.is_processing = False
            return {
                "status": "error",
                "error": f"HTTP {response.status_code}",
                "message": f"Server returned: {response.text}"
            }
        
        # Parse SSE stream and capture workflow progress
        client = sseclient.SSEClient(response)
        final_result = None
        current_step = "🚀 Starting AI analysis..."
        step_count = 0
        total_steps = 5
        workflow_events = []  # Capture all events for progress display
        
        # Create simple real-time event display with loading indicator
        events_display = None
        if progress_container:
            with progress_container:
                events_display = st.empty()
                # Show loading spinner with message
                events_display.markdown("""
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; border: 2px solid #f3f3f3; border-top: 2px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    <span><strong>🚀 Starting analysis...</strong></span>
                </div>
                <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                </style>
                """, unsafe_allow_html=True)
                print(f"[DEBUG] Created events display with loading spinner")
        else:
            print(f"[DEBUG] No progress_container provided, events display will not be shown")
        
        # Add timeout and debugging
        import time
        start_time = time.time()
        timeout = 120  # 2 minutes
        events_received = 0
        
        print(f"[DEBUG] Starting SSE stream processing...")
        
        for event in client.events():
            try:
                # Check if user requested to stop
                if st.session_state.stop_processing:
                    print(f"[DEBUG] Processing stopped by user")
                    if events_display:
                        events_display.markdown("🛑 **Stopped by user**")
                    st.session_state.is_processing = False
                    return {
                        "status": "stopped",
                        "message": "Processing stopped by user"
                    }
                
                # Check for timeout
                if time.time() - start_time > timeout:
                    print(f"[DEBUG] SSE stream timeout after {timeout}s")
                    if events_display:
                        events_display.markdown("⏰ **Stream timeout - falling back to regular query**")
                    break
                
                events_received += 1
                print(f"[DEBUG] Received SSE event #{events_received}: {event.event}, data length: {len(event.data) if event.data else 0}")
                
                # Skip empty events
                if not event.data or not event.data.strip():
                    print(f"[DEBUG] Skipping empty event #{events_received}")
                    continue
                
                # Try to parse JSON data and display it simply
                try:
                    event_data = json.loads(event.data)
                    event_type = event_data.get("type", "")
                    
                    # Debug logging to see event structure
                    print(f"[DEBUG] Event type: {event_type}")
                    print(f"[DEBUG] Raw event data: {json.dumps(event_data, indent=2)}")
                    
                    # Skip debug events completely
                    if event_type == "enhanced_event" and event_data.get("event_type") == "debug_info":
                        print(f"[DEBUG] Skipping debug_info event")
                        continue
                    
                    # Capture event for final result
                    captured_event = {
                        "type": event_type,
                        "data": event_data,
                        "timestamp": datetime.now().isoformat()
                    }
                    workflow_events.append(captured_event)
                    
                    # ULTRA SIMPLIFIED: Just display server messages with smart deduplication
                    if events_display and event_type:
                        display_message = ""
                        should_display = False
                        
                        if event_type == "enhanced_step_update":
                            # Get the step info
                            step_data = event_data.get("current_step", {})
                            step_message = step_data.get("message", "")
                            status = step_data.get("status", "")
                            
                            # Only display if the step has a meaningful message and is complete or significant
                            if step_message and (status == "complete" or "started" in step_message.lower()):
                                # Use the server's message as-is (it already has emojis)
                                display_message = step_message
                                should_display = True
                        
                        elif event_type == "workflow_completed":
                            display_message = "🎉 Analysis completed!"
                            should_display = True
                            # Clear the spinner for completion
                            if events_display:
                                events_display.markdown(f"**✅ {display_message}**")
                                should_display = False  # Don't show again below
                        
                        elif event_type == "enhanced_completed":
                            display_message = "🎉 Enhanced analysis completed!"
                            # Clear the spinner for completion
                            if events_display:
                                events_display.markdown(f"**✅ {display_message}**")
                                should_display = False  # Don't show again below
                        
                        elif event_type == "connection_established":
                            display_message = event_data.get("message", "🔗 Connection established")
                            should_display = True
                        
                        # Display only if we determined it should be shown
                        if should_display and display_message:
                            # Show loading spinner with the message
                            spinner_html = """
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div style="width: 16px; height: 16px; border: 2px solid #f3f3f3; border-top: 2px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                                <span><strong>{}</strong></span>
                            </div>
                            <style>
                            @keyframes spin {{
                                0% {{ transform: rotate(0deg); }}
                                100% {{ transform: rotate(360deg); }}
                            }}
                            </style>
                            """.format(display_message)
                            events_display.markdown(spinner_html, unsafe_allow_html=True)
                        
                        # Check for completion events
                        if event_type in ["workflow_completed", "enhanced_completed"]:
                            final_result = event_data
                            break
                
                except json.JSONDecodeError:
                    continue  # Skip non-JSON events
                except Exception as e:
                    print(f"Error processing SSE event: {e}")
                    continue
            
            except Exception as e:
                print(f"Error in event loop: {e}")
                continue
                
        
        # Check if we received events but no final result
        if workflow_events and not final_result:
            print(f"[DEBUG] Received {len(workflow_events)} events but no final result. Last few events:")
            for event in workflow_events[-3:]:
                if isinstance(event, dict):
                    event_type = event.get('type', 'unknown')
                    print(f"[DEBUG] Event: {event_type}")
                    if event_type == 'workflow_completed':
                        final_result = event.get('data', {})
        
        if final_result:
            # Add workflow events to the result for display
            final_result["workflow_events"] = workflow_events
            print(f"[DEBUG] *** FINAL RESULT WITH {len(workflow_events)} WORKFLOW EVENTS ***")
            st.session_state.is_processing = False
            return {
                "status": "success",
                "enhanced_result": final_result
            }
        elif workflow_events:
            # We got events but no final result - try to create a response from events
            print(f"[DEBUG] No final result but got {len(workflow_events)} events, falling back to regular query")
            if events_display:
                events_display.markdown("🔄 **Stream incomplete, falling back to regular query...**")
            st.session_state.is_processing = False
            return send_query_fallback(query, conversation_id)
        else:
            # No events received at all - fallback to regular endpoint
            print(f"[DEBUG] No events received, falling back to regular query")
            if events_display:
                events_display.markdown("🔄 **No streaming data received, using regular query...**")
            st.session_state.is_processing = False
            return send_query_fallback(query, conversation_id)
            
    except Exception as e:
        print(f"Streaming failed, falling back to regular query: {e}")
        st.session_state.is_processing = False
        return send_query_fallback(query, conversation_id)

def render_workflow_progress(workflow_events: List[Dict[str, Any]], workflow_result: Dict[str, Any], is_current_session: bool = False):
    """Render detailed workflow progress in a collapsible section - SIMPLIFIED VERSION."""
    if not workflow_events:
        print(f"[DEBUG] No workflow events to display")
        return
    
    print(f"[DEBUG] Rendering workflow progress with {len(workflow_events)} events, current_session={is_current_session}")
    
    # Use different styling for current vs past workflows
    if is_current_session:
        expander_label = "📋 Current Workflow Steps & Progress"
        expanded_default = True
    else:
        expander_label = "📋 Previous Workflow Steps"
        expanded_default = False
    
    # Use expander without key parameter (not supported in this Streamlit version)
    with st.expander(expander_label, expanded=expanded_default):
        st.write("**🚀 Enhanced multi-step workflow execution**")
        
        # ULTRA SIMPLIFIED: Just display meaningful steps without duplication
        step_number = 0
        displayed_steps = set()  # Track displayed steps to avoid duplicates
        
        for event in workflow_events:
            if not isinstance(event, dict):
                continue
                
            event_type = event.get("type", "unknown")
            event_data = event.get("data", {})
            
            # Skip debug events
            if event_type == "enhanced_event" and event_data.get("event_type") == "debug_info":
                continue
                
            display_message = ""
            should_display = False
            
            if event_type == "enhanced_step_update":
                # Get the step info
                step_data = event_data.get("current_step", {})
                step_message = step_data.get("message", "")
                status = step_data.get("status", "")
                
                # Only display if the step has a meaningful message and is complete
                if step_message and status == "complete":
                    # Use the server's message as-is (it already has emojis)
                    display_message = step_message
                    
                    # Check if we've already displayed this step
                    if display_message not in displayed_steps:
                        should_display = True
                        displayed_steps.add(display_message)
            
            elif event_type == "connection_established":
                display_message = event_data.get("message", "🔗 Connection established")
                if display_message not in displayed_steps:
                    should_display = True
                    displayed_steps.add(display_message)
                    
            elif event_type == "workflow_completed":
                display_message = "🎉 Analysis completed!"
                if display_message not in displayed_steps:
                    should_display = True
                    displayed_steps.add(display_message)
                    
            elif event_type == "enhanced_completed":
                display_message = "🎉 Enhanced analysis completed!"
                if display_message not in displayed_steps:
                    should_display = True
                    displayed_steps.add(display_message)
            
            # Display the message if it should be shown
            if should_display and display_message:
                step_number += 1
                st.write(f"**{step_number}.** {display_message}")
        
        # Show simple summary
        st.write("---")
        st.write(f"**📊 Summary:** {step_number} events processed")
        
        # Show execution time if available
        if workflow_result:
            total_time = workflow_result.get("total_execution_time", 0) or workflow_result.get("execution_time", 0)
            if total_time > 0:
                st.write(f"**⏱️ Total Time:** {total_time:.2f}s")

def send_query_fallback(query: str, conversation_id: str = None) -> Dict[str, Any]:
    """Fallback method for sending queries without streaming."""
    try:
        # Prepare conversation history for context
        conversation_history = []
        if conversation_id and st.session_state.current_conversation_history:
            # Include recent conversation history for context (last 10 messages)
            recent_history = st.session_state.current_conversation_history[-10:]
            conversation_history = [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp", "")
                }
                for msg in recent_history
            ]
        
        payload = {
            "query": query,
            "session_id": st.session_state.current_session_id,
            "conversation_id": conversation_id,
            "conversation_history": conversation_history,
            "include_analysis": True,
            "max_steps": 5
        }
        
        print(f"[DEBUG] Sending query with {len(conversation_history)} history messages")
        
        response = requests.post(
            f"{BACKEND_URL}/enhanced/query",
            json=payload,
            timeout=600
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "error": f"HTTP {response.status_code}",
                "message": f"Server returned: {response.text}"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to send query"
        }

def send_query(query: str, conversation_id: str = None) -> Dict[str, Any]:
    """Send a query to the enhanced orchestrator (compatibility wrapper)."""
    return send_query_fallback(query, conversation_id)

def render_sidebar():
    """Render the sidebar with conversation management."""
    with st.sidebar:
        # Add sidebar styling
        st.markdown(
            """
            <style>
            /* Modern sidebar styling similar to ChatGPT */
            .css-1d391kg {
                background-color: #171717;
                color: white;
                border-right: 1px solid #333;
            }
            
            .sidebar-title {
                font-size: 0.875rem;
                font-weight: 600;
                color: #9ca3af;
                margin-bottom: 0.75rem;
                padding: 0 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            
            /* New chat button styling - clean and visible */
            .css-1d391kg .stButton:first-child > button {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                padding: 12px !important;
                background-color: transparent !important;
                border: 1px solid #4b5563 !important;
                border-radius: 8px !important;
                color: white !important;
                text-decoration: none !important;
                font-size: 14px !important;
                font-weight: 500 !important;
                margin-bottom: 16px !important;
                cursor: pointer !important;
                transition: all 0.2s !important;
                width: 100% !important;
                text-align: center !important;
            }
            
            .css-1d391kg .stButton:first-child > button:hover {
                background-color: rgba(255, 255, 255, 0.05) !important;
                border-color: #6b7280 !important;
                color: white !important;
            }
            
            /* Chat history item styling - clean links */
            .chat-item {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0.75rem;
                margin-bottom: 0.25rem;
                border-radius: 8px;
                background-color: transparent;
                color: white;
                text-decoration: none;
                cursor: pointer;
                transition: all 0.2s;
                font-size: 0.875rem;
                width: 100%;
                border: none;
                outline: none;
            }
            
            .chat-item:hover {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
            }
            
            .chat-item.active {
                background-color: rgba(59, 130, 246, 0.3);
                color: white;
            }
            
            .chat-title-text {
                flex: 1;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                margin-right: 0.5rem;
                color: white;
                text-align: left;
            }
            
            /* Section divider */
            .section-divider {
                border-top: 1px solid #374151;
                margin: 1rem 0;
            }
            
            /* Settings expander styling */
            .css-1d391kg .streamlit-expander {
                background-color: transparent;
                border: 1px solid #374151;
                border-radius: 6px;
            }
            
            .css-1d391kg .streamlit-expander .streamlit-expanderHeader {
                color: #9ca3af;
                font-size: 0.875rem;
            }
            
            /* Override Streamlit button styles in sidebar */
            .css-1d391kg .stButton > button {
                background-color: transparent !important;
                border: none !important;
                color: #e5e7eb !important;
                border-radius: 8px !important;
                padding: 10px 12px !important;
                font-size: 14px !important;
                font-weight: 400 !important;
                width: 100% !important;
                text-align: left !important;
                transition: all 0.2s !important;
                margin: 2px 0 !important;
                cursor: pointer !important;
                box-shadow: none !important;
                outline: none !important;
            }
            
            .css-1d391kg .stButton > button:hover {
                background-color: rgba(255, 255, 255, 0.1) !important;
                border: none !important;
                color: white !important;
                box-shadow: none !important;
            }
            
            .css-1d391kg .stButton > button:focus {
                box-shadow: none !important;
                border: none !important;
                outline: none !important;
                background-color: rgba(255, 255, 255, 0.1) !important;
            }
            
            .css-1d391kg .stButton > button:active {
                background-color: rgba(255, 255, 255, 0.15) !important;
                border: none !important;
                transform: none !important;
                box-shadow: none !important;
            }
            
            /* Primary button for current conversation */
            .css-1d391kg .stButton > button[kind="primary"] {
                background-color: rgba(59, 130, 246, 0.2) !important;
                color: white !important;
            }
            
            .css-1d391kg .stButton > button[kind="primary"]:hover {
                background-color: rgba(59, 130, 246, 0.3) !important;
            }
            
            /* Delete button styling */
            .css-1d391kg .stButton > button[aria-label*="Delete"] {
                background-color: transparent !important;
                color: #ef4444 !important;
                font-size: 16px !important;
                padding: 8px !important;
                width: 32px !important;
                height: 32px !important;
                border-radius: 6px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
            
            .css-1d391kg .stButton > button[aria-label*="Delete"]:hover {
                background-color: rgba(239, 68, 68, 0.1) !important;
                color: #dc2626 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="sidebar-title">Chats</div>', unsafe_allow_html=True)
        
        # Load and display conversations
        if st.session_state.current_session_id:
            conversations = load_conversations()
            st.session_state.conversations = conversations
            
            # New Chat button
            if st.button("✏️ New chat", key="new_chat_main", use_container_width=True):
                conversation_id = create_new_conversation()
                if conversation_id:
                    st.success("New conversation created!")
                    st.rerun()
            
            if conversations:
                for conv in conversations:
                    conv_id = conv.get("conversation_id", "")
                    title = conv.get("title", "New Chat")
                    last_activity = conv.get("last_activity", "")
                    
                    # Format last activity
                    try:
                        if last_activity:
                            dt = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                            time_str = dt.strftime("%m/%d")
                        else:
                            time_str = "Today"
                    except:
                        time_str = "Today"
                    
                    # Check if this is the current conversation
                    is_current = conv_id == st.session_state.current_conversation_id
                    
                    # Create conversation item with columns for layout
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        # Use different styling for current vs other conversations
                        button_type = "primary" if is_current else "secondary"
                        if st.button(
                            title,
                            key=f"conv_{conv_id}",
                            type=button_type,
                            help=f"Last activity: {time_str}",
                            use_container_width=True
                        ):
                            # Switch to this conversation
                            print(f"[DEBUG] Switching to conversation: {conv_id}")
                            st.session_state.current_conversation_id = conv_id
                            
                            # Load conversation history from SQLite
                            messages = conversation_storage.get_conversation_messages(conv_id)
                            st.session_state.current_conversation_history = [
                                {
                                    "role": msg["role"],
                                    "content": msg["content"],
                                    "timestamp": msg["created_at"],
                                    "metadata": msg.get("metadata", {})
                                }
                                for msg in messages
                            ]
                            
                            print(f"[DEBUG] Loaded {len(st.session_state.current_conversation_history)} messages for conversation {conv_id}")
                            
                            # Reset auto-load flag so conversations can be switched manually
                            st.session_state.conversation_auto_loaded = True
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑", key=f"del_{conv_id}", help="Delete conversation"):
                            # Delete conversation
                            conversation_storage.delete_conversation(conv_id)
                            st.success("Deleted!")
                            
                            # If this was the current conversation, clear it
                            if conv_id == st.session_state.current_conversation_id:
                                st.session_state.current_conversation_id = None
                                st.session_state.current_conversation_history = []
                            
                            # Refresh conversations list
                            st.session_state.conversations = load_conversations()
                            st.rerun()
                    
            else:
                st.info("No conversations yet")
        
        # Conversation summarization feature for current conversation
        if (st.session_state.current_conversation_id and 
            len(st.session_state.current_conversation_history) >= 4):
            st.markdown("---")
            if st.button("📝 Summarize Chat", use_container_width=True):
                summary = conversation_storage.generate_conversation_summary(
                    st.session_state.current_conversation_id
                )
                if summary:
                    st.success(f"💡 {summary}")
                else:
                    st.info("No summary generated for this conversation.")
        
        st.markdown("---")
        
        # Simple session controls at bottom
        if st.session_state.current_session_id:
            with st.expander("⚙️ Settings"):
                st.session_state.show_debug = st.checkbox(
                    "Debug Mode", 
                    value=st.session_state.show_debug
                )
                
                if st.button("🔄 Reset Session"):
                    # Clear session state and reinitialize
                    st.session_state.mcp_session_initialized = False
                    st.session_state.current_session_id = None
                    st.session_state.current_conversation_id = None
                    st.session_state.conversations = []
                    st.rerun()

def render_chat_interface():
    """Render the main chat interface."""
    # Add ChatGPT-like styling
    st.markdown(
        """
        <style>
        /* Hide the default Streamlit header and footer */
        header[data-testid="stHeader"] {
            height: 0px;
            background: transparent;
        }
        
        /* Main container styling */
        .main .block-container {
            max-width: 48rem;
            margin: 0 auto;
            padding: 1rem;
            padding-bottom: 8rem;
        }
        
        /* Chat message styling */
        .stChatMessage {
            margin: 1rem 0;
            background: transparent;
        }
        
        /* Fix input container styling - remove white background */
        .stChatInput {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: transparent;
            padding: 1rem;
            z-index: 999;
        }
        
        .stChatInput > div {
            max-width: 48rem;
            margin: 0 auto;
            background: rgb(38, 39, 48);
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            min-height: 5rem;
        }
        
        /* Input field styling */
        .stChatInput input {
            border: none;
            background: transparent;
            font-size: 1rem;
            padding: 0.75rem;
        }
        
        /* Custom title styling */
        .chat-title {
            text-align: center;
            font-size: 2rem;
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            color: #1f2937;
        }
        
        /* Welcome message styling */
        .welcome-container {
            text-align: center;
            margin: 4rem 0;
            padding: 2rem;
            min-height: 60vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .welcome-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 1rem;
        }
        
        .welcome-subtitle {
            font-size: 1rem;
            color: #6b7280;
            line-height: 1.5;
            max-width: 32rem;
            margin: 0 auto;
        }
        
        /* Better sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main .block-container {
                max-width: 100%;
                padding: 0.5rem;
                padding-bottom: 8rem;
            }
            
            .stChatInput > div {
                margin: 0 0.5rem;
                max-width: calc(100% - 1rem);
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Custom title
    st.markdown('<div class="chat-title">🚀 Enhanced MCP Assistant</div>', unsafe_allow_html=True)
    
    # Initialize session if needed
    session_data = get_or_create_session()
    
    # Auto-load the most recent conversation immediately after session initialization
    auto_load_recent_conversation()
    
    # Display session status
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
    
    # Check if this is a fresh start (no conversation history)
    has_messages = bool(st.session_state.current_conversation_history)
    
    if has_messages:
        # Existing conversation: Show history first, then input at bottom
        # Display conversation history
        for message in st.session_state.current_conversation_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Show workflow progress for assistant messages (collapsed for old messages)
                    if "metadata" in message:
                        metadata = message["metadata"]
                        workflow_events = metadata.get("workflow_events", [])
                        if workflow_events:
                            # Create a simplified workflow result for past messages
                            workflow_result = {
                                "workflow_id": metadata.get("workflow_id"),
                                "agents_used": metadata.get("agents_used", []),
                                "total_execution_time": metadata.get("execution_time", 0),
                                "total_steps": len([e for e in workflow_events if isinstance(e, dict) and e.get("type") == "step_completed"])
                            }
                            # Only show if not currently processing (to avoid duplication with current workflow)
                            if not st.session_state.is_processing:
                                render_workflow_progress(workflow_events, workflow_result, is_current_session=False)
                    
                    # Show additional debug info if available
                    if "metadata" in message:
                        metadata = message["metadata"]
                        if st.session_state.show_debug and metadata:
                            with st.expander("📊 Response Details"):
                                st.json(metadata)

        # Chat input at bottom for existing conversations
        if st.session_state.is_processing:
            user_input = st.chat_input(
                "⏳ Processing your request... Please wait",
                disabled=True
            )
            # Show stop button below the input in a centered way
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("🛑 Stop", type="secondary", use_container_width=True):
                    st.session_state.stop_processing = True
                    st.session_state.is_processing = False
                    st.rerun()
        else:
            user_input = st.chat_input("Ask me anything about your data...")
    
    else:
        # New conversation: Show welcome message in center and input in middle
        st.markdown(
            """
            <div class="welcome-container">
                <div class="welcome-title">👋 Welcome to Enhanced MCP Assistant</div>
                <div class="welcome-subtitle">Ask me anything about your data and I'll help you analyze it using multiple database agents with intelligent orchestration.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Input for new conversations (should be centered and prominent)
        if st.session_state.is_processing:
            user_input = st.chat_input(
                "⏳ Processing your request... Please wait",
                disabled=True
            )
            # Center the stop button below the input
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("🛑 Stop Processing", type="secondary", use_container_width=True):
                    st.session_state.stop_processing = True
                    st.session_state.is_processing = False
                    st.rerun()
        else:
            user_input = st.chat_input("Ask me anything about your data...")
    
    # Handle user input (same logic for both cases)
    if user_input:
        print(f"[DEBUG] ========== NEW QUERY STARTED ==========")
        print(f"[DEBUG] Query: {user_input}")
        print(f"[DEBUG] Current processing state: {st.session_state.is_processing}")
        
        # Prevent multiple simultaneous queries
        if st.session_state.is_processing:
            st.warning("⚠️ Please wait for the current query to complete before sending another.")
            return
        
        # Set processing state immediately
        st.session_state.is_processing = True
        
        # Ensure we have a conversation
        if not st.session_state.current_conversation_id:
            conversation_id = create_new_conversation()
            if not conversation_id:
                st.error("Failed to create conversation. Please try again.")
                st.session_state.is_processing = False  # Reset on error
                return
        
        # Add user message to history and SQLite immediately
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.current_conversation_history.append(user_message)
        
        print(f"[DEBUG] Processing user input: {user_input[:50]}...")
        print(f"[DEBUG] Conversation history length before: {len(st.session_state.current_conversation_history)}")
        
        # Store in SQLite
        conversation_storage.add_message(
            st.session_state.current_conversation_id,
            "user",
            user_input
        )
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Send query and display response
        with st.chat_message("assistant"):
            # Create a placeholder for progress display
            progress_placeholder = st.empty()
            
            print(f"[DEBUG] About to send query: {user_input}")
            print(f"[DEBUG] Conversation ID: {st.session_state.current_conversation_id}")
            print(f"[DEBUG] Session ID: {st.session_state.current_session_id}")
            
            # Use streaming method for real-time progress
            response = send_query_with_streaming(
                user_input, 
                st.session_state.current_conversation_id,
                progress_placeholder
            )
            
            # Clear progress display after completion
            progress_placeholder.empty()
            
            print(f"[DEBUG] Got response: {response.get('status', 'no_status')}")
            
            if response.get("status") == "success":
                # Handle different response formats
                if "enhanced_result" in response:
                    # Streaming response format
                    workflow_result = response.get("enhanced_result", {})
                    response_text = workflow_result.get("final_answer", "Analysis completed successfully!")
                else:
                    # Fallback response format (direct from backend)
                    result = response.get("result", {})
                    workflow_result = result
                    response_text = result.get("greeting") or result.get("final_answer", "Analysis completed successfully!")
                
                print(f"[DEBUG] Displaying response: {response_text[:100]}...")
                print(f"[DEBUG] Response length: {len(response_text)}")
                
                # Display the response directly (not in a separate container)
                st.write(response_text)
                
                # Enhanced visualization for data results
                execution_results = workflow_result.get("results", [])
                if execution_results:
                    # Filter successful results with data
                    data_results = [r for r in execution_results if r.get("success") and r.get("data")]
                    
                    if data_results:
                        st.markdown("### 📊 Data Visualization")
                        render_enhanced_results(data_results)
                        print(f"[DEBUG] Rendered data results: {len(data_results)} items")
                
                # Show detailed workflow progress (expanded by default)
                workflow_events = workflow_result.get("workflow_events", [])
                if workflow_events:
                    render_workflow_progress(workflow_events, workflow_result, is_current_session=True)
                    print(f"[DEBUG] Rendered workflow progress: {len(workflow_events)} events")
                elif st.session_state.get('show_debug', False):
                    st.info("Using fallback method - no workflow events available")
                
                # Add to conversation history and SQLite
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "workflow_id": workflow_result.get("workflow_id"),
                        "agents_used": workflow_result.get("agents_used", []),
                        "execution_time": workflow_result.get("total_execution_time", 0),
                        "workflow_events": workflow_events,  # Store events for future viewing
                        "fallback_used": True  # Mark that fallback was used
                    }
                }
                st.session_state.current_conversation_history.append(assistant_message)
                
                print(f"[DEBUG] Added assistant message to history")
                print(f"[DEBUG] Conversation history length after: {len(st.session_state.current_conversation_history)}")
                
                # Store in SQLite
                conversation_storage.add_message(
                    st.session_state.current_conversation_id,
                    "assistant",
                    response_text,
                    assistant_message["metadata"]
                )
                
                # Generate conversation title if this is the first exchange
                if len(st.session_state.current_conversation_history) == 2:  # User + Assistant message
                    try:
                        title = conversation_storage.generate_conversation_title_with_llm(
                            st.session_state.current_conversation_id,
                            BACKEND_URL
                        )
                        st.success(f"💬 Conversation titled: '{title}'")
                    except Exception as e:
                        print(f"[DEBUG] Title generation failed: {e}")
                        # Fallback to simple title generation
                        title = conversation_storage.generate_conversation_title(
                            st.session_state.current_conversation_id
                        )
                        st.success(f"💬 Conversation titled: '{title}' (fallback)")
                    
                    # Refresh conversations to show new title
                    st.session_state.conversations = load_conversations()
            
            else:
                # Handle error
                error_msg = response.get("message", "Unknown error occurred")
                st.error(f"❌ {error_msg}")
                
                if st.session_state.show_debug:
                    st.json(response)
        
        # Update last query time
        st.session_state.last_query_time = time.time()
        
        # Reset processing state
        st.session_state.is_processing = False
        st.session_state.stop_processing = False
        st.session_state.current_request_id = None
        
        # Refresh conversations list to update activity
        st.session_state.conversations = load_conversations()
        
        print(f"[DEBUG] ========== QUERY COMPLETED ==========")
        print(f"[DEBUG] Final conversation history length: {len(st.session_state.current_conversation_history)}")
        
        # Temporarily disable rerun to test if it's causing the issue
        # if len(st.session_state.current_conversation_history) <= 2:  # User + Assistant message
        #     st.rerun()

def main():
    """Main application function."""
    init_streamlit_config()
    initialize_session_state()
    
    # Initialize session early to ensure proper conversation loading
    if not st.session_state.mcp_session_initialized:
        get_or_create_session()
    
    # Render sidebar and main interface
    render_sidebar()
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🚀 Enhanced MCP Assistant | "
        "Powered by Multi-Agent Orchestration | "
        f"Session: {st.session_state.current_session_id[:8] if st.session_state.current_session_id else 'Not initialized'}..."
    )

if __name__ == "__main__":
    main()
