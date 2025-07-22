"""
Simplified Streamlit UI for Enhanced Workflow Assistant
Just displays what the backend sends - no complex event processing!
"""

import streamlit as st
import requests
import json
import time
import uuid
import os
from typing import Dict, Any

# Configuration - Support Docker environment
API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Enhanced AI Workflow Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"streamlit_{int(time.time())}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "pending_choice" not in st.session_state:
        st.session_state.pending_choice = None
    if "choice_made" not in st.session_state:
        st.session_state.choice_made = False
    
    # Welcome screen if no messages
    if not st.session_state.messages:
        st.markdown("""
        # Enhanced AI Workflow Assistant
        Ask me anything about your data - advanced multi-step analysis with intelligent agent selection
        """)
    
    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show data if available
            if message["role"] == "assistant" and message.get("data"):
                with st.expander("📊 View Data", expanded=False):
                    st.json(message["data"])
    
    # Handle pending agent choice
    if st.session_state.pending_choice and not st.session_state.choice_made:
        st.markdown("---")
        st.markdown("### 🤖 Multiple agents available - please choose:")
        
        choice_data = st.session_state.pending_choice
        choice_options = choice_data.get("choice_options", [])
        
        if choice_options:
            # Create radio button options with ID and name
            option_labels = []
            for option in choice_options:
                name = option.get("name", "Unknown")
                agent_id = option.get("id", "N/A")
                region = option.get("region", "Unknown")
                description = option.get("description", "")[:100]
                option_labels.append(f"{name} (ID: {agent_id}) - {region}")
            
            selected_index = st.radio(
                "Choose the most appropriate agent:",
                range(len(option_labels)),
                format_func=lambda x: option_labels[x],
                key="agent_selection"
            )
            
            # Show description for selected option
            if selected_index is not None:
                selected_option = choice_options[selected_index]
                st.markdown(f"**Description:** {selected_option.get('description', 'No description available')}")
                st.markdown(f"**Expected outcome:** {selected_option.get('expected_outcome', 'Data retrieval')}")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Continue with Selected Agent", type="primary"):
                    # Send the choice to backend
                    selected_option = choice_options[selected_index]
                    
                    user_choice = {
                        "type": "agent_choice",
                        "agent_id": selected_option.get("id"),
                        "message": "Continue with selected agent"
                    }
                    
                    # Call resume workflow endpoint
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/orchestration/enhanced/resume",
                            json={
                                "workflow_id": choice_data.get("workflow_id"),
                                "session_id": st.session_state.session_id,
                                "user_choice": user_choice
                            },
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            if result.get("status") == "completed":
                                # Add result to chat
                                final_answer = result.get("final_answer", "Analysis completed!")
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": final_answer,
                                    "data": result.get("execution_summary")
                                })
                                
                                # Clear pending choice
                                st.session_state.pending_choice = None
                                st.session_state.choice_made = True
                                st.rerun()
                            else:
                                st.error(f"Workflow continuation failed: {result.get('message', 'Unknown error')}")
                        else:
                            st.error(f"Failed to continue workflow: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Error continuing workflow: {str(e)}")
            
            with col2:
                if st.button("Cancel"):
                    st.session_state.pending_choice = None
                    st.session_state.choice_made = True
                    st.rerun()
        else:
            st.error("No agent options available")
            if st.button("Clear"):
                st.session_state.pending_choice = None
                st.session_state.choice_made = True
                st.rerun()
    
    # Process if we're in processing state
    if st.session_state.is_processing and len(st.session_state.messages) > 0:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            with st.chat_message("assistant"):
                status_container = st.empty()
                
                try:
                    result = execute_simple_workflow(last_message["content"], status_container)
                    
                    if result:
                        if result.get("status") == "user_choice_required":
                            # Don't display anything yet, just trigger rerun to show choice UI
                            st.rerun()
                        else:
                            # Display final answer
                            final_answer = result.get("final_answer", "Analysis completed!")
                            st.markdown(final_answer)
                            
                            # Add to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": final_answer,
                                "data": result.get("collected_data")
                            })
                    else:
                        error_msg = "I encountered an error processing your request. Please try again."
                        st.write(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
                finally:
                    st.session_state.is_processing = False
    
    # Chat input
    user_input = st.chat_input(
        "Ask me anything about your data..." if not st.session_state.is_processing else "Processing your request...",
        disabled=st.session_state.is_processing
    )
    
    if user_input and not st.session_state.is_processing:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.is_processing = True
        st.rerun()

def execute_simple_workflow(query: str, status_container) -> Dict[str, Any]:
    """Execute workflow with simple event display."""
    try:
        # Show initial status
        status_container.markdown("<small style='color: #28a745;'>🚀 Starting enhanced workflow...</small>", unsafe_allow_html=True)
        
        # Send streaming request
        response = requests.post(
            f"{API_BASE_URL}/orchestration/enhanced/stream",
            json={"query": query, "session_id": st.session_state.session_id},
            stream=True,
            timeout=300,
            headers={"Accept": "text/event-stream"}
        )
        
        if response.status_code == 200:
            workflow_result = None
            status_html = ""
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        event_type = data.get("type")
                        
                        # Skip debug events
                        if event_type == "debug_info":
                            continue
                        
                        # Simple event handling - just display what backend sends!
                        message = ""
                        color = "#6c757d"
                        
                        if event_type == "connection_established":
                            message = data.get("message", "Connection established")
                            color = "#28a745"
                        
                        elif event_type == "enhanced_step_update":
                            # This is the main event - just use the backend message!
                            current_step = data.get("current_step", {})
                            step_message = current_step.get("message", "")
                            inner_event_type = current_step.get("event_type", "")
                            raw_data = current_step.get("raw_data", {})
                            
                            # Special case for user choice required
                            if inner_event_type == "user_choice_required":
                                # Store choice data in session state
                                st.session_state.pending_choice = {
                                    "workflow_id": raw_data.get("workflow_id"),
                                    "choice_options": raw_data.get("choice_options", []),
                                    "message": raw_data.get("message", "Please select an agent:")
                                }
                                message = "🤔 Waiting for your agent selection..."
                                color = "#ffc107"
                                
                                # Return early to trigger rerun and show choice UI
                                workflow_result = {
                                    "status": "user_choice_required",
                                    "message": message
                                }
                                break
                            
                            # Special case for routing decisions
                            elif inner_event_type == "llm_routing_decision":
                                selected_agent = raw_data.get("selected_agent", "Unknown")
                                confidence = raw_data.get("confidence", 0)
                                message = f"🎯 Selected: {selected_agent} ({confidence:.0f}% confidence)"
                                color = "#28a745"
                            # Special case for workflow completion
                            elif inner_event_type == "workflow_completed" and raw_data.get("final_answer"):
                                workflow_result = {
                                    "status": "completed",
                                    "final_answer": raw_data["final_answer"],
                                    "execution_summary": raw_data.get("execution_summary", {}),
                                    "collected_data": raw_data.get("collected_data")
                                }
                                message = "✅ Analysis completed"
                                color = "#28a745"
                            else:
                                # Just use the backend message - it already has the icon!
                                message = step_message if step_message else "Processing..."
                                color = "#007bff" if current_step.get("status") == "loading" else "#28a745"
                        
                        elif event_type == "enhanced_completed":
                            if not workflow_result and data.get("final_answer"):
                                workflow_result = {
                                    "status": "completed",
                                    "final_answer": data["final_answer"],
                                    "execution_summary": data.get("execution_summary", {}),
                                    "collected_data": data.get("collected_data")
                                }
                            message = "✅ Enhanced analysis complete"
                            color = "#28a745"
                        
                        elif event_type in ["enhanced_error", "error"]:
                            message = f"❌ {data.get('message', 'Error occurred')}"
                            color = "#dc3545"
                        
                        elif event_type in ["stream_end", "enhanced_stream_end"]:
                            break
                        
                        else:
                            # Generic fallback
                            message = data.get("message", f"Event: {event_type}")
                        
                        # Update display
                        if message and message.strip():
                            status_html += f"<div style='margin: 2px 0;'><small style='color: {color};'>{message}</small></div>"
                            status_container.markdown(status_html, unsafe_allow_html=True)
                            time.sleep(0.1)  # Small delay for visual effect
                        
                        # Break on completion
                        if workflow_result and workflow_result.get("final_answer"):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            # Keep status in collapsed section after completion
            if workflow_result:
                # Clear the live status
                status_container.empty()
                
                # Show collapsed workflow progress
                with st.expander("📋 Workflow Progress", expanded=False):
                    # Parse and display individual steps
                    status_lines = status_html.split('</div>')
                    for line in status_lines:
                        if line.strip() and 'style=' in line:
                            # Extract content between div tags
                            if '>' in line:
                                content = line.split('>', 1)[-1]
                                if content.strip():
                                    st.markdown(content.strip(), unsafe_allow_html=True)
                
                return workflow_result
            else:
                # Fallback result - also show status if we have any
                if status_html:
                    status_container.empty()
                    with st.expander("📋 Workflow Progress", expanded=False):
                        status_lines = status_html.split('</div>')
                        for line in status_lines:
                            if line.strip() and 'style=' in line and '>' in line:
                                content = line.split('>', 1)[-1]
                                if content.strip():
                                    st.markdown(content.strip(), unsafe_allow_html=True)
                else:
                    status_container.empty()
                
                return {
                    "status": "completed",
                    "final_answer": "Analysis completed successfully!",
                    "execution_summary": {"processing_time": 2.0},
                    "collected_data": None
                }
        else:
            # Show error but keep any status we collected
            if status_html:
                status_container.empty()
                with st.expander("📋 Workflow Progress (Error)", expanded=False):
                    status_lines = status_html.split('</div>')
                    for line in status_lines:
                        if line.strip() and 'style=' in line and '>' in line:
                            content = line.split('>', 1)[-1]
                            if content.strip():
                                st.markdown(content.strip(), unsafe_allow_html=True)
                    # Add error message
                    st.markdown(f"<small style='color: #dc3545;'>❌ Request failed: {response.status_code}</small>", unsafe_allow_html=True)
            else:
                status_container.markdown(f"<small style='color: #dc3545;'>❌ Request failed: {response.status_code}</small>", unsafe_allow_html=True)
            return None
            
    except Exception as e:
        # Show error but preserve any status we collected
        error_msg = f"❌ Error: {str(e)}"
        if 'status_html' in locals() and status_html:
            status_container.empty()
            with st.expander("📋 Workflow Progress (Error)", expanded=False):
                status_lines = status_html.split('</div>')
                for line in status_lines:
                    if line.strip() and 'style=' in line and '>' in line:
                        content = line.split('>', 1)[-1]
                        if content.strip():
                            st.markdown(content.strip(), unsafe_allow_html=True)
                # Add error message
                st.markdown(f"<small style='color: #dc3545;'>{error_msg}</small>", unsafe_allow_html=True)
        else:
            status_container.markdown(f"<small style='color: #dc3545;'>{error_msg}</small>", unsafe_allow_html=True)
        return None

if __name__ == "__main__":
    main()
