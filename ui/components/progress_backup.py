"""
Progress display components for real-time workflow updates
"""

import streamlit as st
from typing import Dict, Any, List, Set
from datetime import datetime

from ..styles.styles import PROGRESS_STYLES


class ProgressDisplay:
    """Component for displaying workflow progress"""
    
    def __init__(self):
        self.current_message = ""
        self.events_displayed = set()
    
    def render_loading_spinner(self, container, message: str = "Starting analysis..."):
        """Render a loading spinner with message"""
        spinner_html = f"""
        <div style="display: flex; align-items: center; gap: 10px;">
            <div class="loading-spinner"></div>
            <span><strong>{message}</strong></span>
        </div>
        {PROGRESS_STYLES}
        """
        container.markdown(spinner_html, unsafe_allow_html=True)
    
    def update_progress(self, container, event_type: str, event_data: Dict[str, Any]):
        """Update progress display based on event"""
        display_message = ""
        should_display = False
        
        if event_type == "enhanced_step_update":
            step_data = event_data.get("current_step", {})
            step_message = step_data.get("message", "")
            status = step_data.get("status", "")
            details = step_data.get("details", {})
            event_subtype = step_data.get("event_type", "")
            
            # Handle specific detailed events
            if event_subtype == "agent_started":
                agent_name = details.get("agent_name", "")
                agent_type = details.get("agent_type", "")
                display_message = f"🚀 Starting agent: {agent_name} ({agent_type})"
                should_display = True
            
            elif event_subtype == "sql_generated":
                database_type = details.get("database_type", "")
                display_message = f"✅ Generated SQL query for {database_type}"
                should_display = True
            
            elif event_subtype == "query_execution":
                database_type = details.get("database_type", "")
                display_message = f"⚡ Executing ({database_type}) query"
                should_display = True
            
            elif event_subtype == "query_results":
                database_type = details.get("database_type", "")
                row_count = details.get("row_count", 0)
                display_message = f"📊 Retrieved {row_count} rows from {database_type}"
                should_display = True
            
            elif event_subtype == "payload_generation":
                agent_name = details.get("agent_name", "")
                display_message = f"📦 Generating payload for {agent_name}"
                should_display = True
            
            elif event_subtype == "api_call":
                endpoint = details.get("endpoint", "")
                display_message = f"🌐 Calling service endpoint: {endpoint}"
                should_display = True
            
            elif event_subtype == "service_response":
                agent_name = details.get("agent_name", "")
                status_code = details.get("status_code", 0)
                display_message = f"📨 Got response from {agent_name}: {status_code}"
                should_display = True
            
            # Only display meaningful completed steps if no specific handler above
            elif step_message and status == "complete" and not should_display:
                display_message = step_message
                should_display = True
        
        elif event_type == "connection_established":
            display_message = event_data.get("message", "🔗 Connection established")
            should_display = True
        
        elif event_type == "workflow_completed":
            display_message = "🎉 Analysis completed!"
            should_display = True
        
        elif event_type == "enhanced_completed":
            display_message = "🎉 Enhanced analysis completed!"
            should_display = True
        
        # Display if new and meaningful
        if should_display and display_message and display_message not in self.events_displayed:
            self.events_displayed.add(display_message)
            self.current_message = display_message
            self.render_loading_spinner(container, display_message)
    
    def render_completion(self, container, message: str = "Analysis completed!"):
        """Render completion state"""
        completion_html = f"""
        <div style="display: flex; align-items: center; gap: 10px; color: #10b981;">
            <span style="font-size: 1.2em;">✅</span>
            <span><strong>{message}</strong></span>
        </div>
        """
        container.markdown(completion_html, unsafe_allow_html=True)
    
    def clear(self):
        """Clear the progress display state"""
        self.current_message = ""
        self.events_displayed.clear()


class WorkflowProgressRenderer:
    """Component for rendering detailed workflow progress"""
    
    def render_workflow_progress(
        self, 
        workflow_events: List[Dict[str, Any]], 
        workflow_result: Dict[str, Any], 
        is_current_session: bool = False
    ):
        """Render detailed workflow progress in a collapsible section"""
        if not workflow_events:
            return
        
        # Use different labels for current vs past workflows
        if is_current_session:
            expander_label = "📋 Current Workflow Steps & Progress"
            expanded_default = True
        else:
            expander_label = "📋 Previous Workflow Steps"
            expanded_default = False
        
        with st.expander(expander_label, expanded=expanded_default):
            st.write("**🚀 Enhanced multi-step workflow execution**")
            
            # Display meaningful steps without duplication
            step_number = 0
            displayed_steps = set()
            
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
                    step_data = event_data.get("current_step", {})
                    step_message = step_data.get("message", "")
                    status = step_data.get("status", "")
                    details = step_data.get("details", {})
                    event_subtype = step_data.get("event_type", "")
                    
                    # Handle specific detailed events
                    if event_subtype == "agent_started":
                        agent_name = details.get("agent_name", "")
                        agent_type = details.get("agent_type", "")
                        display_message = f"🚀 Starting agent: {agent_name} ({agent_type})"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "sql_generated":
                        database_type = details.get("database_type", "")
                        display_message = f"✅ Generated SQL query for {database_type}"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "query_execution":
                        database_type = details.get("database_type", "")
                        display_message = f"⚡ Executing ({database_type}) query"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "query_results":
                        database_type = details.get("database_type", "")
                        row_count = details.get("row_count", 0)
                        display_message = f"📊 Retrieved {row_count} rows from {database_type}"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "payload_generation":
                        agent_name = details.get("agent_name", "")
                        display_message = f"📦 Generating payload for {agent_name}"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "api_call":
                        endpoint = details.get("endpoint", "")
                        display_message = f"🌐 Calling service endpoint: {endpoint}"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "service_response":
                        agent_name = details.get("agent_name", "")
                        status_code = details.get("status_code", 0)
                        display_message = f"📨 Got response from {agent_name}: {status_code}"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    # Only display meaningful completed steps if no specific handler above
                    elif step_message and status == "complete" and not should_display:
                        display_message = step_message
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                
                elif event_type == "connection_established":
                    display_message = event_data.get("message", "🔗 Connection established")
                    if display_message not in displayed_steps:
                        should_display = True
                        displayed_steps.add(display_message)
                
                elif event_type in ["workflow_completed", "enhanced_completed"]:
                    display_message = "🎉 Analysis completed!"
                    if display_message not in displayed_steps:
                        should_display = True
                        displayed_steps.add(display_message)
                
                # Display the message
                if should_display and display_message:
                    step_number += 1
                    st.write(f"**{step_number}.** {display_message}")
            
            # Show summary
            st.write("---")
            st.write(f"**📊 Summary:** {step_number} events processed")
            
            # Show execution time if available
            if workflow_result:
                total_time = (workflow_result.get("total_execution_time", 0) or 
                            workflow_result.get("execution_time", 0))
                if total_time > 0:
                    st.write(f"**⏱️ Total Time:** {total_time:.2f}s")
