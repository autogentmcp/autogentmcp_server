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
        
        # Debug logging to see what events we're receiving
        print(f"[ProgressDisplay] Received event: type={event_type}, data keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'not_dict'}")
        
        if event_type == "enhanced_step_update":
            step_data = event_data.get("current_step", {})
            step_message = step_data.get("message", "")
            status = step_data.get("status", "")
            details = step_data.get("details", {})
            event_subtype = step_data.get("event_type", "")
            
            print(f"[ProgressDisplay] Enhanced step: subtype={event_subtype}, message={step_message[:50]}..., status={status}")
            
            # Handle specific detailed events
            if event_subtype == "workflow_starting":
                display_message = "🔄 Initializing workflow execution..."
                should_display = True
            
            elif event_subtype == "workflow_started":
                display_message = "🚀 Simple Orchestrator workflow started"
                should_display = True
            
            elif event_subtype == "step_started":
                step_id = step_data.get("step_id", "")
                
                # First, try to get the description from the event data structure
                # The StreamEvent has data field with description
                description = ""
                
                # Check if there's a description in the details (from streaming service)
                details = step_data.get("details", {})
                if details and isinstance(details, dict):
                    description = details.get("description", "")
                
                # If no description in details, check the original event data
                if not description and "data" in event_data and isinstance(event_data["data"], dict):
                    description = event_data["data"].get("description", "")
                
                # Also check step_data for description
                if not description:
                    description = step_data.get("description", "")
                
                # Log to see what we're getting
                print(f"[ProgressDisplay] Step started - step_id: {step_id}, description: '{description}', details: {list(details.keys()) if details else 'none'}")
                
                if description:
                    display_message = description
                elif step_id == "analyze":
                    display_message = "🧠 Understanding your request"
                elif step_id == "execute":
                    display_message = "⚡ Executing agent plan"
                elif step_id == "respond":
                    display_message = "📝 Generating final response"
                elif step_id == "table_selection":
                    display_message = "📋 Analyzing relevant tables"
                elif step_id == "sql_generation":
                    display_message = "🔧 Generating SQL query"
                elif step_id == "query_execution":
                    display_message = "⚡ Executing database query"
                elif step_id == "result_processing":
                    display_message = "📊 Processing query results"
                else:
                    display_message = step_message or "▶️ Starting step"
                should_display = True
            
            elif event_subtype == "step_completed":
                step_id = step_data.get("step_id", "")
                # Use the dynamic description from the event data first, but modify for completion
                description = step_data.get("details", {}).get("description") or step_data.get("description", "")
                
                if description:
                    # Convert the description to a completion message
                    if description.startswith("📋"):
                        display_message = description.replace("📋 Analyzing", "✅ Analyzed")
                    elif description.startswith("🔧"):
                        display_message = description.replace("🔧 Generating", "✅ Generated")
                    elif description.startswith("⚡"):
                        display_message = description.replace("⚡ Executing", "✅ Executed")
                    elif description.startswith("📊"):
                        display_message = description.replace("📊 Processing", "✅ Processed")
                    else:
                        display_message = f"✅ {description.replace('📋 ', '').replace('🔧 ', '').replace('⚡ ', '').replace('📊 ', '')}"
                elif step_id == "analyze":
                    display_message = "✅ Request analysis complete"
                elif step_id == "execute":
                    display_message = "✅ Agent execution complete"
                elif step_id == "respond":
                    display_message = "✅ Response generation complete"
                elif step_id == "table_selection":
                    display_message = "✅ Table analysis complete"
                elif step_id == "sql_generation":
                    display_message = "✅ SQL query generated"
                elif step_id == "query_execution":
                    display_message = "✅ Database query executed"
                elif step_id == "result_processing":
                    display_message = "✅ Results processed successfully"
                else:
                    display_message = step_message or "✅ Step completed"
                should_display = True
            
            elif event_subtype == "agent_started":
                # Get message from event details
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    agent_name = details.get("agent_name", "")
                    agent_type = details.get("agent_type", "")
                    display_message = f"🚀 Starting agent: {agent_name} ({agent_type})"
                should_display = True
            
            elif event_subtype == "payload_generation":
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    agent_name = details.get("agent_name", "")
                    display_message = f"📦 Generating payload for {agent_name}"
                should_display = True
            
            elif event_subtype == "api_call":
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    endpoint = details.get("endpoint", "")
                    display_message = f"🌐 Calling service endpoint: {endpoint}"
                should_display = True
            
            elif event_subtype == "service_response":
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    agent_name = details.get("agent_name", "")
                    status_code = details.get("status_code", 0)
                    display_message = f"📨 Got response from {agent_name}: {status_code}"
                should_display = True
            
            elif event_subtype == "query_execution":
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    database_type = details.get("database_type", "")
                    display_message = f"⚡ Executing ({database_type}) query"
                should_display = True
            
            elif event_subtype == "query_results":
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    database_type = details.get("database_type", "")
                    row_count = details.get("row_count", 0)
                    display_message = f"📊 Retrieved {row_count} rows from {database_type}"
                should_display = True
            
            elif event_subtype == "sql_generated":
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    database_type = details.get("database_type", "")
                    display_message = f"✅ Generated SQL query for {database_type}"
                should_display = True
            
            elif event_subtype == "query_execution":
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    database_type = details.get("database_type", "")
                    display_message = f"⚡ Executing ({database_type}) query"
                should_display = True
            
            elif event_subtype == "query_results":
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    database_type = details.get("database_type", "")
                    row_count = details.get("row_count", 0)
                    display_message = f"📊 Retrieved {row_count} rows from {database_type}"
                should_display = True
            
            elif event_subtype == "payload_generation":
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    agent_name = details.get("agent_name", "")
                    display_message = f"📦 Generating payload for {agent_name}"
                should_display = True
            
            elif event_subtype == "api_call":
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    endpoint = details.get("endpoint", "")
                    display_message = f"🌐 Calling service endpoint: {endpoint}"
                should_display = True
            
            elif event_subtype == "service_response":
                details = step_data.get("details", {})
                display_message = details.get("message", "")
                if not display_message:
                    agent_name = details.get("agent_name", "")
                    status_code = details.get("status_code", 0)
                    display_message = f"📨 Got response from {agent_name}: {status_code}"
                should_display = True
            
            elif event_subtype == "workflow_completed":
                display_message = "✅ Analysis completed successfully"
                should_display = True
            
            elif event_subtype == "progress_update":
                # Handle progress updates during processing
                display_message = step_message or "📈 Processing..."
                should_display = True
            
            elif event_subtype == "debug_info":
                # Extract meaningful debug messages for dynamic details
                debug_message = step_message.replace("🔍 ", "").strip()
                
                # Log the full debug message for debugging
                print(f"[ProgressDisplay] Debug message received: '{debug_message}'")
                
                if debug_message and len(debug_message) > 3:  # Only show non-trivial messages
                    # Check for specific patterns we want to highlight
                    if "Starting agent:" in debug_message:
                        display_message = f"🚀 {debug_message}"
                        should_display = True
                    elif "Generating" in debug_message and "SQL" in debug_message:
                        display_message = f"🔧 {debug_message}"
                        should_display = True
                    elif "Executing" in debug_message and ("query" in debug_message.lower() or "bigquery" in debug_message.lower()):
                        display_message = f"⚡ {debug_message}"
                        should_display = True
                    elif "Retrieved" in debug_message and "rows" in debug_message:
                        display_message = f"� {debug_message}"
                        should_display = True
                    elif "Calling" in debug_message and ("service" in debug_message or "endpoint" in debug_message):
                        display_message = f"🌐 {debug_message}"
                        should_display = True
                    elif "Response" in debug_message and ("received" in debug_message or "status" in debug_message):
                        display_message = f"📨 {debug_message}"
                        should_display = True
                    else:
                        # Show other meaningful debug messages with debug icon
                        display_message = f"🔍 {debug_message}"
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
                    if event_subtype == "workflow_starting":
                        display_message = "🔄 Initializing workflow execution..."
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "workflow_started":
                        display_message = "🚀 Simple Orchestrator workflow started"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "step_started":
                        step_id = step_data.get("step_id", "")
                        # Use the message from the event data first
                        details = step_data.get("details", {})
                        display_message = details.get("message", "") or step_data.get("message", "")
                        
                        if not display_message:
                            if step_id == "analyze":
                                display_message = "🧠 Understanding your request"
                            elif step_id == "execute":
                                display_message = "⚡ Executing agent plan"
                            elif step_id == "respond":
                                display_message = "📝 Generating final response"
                            elif step_id == "table_selection":
                                display_message = "📋 Analyzing relevant tables"
                            elif step_id == "sql_generation":
                                display_message = "🔧 Generating SQL query"
                            elif step_id == "query_execution":
                                display_message = "⚡ Executing database query"
                            elif step_id == "result_processing":
                                display_message = "📊 Processing query results"
                            else:
                                display_message = step_message or "▶️ Starting step"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "step_completed":
                        step_id = step_data.get("step_id", "")
                        # Use the message from the event data first
                        details = step_data.get("details", {})
                        display_message = details.get("message", "") or step_data.get("message", "")
                        
                        if not display_message:
                            if step_id == "analyze":
                                display_message = "✅ Request analysis complete"
                            elif step_id == "execute":
                                display_message = "✅ Agent execution complete"
                            elif step_id == "respond":
                                display_message = "✅ Response generation complete"
                            elif step_id == "table_selection":
                                display_message = "✅ Table analysis complete"
                            elif step_id == "sql_generation":
                                display_message = "✅ SQL query generated"
                            elif step_id == "query_execution":
                                display_message = "✅ Database query executed"
                            elif step_id == "result_processing":
                                display_message = "✅ Results processed successfully"
                            else:
                                display_message = step_message or "✅ Step completed"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "agent_started":
                        details = step_data.get("details", {})
                        display_message = details.get("message", "")
                        if not display_message:
                            agent_name = details.get("agent_name", "")
                            agent_type = details.get("agent_type", "")
                            display_message = f"🚀 Starting agent: {agent_name} ({agent_type})"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "sql_generated":
                        details = step_data.get("details", {})
                        display_message = details.get("message", "")
                        if not display_message:
                            database_type = details.get("database_type", "")
                            display_message = f"✅ Generated SQL query for {database_type}"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "query_execution":
                        details = step_data.get("details", {})
                        display_message = details.get("message", "")
                        if not display_message:
                            database_type = details.get("database_type", "")
                            display_message = f"⚡ Executing ({database_type}) query"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "query_results":
                        details = step_data.get("details", {})
                        display_message = details.get("message", "")
                        if not display_message:
                            database_type = details.get("database_type", "")
                            row_count = details.get("row_count", 0)
                            display_message = f"📊 Retrieved {row_count} rows from {database_type}"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "payload_generation":
                        details = step_data.get("details", {})
                        display_message = details.get("message", "")
                        if not display_message:
                            agent_name = details.get("agent_name", "")
                            display_message = f"📦 Generating payload for {agent_name}"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "api_call":
                        details = step_data.get("details", {})
                        display_message = details.get("message", "")
                        if not display_message:
                            endpoint = details.get("endpoint", "")
                            display_message = f"🌐 Calling service endpoint: {endpoint}"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "service_response":
                        details = step_data.get("details", {})
                        display_message = details.get("message", "")
                        if not display_message:
                            agent_name = details.get("agent_name", "")
                            status_code = details.get("status_code", 0)
                            display_message = f"📨 Got response from {agent_name}: {status_code}"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "workflow_completed":
                        display_message = "✅ Analysis completed successfully"
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "progress_update":
                        # Handle progress updates during processing
                        display_message = step_message or "📈 Processing..."
                        if display_message not in displayed_steps:
                            should_display = True
                            displayed_steps.add(display_message)
                    
                    elif event_subtype == "debug_info":
                        # Extract meaningful debug messages for dynamic details
                        debug_message = step_message.replace("🔍 ", "").strip()
                        
                        if debug_message and len(debug_message) > 3:  # Only show non-trivial messages
                            # Check for specific patterns we want to highlight
                            if "Starting agent:" in debug_message:
                                display_message = f"🚀 {debug_message}"
                            elif "Generating" in debug_message and "SQL" in debug_message:
                                display_message = f"🔧 {debug_message}"
                            elif "Executing" in debug_message and ("query" in debug_message.lower() or "bigquery" in debug_message.lower()):
                                display_message = f"⚡ {debug_message}"
                            elif "Retrieved" in debug_message and "rows" in debug_message:
                                display_message = f"📊 {debug_message}"
                            elif "Calling" in debug_message and ("service" in debug_message or "endpoint" in debug_message):
                                display_message = f"🌐 {debug_message}"
                            elif "Response" in debug_message and ("received" in debug_message or "status" in debug_message):
                                display_message = f"📨 {debug_message}"
                            else:
                                display_message = f"🔍 {debug_message}"
                            
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
