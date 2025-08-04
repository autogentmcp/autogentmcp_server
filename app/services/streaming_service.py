"""
Streaming service - handles real-time streaming functionality.
"""
from typing import AsyncGenerator, Dict, Any
from fastapi.responses import StreamingResponse
from app.workflows.workflow_streamer import workflow_streamer
from app.orchestrator import simple_orchestrator
from app.utils.data_conversion import convert_decimals_to_float
from app.models.requests import StreamingQueryRequest
import json
import uuid
import time
import asyncio


class StreamingService:
    """Service for handling real-time streaming of workflow execution."""
    
    def __init__(self):
        self.workflow_streamer = workflow_streamer
        self.orchestrator = simple_orchestrator
    
    async def create_streaming_response(
        self, 
        request: StreamingQueryRequest
    ) -> StreamingResponse:
        """
        Create a streaming response for query execution.
        
        Args:
            request: Streaming query request
            
        Returns:
            StreamingResponse with real-time updates
        """
        async def generate_enhanced_streaming():
            """Generate enhanced streaming response with multi-step orchestration."""
            client_id = str(uuid.uuid4())
            workflow_task = None
            workflow_completed = False
            
            try:
                # Subscribe to workflow events
                print(f"[StreamingService] Adding client {client_id} with session_id: {request.session_id}")
                self.workflow_streamer.add_client(client_id, session_id=request.session_id)
                
                # Send initial connection status
                yield f"data: {json.dumps({
                    'type': 'connection_established',
                    'message': '🚀 Simple orchestrator workflow initiated',
                    'timestamp': time.time()
                }, default=str)}\n\n"
                
                # Step mapping for UI display with enhanced details
                step_mapping = {
                    "workflow_started": {"name": "Workflow Started", "status": "loading", "message": "🚀 Simple Orchestrator workflow started", "next": "Understanding Request"},
                    "workflow_completed": {"name": "Analysis Complete", "status": "complete", "message": "✅ Analysis completed successfully", "next": "Finished"},
                    "error": {"name": "Error", "status": "error", "message": "❌ Error occurred", "next": "Review"},
                    "step_started": {"name": "Processing", "status": "loading", "message": "▶️ Starting step", "next": "In Progress"},
                    "step_completed": {"name": "Step Complete", "status": "complete", "message": "✅ Step completed", "next": "Next Step"},
                    "analyze": {"name": "Understanding Request", "status": "loading", "message": "🧠 Understanding your request", "next": "Planning Execution"},
                    "execute": {"name": "Executing Plan", "status": "loading", "message": "⚡ Executing agent plan", "next": "Generating Response"},
                    "respond": {"name": "Generating Response", "status": "loading", "message": "📝 Generating final response", "next": "Complete"},
                    
                    # Enhanced data agent steps
                    "table_selection": {"name": "Selecting Tables", "status": "loading", "message": "📋 Analyzing relevant tables", "next": "SQL Generation"},
                    "sql_generation": {"name": "Generating SQL", "status": "loading", "message": "🔧 Generating SQL query", "next": "Query Execution"},
                    "sql_generated": {"name": "SQL Generated", "status": "complete", "message": "✅ SQL query generated", "next": "Executing Query"},
                    "data_query": {"name": "Executing Query", "status": "loading", "message": "🗄️ Executing database query", "next": "Processing Results"},
                    "query_execution": {"name": "Query Execution", "status": "loading", "message": "⚡ Running query on database", "next": "Results Processing"},
                    "query_results": {"name": "Query Results", "status": "complete", "message": "📊 Retrieved query results", "next": "Analysis"},
                    "database_results": {"name": "Query Results", "status": "complete", "message": "📊 Retrieved query results", "next": "Analysis"},
                    
                    # Enhanced application agent steps
                    "agent_selection": {"name": "Agent Selection", "status": "loading", "message": "🎯 Selecting appropriate agent", "next": "Payload Generation"},
                    "agent_started": {"name": "Agent Started", "status": "loading", "message": "🚀 Starting agent execution", "next": "Processing"},
                    "payload_generation": {"name": "Generating Payload", "status": "loading", "message": "📦 Preparing API payload", "next": "Service Call"},
                    "api_call": {"name": "Calling Service", "status": "loading", "message": "🌐 Calling service endpoint", "next": "Response Processing"},
                    "service_response": {"name": "Service Response", "status": "complete", "message": "📨 Received service response", "next": "Analysis"},
                    
                    # LLM and routing steps
                    "llm_thinking": {"name": "AI Thinking", "status": "loading", "message": "🤔 AI analyzing request", "next": "Decision Making"},
                    "llm_routing_decision": {"name": "Route Selection", "status": "complete", "message": "🛣️ Selected execution path", "next": "Agent Execution"},
                    "route_selected": {"name": "Route Confirmed", "status": "complete", "message": "✅ Execution route confirmed", "next": "Processing"},
                    
                    # Progress and debug steps
                    "progress_update": {"name": "Progress Update", "status": "loading", "message": "📈 Progress update", "next": "Continuing"},
                    "debug_info": {"name": "Debug Info", "status": "info", "message": "🔍 Debug information", "next": "Processing"}
                }
                
                # Start simple workflow
                workflow_task = asyncio.create_task(
                    self.orchestrator.execute_workflow(
                        user_query=request.query,
                        session_id=request.session_id
                    )
                )
                
                # Send immediate workflow starting status
                yield f"data: {json.dumps({
                    'type': 'enhanced_step_update',
                    'current_step': {
                        'name': 'Workflow Starting',
                        'status': 'loading',
                        'message': '🔄 Initializing workflow execution...',
                        'event_type': 'workflow_starting',
                        'step_id': None,
                        'details': {}
                    },
                    'progress': {
                        'current_step_number': 0,
                        'estimated_total': 'Dynamic',
                        'workflow_type': 'simple_orchestration'
                    },
                    'timestamp': time.time()
                }, default=str)}\n\n"
                
                # Give the workflow task a moment to start
                await asyncio.sleep(0.1)
                
                # Stream workflow events
                events_processed = 0
                async for event_sse in self.workflow_streamer.stream_events(client_id):
                    events_processed += 1
                    
                    if event_sse.startswith("data: "):
                        try:
                            event_data = json.loads(event_sse[6:].strip())
                            event_type = event_data.get("type") or event_data.get("event_type")
                            
                            # Handle timeout and heartbeat events
                            if event_type == "timeout":
                                yield f"data: {json.dumps({
                                    'type': 'timeout',
                                    'message': event_data.get("message", "Process taking longer than expected"),
                                    'elapsed_seconds': event_data.get("elapsed_seconds", 0)
                                }, default=str)}\n\n"
                                break
                                
                            elif event_type == "heartbeat":
                                yield event_sse
                                continue
                            
                            # Map workflow events to UI steps with enhanced details
                            step_info = None
                            final_message = event_data.get('message', '')
                            additional_details = {}
                            
                            # Extract detailed information based on event type
                            if event_type == "step_started" or event_type == "step_completed":
                                step_id = event_data.get("step_id", "unknown")
                                if step_id in step_mapping:
                                    step_info = step_mapping[step_id]
                                    final_message = event_data.get('description', '') or step_info.get("message", "Processing...")
                                    
                            elif event_type == "sql_generated":
                                step_info = step_mapping.get("sql_generated", {"name": "SQL Generated", "status": "complete"})
                                sql_query = event_data.get('data', {}).get('sql_query', '')
                                database_type = event_data.get('data', {}).get('database_type', '')
                                reasoning = event_data.get('data', {}).get('llm_reasoning', '')
                                final_message = f"✅ Generated {database_type} query"
                                additional_details = {
                                    "sql_query": sql_query[:200] + "..." if len(sql_query) > 200 else sql_query,
                                    "database_type": database_type,
                                    "reasoning": reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
                                }
                                
                            elif event_type == "data_query":
                                step_info = step_mapping.get("data_query", {"name": "Executing Query", "status": "loading"})
                                query = event_data.get('data', {}).get('query', '')
                                database_type = event_data.get('data', {}).get('database_type', '')
                                final_message = f"🗄️ Executing {database_type} query"
                                additional_details = {
                                    "query_preview": query[:100] + "..." if len(query) > 100 else query,
                                    "database_type": database_type
                                }
                                
                            elif event_type == "llm_thinking":
                                step_info = step_mapping.get("llm_thinking", {"name": "AI Thinking", "status": "loading"})
                                thinking_about = event_data.get('data', {}).get('thinking_about', '')
                                final_message = f"🤔 {thinking_about}"
                                additional_details = {"thinking_about": thinking_about}
                                
                            elif event_type == "llm_routing_decision":
                                step_info = step_mapping.get("llm_routing_decision", {"name": "Route Selection", "status": "complete"})
                                route_data = event_data.get('data', {})
                                selected_agent = route_data.get('selected_agent', '')
                                confidence = route_data.get('confidence', 0)
                                reasoning = route_data.get('reasoning', '')
                                final_message = f"🛣️ Selected: {selected_agent} (confidence: {confidence:.0%})"
                                additional_details = {
                                    "selected_agent": selected_agent,
                                    "confidence": confidence,
                                    "reasoning": reasoning[:150] + "..." if len(reasoning) > 150 else reasoning
                                }
                                
                            elif event_type == "debug_info":
                                step_info = step_mapping.get("debug_info", {"name": "Debug Info", "status": "info"})
                                debug_message = event_data.get('data', {}).get('debug_message', '')
                                debug_data = event_data.get('data', {}).get('debug_data', {})
                                final_message = f"🔍 {debug_message}"
                                additional_details = {"debug_data": debug_data}
                                
                                # Special handling for specific debug steps
                                debug_message = event_data.get('data', {}).get('debug_message', '')
                                if "Starting agent:" in debug_message:
                                    step_info = {"name": "Agent Selected", "status": "loading", "message": debug_message}
                                elif "Generating payload" in debug_message:
                                    step_info = {"name": "Generating Payload", "status": "loading", "message": debug_message}
                                elif "Calling service endpoint" in debug_message:
                                    step_info = {"name": "Calling Service", "status": "loading", "message": debug_message}
                                elif "Received response" in debug_message:
                                    step_info = {"name": "Service Response", "status": "complete", "message": debug_message}
                                elif "Retrieved" in debug_message and "rows" in debug_message:
                                    step_info = {"name": "Query Results", "status": "complete", "message": debug_message}
                                else:
                                    final_message = f"🔍 {debug_message}"
                                
                            elif event_type == "progress_update":
                                step_info = step_mapping.get("progress_update", {"name": "Progress Update", "status": "loading"})
                                progress = event_data.get('data', {}).get('progress_percent', 0)
                                current_step = event_data.get('data', {}).get('current_step', '')
                                final_message = f"📈 {current_step} ({progress:.0f}%)"
                            
                            # Handle new detailed event types
                            elif event_type == "agent_started":
                                agent_name = event_data.get('data', {}).get('agent_name', '')
                                agent_type = event_data.get('data', {}).get('agent_type', '')
                                step_info = {"name": "Agent Started", "status": "loading", "event_type": "agent_started"}
                                final_message = f"🚀 Starting agent: {agent_name} ({agent_type})"
                                additional_details = {
                                    "agent_name": agent_name,
                                    "agent_type": agent_type
                                }
                            
                            elif event_type == "payload_generation":
                                agent_name = event_data.get('data', {}).get('agent_name', '')
                                step_info = {"name": "Generating Payload", "status": "loading", "event_type": "payload_generation"}
                                final_message = f"📦 Generating payload for {agent_name}"
                                additional_details = {"agent_name": agent_name}
                            
                            elif event_type == "api_call":
                                endpoint = event_data.get('data', {}).get('endpoint', '')
                                step_info = {"name": "API Call", "status": "loading", "event_type": "api_call"}
                                final_message = f"🌐 Calling service endpoint: {endpoint}"
                                additional_details = {"endpoint": endpoint}
                            
                            elif event_type == "service_response":
                                agent_name = event_data.get('data', {}).get('agent_name', '')
                                status_code = event_data.get('data', {}).get('status_code', 0)
                                step_info = {"name": "Service Response", "status": "complete", "event_type": "service_response"}
                                final_message = f"📨 Got response from {agent_name}: {status_code}"
                                additional_details = {
                                    "agent_name": agent_name,
                                    "status_code": status_code
                                }
                            
                            elif event_type == "query_execution":
                                database_type = event_data.get('data', {}).get('database_type', '')
                                query_preview = event_data.get('data', {}).get('query_preview', '')
                                step_info = {"name": "Query Execution", "status": "loading", "event_type": "query_execution"}
                                final_message = f"⚡ Executing ({database_type}) query"
                                additional_details = {
                                    "database_type": database_type,
                                    "query_preview": query_preview
                                }
                            
                            elif event_type == "query_results":
                                database_type = event_data.get('data', {}).get('database_type', '')
                                row_count = event_data.get('data', {}).get('row_count', 0)
                                execution_time = event_data.get('data', {}).get('execution_time', 0)
                                step_info = {"name": "Query Results", "status": "complete", "event_type": "query_results"}
                                final_message = f"📊 Retrieved {row_count} rows from {database_type}"
                                additional_details = {
                                    "database_type": database_type,
                                    "row_count": row_count,
                                    "execution_time": execution_time
                                }
                                
                            elif event_type in step_mapping:
                                step_info = step_mapping[event_type]
                                final_message = event_data.get('message', '') or step_info.get("message", "Processing...")
                            
                            # Handle workflow completion
                            if event_type == "workflow_completed":
                                final_answer = event_data.get('data', {}).get('final_answer', '')
                                execution_time = event_data.get('data', {}).get('execution_time_seconds', 0)
                                yield f"data: {json.dumps({
                                    'type': 'workflow_completed',
                                    'final_answer': final_answer,
                                    'execution_time': execution_time,
                                    'message': final_message or '✅ Analysis completed successfully',
                                    'timestamp': time.time()
                                }, default=str)}\n\n"
                                workflow_completed = True
                            
                            if step_info:
                                current_step = {
                                    "name": step_info["name"],
                                    "status": "complete" if event_type == "step_completed" else step_info.get("status", "loading"),
                                    "message": final_message,
                                    "event_type": event_type,
                                    "step_id": event_data.get("step_id", ""),
                                    "details": additional_details  # Include additional details
                                }
                                
                                # Create enhanced step update with detailed information
                                enhanced_update = {
                                    'type': 'enhanced_step_update',
                                    'current_step': current_step,
                                    'progress': {
                                        'current_step_number': events_processed,
                                        'estimated_total': 'Dynamic',
                                        'workflow_type': 'simple_orchestration'
                                    },
                                    'timestamp': time.time()
                                }
                                
                                # Add specific information for different event types
                                if event_type == "sql_generated":
                                    enhanced_update['sql_info'] = additional_details
                                elif event_type == "data_query":
                                    enhanced_update['query_info'] = additional_details
                                elif event_type == "llm_routing_decision":
                                    enhanced_update['routing_info'] = additional_details
                                elif event_type == "debug_info":
                                    enhanced_update['debug_info'] = additional_details
                                elif event_type == "agent_started":
                                    enhanced_update['agent_info'] = additional_details
                                elif event_type == "payload_generation":
                                    enhanced_update['payload_info'] = additional_details
                                elif event_type == "api_call":
                                    enhanced_update['api_info'] = additional_details
                                elif event_type == "service_response":
                                    enhanced_update['response_info'] = additional_details
                                elif event_type == "query_execution":
                                    enhanced_update['execution_info'] = additional_details
                                elif event_type == "query_results":
                                    enhanced_update['results_info'] = additional_details
                                
                                yield f"data: {json.dumps(enhanced_update, default=str)}\n\n"
                            
                        except json.JSONDecodeError:
                            continue
                    
                    # Check if we should exit the streaming loop
                    if workflow_completed and workflow_task and workflow_task.done():
                        break
                
                # Get final workflow result
                if workflow_task.done() and not workflow_task.exception():
                    workflow_result = await workflow_task
                    safe_result = convert_decimals_to_float(workflow_result)
                    
                    response_data = {
                        'type': 'enhanced_completed',
                        'final_answer': safe_result.get('greeting') or safe_result.get('final_answer'),
                        'greeting': safe_result.get('greeting'),
                        'workflow_type': 'simple_orchestration',
                        'timestamp': time.time()
                    }
                    
                    # Include results if available
                    if 'results' in safe_result:
                        response_data.update({
                            'execution_summary': safe_result.get('execution_summary'),
                            'agents_used': safe_result.get('agents_used', []),
                            'total_data_points': safe_result.get('total_data_points', 0),
                            'results': safe_result.get('results', []),
                            'visualization_ready': safe_result.get('visualization_ready', False),
                            'data_summary': safe_result.get('data_summary', {})
                        })
                    
                    yield f"data: {json.dumps(response_data, default=str)}\n\n"
                
                yield f"data: {json.dumps({'type': 'stream_end', 'timestamp': time.time()}, default=str)}\n\n"
                    
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'timestamp': time.time()}, default=str)}\n\n"
            finally:
                self.workflow_streamer.remove_client(client_id)
        
        return StreamingResponse(
            generate_enhanced_streaming(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )


# Global service instance  
streaming_service = StreamingService()
