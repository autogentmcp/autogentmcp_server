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
                self.workflow_streamer.add_client(client_id, session_id=request.session_id)
                
                # Send initial connection status
                yield f"data: {json.dumps({
                    'type': 'connection_established',
                    'message': '🚀 Simple orchestrator workflow initiated',
                    'timestamp': time.time()
                }, default=str)}\n\n"
                
                # Step mapping for UI display
                step_mapping = {
                    "workflow_started": {"name": "Workflow Started", "status": "loading", "message": "🚀 Simple Orchestrator workflow started", "next": "Understanding Request"},
                    "workflow_completed": {"name": "Analysis Complete", "status": "complete", "message": "✅ Analysis completed successfully", "next": "Finished"},
                    "error": {"name": "Error", "status": "error", "message": "❌ Error occurred", "next": "Review"},
                    "step_started": {"name": "Processing", "status": "loading", "message": "▶️ Starting step", "next": "In Progress"},
                    "step_completed": {"name": "Step Complete", "status": "complete", "message": "✅ Step completed", "next": "Next Step"},
                    "analyze": {"name": "Understanding Request", "status": "loading", "message": "🧠 Understanding your request", "next": "Planning Execution"},
                    "execute": {"name": "Executing Plan", "status": "loading", "message": "⚡ Executing agent plan", "next": "Generating Response"},
                    "respond": {"name": "Generating Response", "status": "loading", "message": "📝 Generating final response", "next": "Complete"},
                }
                
                # Start simple workflow
                workflow_task = asyncio.create_task(
                    self.orchestrator.execute_workflow(
                        user_query=request.query,
                        session_id=request.session_id
                    )
                )
                
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
                            
                            # Map workflow events to UI steps
                            step_info = None
                            final_message = event_data.get('message', '')
                            
                            if event_type == "step_started" or event_type == "step_completed":
                                step_id = event_data.get("step_id", "unknown")
                                if step_id in step_mapping:
                                    step_info = step_mapping[step_id]
                                    final_message = event_data.get('description', '') or step_info.get("message", "Processing...")
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
                                }
                                
                                yield f"data: {json.dumps({
                                    'type': 'enhanced_step_update',
                                    'current_step': current_step,
                                    'progress': {
                                        'current_step_number': events_processed,
                                        'estimated_total': 'Dynamic',
                                        'workflow_type': 'simple_orchestration'
                                    },
                                    'timestamp': time.time()
                                }, default=str)}\n\n"
                            
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
