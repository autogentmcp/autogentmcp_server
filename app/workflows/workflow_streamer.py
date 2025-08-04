"""
Real-time streaming support for workflow execution.
Provides Server-Sent Events (SSE) for live progress updates.
"""
import json
import asyncio
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import threading
import queue

class EventType(Enum):
    """Types of events that can be streamed."""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    USER_INPUT_REQUIRED = "user_input_required"
    PROGRESS_UPDATE = "progress_update"
    DEBUG_INFO = "debug_info"
    ERROR = "error"
    LLM_THINKING = "llm_thinking"
    LLM_ROUTING_DECISION = "llm_routing_decision"
    DATA_QUERY = "data_query"
    AGGREGATION = "aggregation"
    ROUTE_SELECTED = "route_selected"
    SQL_GENERATED = "sql_generated"
    
    # Enhanced event types for detailed streaming
    AGENT_STARTED = "agent_started"
    PAYLOAD_GENERATION = "payload_generation"
    API_CALL = "api_call"
    SERVICE_RESPONSE = "service_response"
    QUERY_EXECUTION = "query_execution"
    QUERY_RESULTS = "query_results"

@dataclass
class StreamEvent:
    """Event that can be streamed to clients."""
    event_type: Any  # Can be EventType enum or string
    workflow_id: str
    timestamp: datetime
    data: Dict[str, Any]
    step_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_sse_format(self) -> str:
        """Convert to Server-Sent Events format with proper JSON serialization."""
        import json
        from datetime import datetime
        
        def serialize_data(obj):
            """Custom serialization for complex objects."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'isoformat') and callable(getattr(obj, 'isoformat')):
                # Handle datetime-like objects
                return obj.isoformat()
            elif hasattr(obj, '__dict__') and not isinstance(obj, (int, float, str, bool, list, dict, type(None))):
                # Handle custom objects with __dict__
                try:
                    return {k: serialize_data(v) for k, v in obj.__dict__.items()}
                except:
                    return str(obj)
            elif isinstance(obj, dict):
                return {k: serialize_data(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize_data(item) for item in obj]
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                # Handle other iterables (but not strings)
                try:
                    return [serialize_data(item) for item in obj]
                except:
                    return str(obj)
            else:
                return obj
        
        # Handle both EventType enum and string event types
        event_type_str = self.event_type.value if hasattr(self.event_type, 'value') else str(self.event_type)
        
        event_data = {
            "type": event_type_str,
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp.isoformat(),
            "data": serialize_data(self.data),
            "step_id": self.step_id,
            "session_id": self.session_id
        }
        
        try:
            return f"data: {json.dumps(event_data, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            # Fallback for problematic data
            print(f"[WorkflowStreamer] JSON serialization error: {e}")
            safe_event_data = {
                "type": event_type_str,
                "workflow_id": self.workflow_id,
                "timestamp": self.timestamp.isoformat(),
                "data": {"message": str(self.data), "serialization_error": str(e)},
                "step_id": self.step_id,
                "session_id": self.session_id
            }
            return f"data: {json.dumps(safe_event_data, ensure_ascii=False, default=str)}\n\n"

class WorkflowStreamer:
    """
    Manages real-time streaming of workflow events.
    
    Features:
    - Server-Sent Events (SSE) support
    - Multiple client connections
    - Event filtering by workflow or session
    - Buffer management for reliability
    - Heartbeat to keep connections alive
    """
    
    def __init__(self):
        self.clients: Dict[str, queue.Queue] = {}  # client_id -> event queue
        self.workflow_clients: Dict[str, List[str]] = {}  # workflow_id -> client_ids
        self.session_clients: Dict[str, List[str]] = {}  # session_id -> client_ids
        self.event_buffer: Dict[str, List[StreamEvent]] = {}  # workflow_id -> events
        self.max_buffer_size = 1000
        self.heartbeat_interval = 30  # seconds
        self._lock = threading.Lock()
        
        # Start heartbeat thread
        self._start_heartbeat()
    
    def add_client(self, client_id: str, workflow_id: Optional[str] = None, session_id: Optional[str] = None) -> queue.Queue:
        """
        Add a new streaming client.
        
        Args:
            client_id: Unique client identifier
            workflow_id: Optional workflow to filter events for
            session_id: Optional session to filter events for
            
        Returns:
            Queue that will receive events for this client
        """
        with self._lock:
            # Create event queue for client
            event_queue = queue.Queue(maxsize=100)
            self.clients[client_id] = event_queue
            
            # Register client for specific workflow
            if workflow_id:
                if workflow_id not in self.workflow_clients:
                    self.workflow_clients[workflow_id] = []
                self.workflow_clients[workflow_id].append(client_id)
                
                # Send buffered events for this workflow
                if workflow_id in self.event_buffer:
                    for event in self.event_buffer[workflow_id][-50:]:  # Last 50 events
                        try:
                            event_queue.put_nowait(event)
                        except queue.Full:
                            pass
            
            # Register client for specific session
            if session_id:
                if session_id not in self.session_clients:
                    self.session_clients[session_id] = []
                self.session_clients[session_id].append(client_id)
            
            print(f"[WorkflowStreamer] Added client {client_id} (workflow: {workflow_id}, session: {session_id})")
            return event_queue
    
    def remove_client(self, client_id: str):
        """Remove a streaming client."""
        with self._lock:
            if client_id in self.clients:
                del self.clients[client_id]
                
                # Remove from workflow subscriptions
                for workflow_id, client_list in self.workflow_clients.items():
                    if client_id in client_list:
                        client_list.remove(client_id)
                
                # Remove from session subscriptions
                for session_id, client_list in self.session_clients.items():
                    if client_id in client_list:
                        client_list.remove(client_id)
                
                print(f"[WorkflowStreamer] Removed client {client_id}")
    
    def emit_event(self, event: StreamEvent):
        """
        Emit an event to all relevant clients.
        
        Args:
            event: Event to emit
        """
        # Add debug logging for detailed events
        event_type_str = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
        if event_type_str in ['agent_started', 'payload_generation', 'api_call', 'service_response', 'query_execution', 'query_results', 'step_started', 'step_completed']:
            print(f"[WorkflowStreamer] Emitting detailed event: {event_type_str}, workflow_id: {event.workflow_id}, session_id: {event.session_id}")
            print(f"[WorkflowStreamer] Event data keys: {list(event.data.keys()) if event.data else 'no_data'}")
        
        with self._lock:
            # Buffer the event
            if event.workflow_id not in self.event_buffer:
                self.event_buffer[event.workflow_id] = []
            
            self.event_buffer[event.workflow_id].append(event)
            
            # Trim buffer if too large
            if len(self.event_buffer[event.workflow_id]) > self.max_buffer_size:
                self.event_buffer[event.workflow_id] = self.event_buffer[event.workflow_id][-self.max_buffer_size//2:]
            
            # Send to workflow-specific clients
            workflow_clients = self.workflow_clients.get(event.workflow_id, [])
            for client_id in workflow_clients:
                if client_id in self.clients:
                    try:
                        self.clients[client_id].put_nowait(event)
                    except queue.Full:
                        print(f"[WorkflowStreamer] Client {client_id} queue full, skipping event")
            
            # Send to session-specific clients
            if event.session_id:
                session_clients = self.session_clients.get(event.session_id, [])
                print(f"[WorkflowStreamer] Event session_id: {event.session_id}, found {len(session_clients)} session clients")
                for client_id in session_clients:
                    if client_id in self.clients and client_id not in workflow_clients:  # Avoid duplicates
                        try:
                            self.clients[client_id].put_nowait(event)
                            print(f"[WorkflowStreamer] Sent {event.event_type} event to session client {client_id}")
                        except queue.Full:
                            print(f"[WorkflowStreamer] Client {client_id} queue full, skipping event")
            
            # Send to all clients if it's a general event
            if not workflow_clients and not event.session_id:
                for client_id, client_queue in self.clients.items():
                    try:
                        client_queue.put_nowait(event)
                    except queue.Full:
                        print(f"[WorkflowStreamer] Client {client_id} queue full, skipping event")
    
    def emit_workflow_started(self, workflow_id: str, session_id: str, title: str, description: str, steps: int):
        """Emit workflow started event."""
        event = StreamEvent(
            event_type=EventType.WORKFLOW_STARTED,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            data={
                "title": title,
                "description": description,
                "total_steps": steps,
                "message": f"Started workflow: {title}"
            }
        )
        self.emit_event(event)
    
    def emit_workflow_completed(self, workflow_id: str, session_id: str, final_answer: str, execution_time: float):
        """Emit workflow completed event."""
        event = StreamEvent(
            event_type=EventType.WORKFLOW_COMPLETED,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            data={
                "final_answer": final_answer,
                "execution_time_seconds": execution_time,
                "message": f"Workflow completed in {execution_time:.1f}s"
            }
        )
        self.emit_event(event)
    
    def emit_step_started(self, workflow_id: str, session_id: str, step_id: str, step_type: str, description: str):
        """Emit step started event."""
        event = StreamEvent(
            event_type=EventType.STEP_STARTED,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id=step_id,
            data={
                "step_type": step_type,
                "description": description,
                "message": f"Starting step: {description}"
            }
        )
        print(f"[WorkflowStreamer] Emitting step started event: {event}")
        self.emit_event(event)
    
    def emit_step_completed(self, workflow_id: str, session_id: str, step_id: str, step_type: str, execution_time: float):
        """Emit step completed event."""
        event = StreamEvent(
            event_type=EventType.STEP_COMPLETED,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id=step_id,
            data={
                "step_type": step_type,
                "execution_time_seconds": execution_time,
                "message": f"Completed {step_type} step in {execution_time:.1f}s"
            }
        )
        self.emit_event(event)
    
    def emit_user_input_required(self, workflow_id: str, session_id: str, step_id: str, input_request: Dict[str, Any]):
        """Emit user input required event."""
        event = StreamEvent(
            event_type=EventType.USER_INPUT_REQUIRED,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id=step_id,
            data={
                "input_request": input_request,
                "message": f"User input required: {input_request.get('prompt', 'Input needed')}"
            }
        )
        self.emit_event(event)
    
    def emit_progress_update(self, workflow_id: str, session_id: str, progress_percent: float, current_step: str):
        """Emit progress update event."""
        event = StreamEvent(
            event_type=EventType.PROGRESS_UPDATE,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            data={
                "progress_percent": progress_percent,
                "current_step": current_step,
                "message": f"Progress: {progress_percent:.1f}% - {current_step}"
            }
        )
        self.emit_event(event)
    
    def emit_llm_thinking(self, workflow_id: str, session_id: str, step_id: str, thinking_about: str):
        """Emit LLM thinking/processing event."""
        event = StreamEvent(
            event_type=EventType.LLM_THINKING,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id=step_id,
            data={
                "thinking_about": thinking_about,
                "message": f"LLM processing: {thinking_about}"
            }
        )
        self.emit_event(event)
    
    def emit_data_query(self, workflow_id: str, session_id: str, step_id: str, query: str, database_type: str):
        """Emit data query event."""
        event = StreamEvent(
            event_type=EventType.DATA_QUERY,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id=step_id,
            data={
                "query": query,
                "database_type": database_type,
                "message": f"Executing {database_type} query: {query[:100]}..."
            }
        )
        self.emit_event(event)
    
    def emit_debug_info(self, workflow_id: str, session_id: str, step_id: str, debug_message: str, debug_data: Any = None):
        """Emit debug information event."""
        event = StreamEvent(
            event_type=EventType.DEBUG_INFO,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id=step_id,
            data={
                "debug_message": debug_message,
                "debug_data": debug_data,
                "message": f"Debug: {debug_message}"
            }
        )
        self.emit_event(event)

    def emit_routing_decision(self, workflow_id: str, session_id: str, route_type: str, selected_agent: str, confidence: float, reasoning: str, candidates: List[Dict[str, Any]] = None):
        """Emit LLM routing decision event."""
        event = StreamEvent(
            event_type=EventType.LLM_ROUTING_DECISION,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id="routing",
            data={
                "route_type": route_type,
                "selected_agent": selected_agent,
                "confidence": confidence,
                "reasoning": reasoning,
                "candidates": candidates or [],
                "message": f"🎯 Selected {route_type}: {selected_agent} (confidence: {confidence:.1f}%)"
            }
        )
        print(f"[WorkflowStreamer] Emitting routing decision: {route_type} - {selected_agent} ({confidence}%)")
        self.emit_event(event)

    def emit_sql_generated(self, workflow_id: str, session_id: str, sql_query: str, database_type: str, llm_reasoning: str = None):
        """Emit SQL generation event."""
        event = StreamEvent(
            event_type=EventType.SQL_GENERATED,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id="sql_generation",
            data={
                "sql_query": sql_query,
                "database_type": database_type,
                "llm_reasoning": llm_reasoning,
                "message": f"📝 Generated SQL: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}"
            }
        )
        print(f"[WorkflowStreamer] Emitting SQL generation: {database_type} - {sql_query[:50]}...")
        self.emit_event(event)
    
    def emit_error(self, workflow_id: str, session_id: str, step_id: str, error_message: str, error_details: Any = None):
        """Emit error event."""
        event = StreamEvent(
            event_type=EventType.ERROR,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id=step_id,
            data={
                "error_message": error_message,
                "error_details": error_details,
                "message": f"Error: {error_message}"
            }
        )
        self.emit_event(event)
    
    def emit_agent_started(self, workflow_id: str, session_id: str, agent_name: str, agent_type: str):
        """Emit agent started event."""
        event = StreamEvent(
            event_type=EventType.AGENT_STARTED,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id="agent_execution",
            data={
                "agent_name": agent_name,
                "agent_type": agent_type,
                "message": f"🚀 Starting agent: {agent_name} ({agent_type})"
            }
        )
        print(f"[WorkflowStreamer] Emitting agent started: {agent_name} ({agent_type})")
        self.emit_event(event)
    
    def emit_payload_generation(self, workflow_id: str, session_id: str, agent_name: str, payload_preview: str = None):
        """Emit payload generation event."""
        event = StreamEvent(
            event_type=EventType.PAYLOAD_GENERATION,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id="payload_generation",
            data={
                "agent_name": agent_name,
                "payload_preview": payload_preview,
                "message": f"📦 Generating payload for {agent_name}"
            }
        )
        print(f"[WorkflowStreamer] Emitting payload generation for: {agent_name}")
        self.emit_event(event)
    
    def emit_api_call(self, workflow_id: str, session_id: str, agent_name: str, endpoint: str, method: str = "POST"):
        """Emit API call event."""
        event = StreamEvent(
            event_type=EventType.API_CALL,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id="api_call",
            data={
                "agent_name": agent_name,
                "endpoint": endpoint,
                "method": method,
                "message": f"🌐 Calling service endpoint: {endpoint}"
            }
        )
        print(f"[WorkflowStreamer] Emitting API call: {method} {endpoint}")
        self.emit_event(event)
    
    def emit_service_response(self, workflow_id: str, session_id: str, agent_name: str, status_code: int, response_size: int = None):
        """Emit service response event."""
        event = StreamEvent(
            event_type=EventType.SERVICE_RESPONSE,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id="service_response",
            data={
                "agent_name": agent_name,
                "status_code": status_code,
                "response_size": response_size,
                "message": f"📨 Got response from {agent_name}: {status_code}" + (f" ({response_size} bytes)" if response_size else "")
            }
        )
        print(f"[WorkflowStreamer] Emitting service response: {status_code}")
        self.emit_event(event)
    
    def emit_query_execution(self, workflow_id: str, session_id: str, database_type: str, query_preview: str):
        """Emit query execution event."""
        event = StreamEvent(
            event_type=EventType.QUERY_EXECUTION,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id="query_execution",
            data={
                "database_type": database_type,
                "query_preview": query_preview,
                "message": f"⚡ Executing ({database_type}) query: {query_preview[:100]}..."
            }
        )
        print(f"[WorkflowStreamer] Emitting query execution: {database_type}")
        self.emit_event(event)
    
    def emit_query_results(self, workflow_id: str, session_id: str, database_type: str, row_count: int, execution_time: float = None):
        """Emit query results event."""
        event = StreamEvent(
            event_type=EventType.QUERY_RESULTS,
            workflow_id=workflow_id,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id="query_results",
            data={
                "database_type": database_type,
                "row_count": row_count,
                "execution_time": execution_time,
                "message": f"📊 Retrieved {row_count} rows from {database_type}" + (f" in {execution_time:.2f}s" if execution_time else "")
            }
        )
        print(f"[WorkflowStreamer] Emitting query results: {row_count} rows")
        self.emit_event(event)
    
    async def stream_events(self, client_id: str) -> AsyncGenerator[str, None]:
        """
        Async generator for streaming events to a client with improved timeout handling.
        
        Args:
            client_id: Client identifier
            
        Yields:
            SSE-formatted event strings
        """
        import json
        from app.config import config  # Import here to avoid circular imports
        
        if client_id not in self.clients:
            error_data = {
                "type": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Client not found",
                "message": "Client not found in active connections"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            return
        
        client_queue = self.clients[client_id]
        last_heartbeat = time.time()
        stream_start_time = time.time()
        
        print(f"[WorkflowStreamer] Starting event stream for client {client_id} with {config.STREAM_TIMEOUT_SECONDS}s timeout")
        
        try:
            while client_id in self.clients:
                try:
                    # Use config-based timeout instead of hardcoded 15 seconds
                    event = client_queue.get(timeout=config.STREAM_TIMEOUT_SECONDS)
                    
                    # Handle both StreamEvent objects and simple dict events
                    if hasattr(event, 'to_sse_format'):
                        # It's a StreamEvent object
                        yield event.to_sse_format()
                    else:
                        # It's a simple dict event from emit_custom_event
                        yield f"data: {json.dumps(event, default=str)}\n\n"
                    
                    client_queue.task_done()
                    
                    # Reset heartbeat timer when we get real events
                    last_heartbeat = time.time()
                    
                except queue.Empty:
                    # Check if we've exceeded total max wait time
                    current_time = time.time()
                    elapsed_since_start = current_time - stream_start_time
                    elapsed_since_heartbeat = current_time - last_heartbeat
                    
                    if elapsed_since_start < config.STREAM_MAX_WAIT_SECONDS:
                        # Send heartbeat to keep connection alive during long operations
                        heartbeat_data = {
                            "type": "heartbeat",
                            "timestamp": datetime.utcnow().isoformat(),
                            "message": f"🔄 Still processing... ({elapsed_since_start:.0f}s elapsed)",
                            "elapsed_seconds": elapsed_since_start
                        }
                        yield f"data: {json.dumps(heartbeat_data)}\n\n"
                        last_heartbeat = current_time
                        print(f"[WorkflowStreamer] Sent heartbeat to client {client_id} after {elapsed_since_start:.0f}s")
                    else:
                        # Exceeded maximum wait time - end the stream
                        print(f"[WorkflowStreamer] Client {client_id} exceeded maximum wait time ({config.STREAM_MAX_WAIT_SECONDS}s), ending stream")
                        timeout_data = {
                            "type": "timeout", 
                            "timestamp": datetime.utcnow().isoformat(),
                            "message": "⏰ Stream timeout - process is taking longer than expected",
                            "elapsed_seconds": elapsed_since_start
                        }
                        yield f"data: {json.dumps(timeout_data)}\n\n"
                        break
                        
                except Exception as e:
                    # Send error with proper format
                    print(f"[WorkflowStreamer] Error streaming events for client {client_id}: {str(e)}")
                    error_data = {
                        "type": "error",
                        "timestamp": datetime.utcnow().isoformat(),
                        "error": str(e),
                        "message": f"Stream error: {str(e)}"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    break
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            print(f"[WorkflowStreamer] Client {client_id} connection cancelled")
        finally:
            print(f"[WorkflowStreamer] Event stream ended for client {client_id}")
            self.remove_client(client_id)
    
    def _start_heartbeat(self):
        """Start heartbeat thread to keep connections alive."""
        def heartbeat_worker():
            while True:
                try:
                    time.sleep(self.heartbeat_interval)
                    
                    # Send heartbeat to all clients
                    with self._lock:
                        for client_id, client_queue in self.clients.items():
                            try:
                                heartbeat_event = StreamEvent(
                                    event_type=EventType.DEBUG_INFO,
                                    workflow_id="system",
                                    timestamp=datetime.utcnow(),
                                    data={
                                        "heartbeat": True,
                                        "active_clients": len(self.clients),
                                        "message": "Heartbeat"
                                    }
                                )
                                client_queue.put_nowait(heartbeat_event)
                            except queue.Full:
                                pass  # Skip heartbeat if queue is full
                except Exception as e:
                    print(f"[WorkflowStreamer] Heartbeat error: {e}")
        
        heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        heartbeat_thread.start()
    
    def get_workflow_events(self, workflow_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events for a workflow."""
        with self._lock:
            if workflow_id in self.event_buffer:
                events = self.event_buffer[workflow_id][-limit:]
                return [
                    {
                        "type": event.event_type.value,
                        "timestamp": event.timestamp.isoformat(),
                        "data": event.data,
                        "step_id": event.step_id
                    }
                    for event in events
                ]
            return []
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get streaming client statistics."""
        with self._lock:
            return {
                "total_clients": len(self.clients),
                "workflow_subscriptions": {wf_id: len(clients) for wf_id, clients in self.workflow_clients.items()},
                "session_subscriptions": {sess_id: len(clients) for sess_id, clients in self.session_clients.items()},
                "buffered_workflows": len(self.event_buffer),
                "total_buffered_events": sum(len(events) for events in self.event_buffer.values())
            }
    
    def emit_custom_event(self, session_id: str, event_type: str, message: str):
        """
        Emit a custom event to session clients.
        
        Args:
            session_id: Session identifier
            event_type: Type of event (string)
            message: Event message
        """
        from datetime import datetime
        
        # Create a StreamEvent with event_type as string (not enum)
        event = StreamEvent(
            event_type=event_type,  # Pass string directly - we'll handle this in to_sse_format
            workflow_id="custom",
            timestamp=datetime.utcnow(),
            session_id=session_id,
            step_id="progress", 
            data={"message": message}
        )
        
        # Send to session-based clients
        with self._lock:
            session_clients = self.session_clients.get(session_id, [])
            for client_id in session_clients:
                if client_id in self.clients:
                    try:
                        self.clients[client_id].put_nowait(event)
                        print(f"[WorkflowStreamer] Sent custom event {event_type} to client {client_id}")
                    except queue.Full:
                        print(f"[WorkflowStreamer] Client {client_id} queue full, skipping custom event")
                    except Exception as e:
                        print(f"[WorkflowStreamer] Failed to send custom event to client {client_id}: {e}")
                        self.remove_client(client_id)

# Global instance
workflow_streamer = WorkflowStreamer()
