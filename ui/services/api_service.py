"""
Backend API service for MCP Chat Interface
Handles all communication with the MCP backend
"""

import json
import time
import uuid
import requests
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime

# Try to import sseclient for streaming support
try:
    import sseclient
    has_sse_support = True
except ImportError:
    has_sse_support = False

from ..config import BACKEND_CONFIG, STREAMING_CONFIG


class MCPBackendService:
    """Service class for backend API communication"""
    
    def __init__(self):
        self.base_url = BACKEND_CONFIG["url"]
        self.timeout = BACKEND_CONFIG["timeout"]
        self.stream_timeout = STREAMING_CONFIG.get("event_timeout", 120)
    
    def send_query_with_streaming(
        self, 
        query: str, 
        session_id: str,
        conversation_id: str = None,
        conversation_history: List[Dict] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Send a query with streaming progress updates.
        
        Args:
            query: The user query
            session_id: Current session ID
            conversation_id: Current conversation ID
            conversation_history: Recent conversation history
            progress_callback: Function to call with progress updates
            
        Returns:
            Response dictionary with results
        """
        print(f"[MCPBackendService] Starting streaming query: {query[:50]}...")
        
        # Check if sseclient is available
        if not has_sse_support:
            print("[MCPBackendService] sseclient not available, falling back to regular query")
            return self.send_query_fallback(query, session_id, conversation_id, conversation_history)
        
        try:
            request_id = str(uuid.uuid4())
            
            payload = {
                "query": query,
                "session_id": session_id,
                "conversation_id": conversation_id,
                "conversation_history": conversation_history or [],
                "include_analysis": True,
                "max_steps": 5,
                "request_id": request_id
            }
            
            print(f"[MCPBackendService] Payload prepared with {len(conversation_history or [])} history messages")
            
            # Use the streaming endpoint
            response = requests.post(
                f"{self.base_url}/orchestration/enhanced/stream",
                json=payload,
                timeout=self.stream_timeout,
                stream=True,
                headers={'Accept': 'text/event-stream'}
            )
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}",
                    "message": f"Server returned: {response.text}"
                }
            
            # Parse SSE stream
            return self._process_sse_stream(response, progress_callback)
            
        except Exception as e:
            print(f"[MCPBackendService] Streaming failed: {e}")
            return self.send_query_fallback(query, session_id, conversation_id, conversation_history)
    
    def _process_sse_stream(self, response, progress_callback=None) -> Dict[str, Any]:
        """Process Server-Sent Events stream"""
        client = sseclient.SSEClient(response)
        final_result = None
        workflow_events = []
        events_received = 0
        start_time = time.time()
        
        print(f"[MCPBackendService] Starting SSE stream processing...")
        
        for event in client.events():
            try:
                # Check for timeout
                if time.time() - start_time > self.stream_timeout:
                    print(f"[MCPBackendService] SSE stream timeout after {self.stream_timeout}s")
                    break
                
                events_received += 1
                
                # Skip empty events
                if not event.data or not event.data.strip():
                    continue
                
                # Parse JSON data
                try:
                    event_data = json.loads(event.data)
                    event_type = event_data.get("type", "")
                    
                    # Debug: Log all events to see what we're getting
                    print(f"[MCPBackendService] Raw event: type='{event_type}', data keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'not_dict'}")
                    
                    # Skip debug events
                    if event_type == "enhanced_event" and event_data.get("event_type") == "debug_info":
                        continue
                    
                    # Store event
                    captured_event = {
                        "type": event_type,
                        "data": event_data,
                        "timestamp": datetime.now().isoformat()
                    }
                    workflow_events.append(captured_event)
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(event_type, event_data)
                    
                    # Check for completion
                    if event_type == "enhanced_completed":
                        print(f"[MCPBackendService] Received enhanced_completed event")
                        final_result = event_data
                        break
                    elif event_type in ["workflow_completed", "completed"]:
                        print(f"[MCPBackendService] Received {event_type} event")
                        if not final_result:
                            final_result = event_data
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"[MCPBackendService] Error processing SSE event: {e}")
                    continue
            
            except Exception as e:
                print(f"[MCPBackendService] Error in event loop: {e}")
                continue
        
        print(f"[MCPBackendService] SSE stream ended. Events processed: {events_received}")
        
        if final_result:
            final_result["workflow_events"] = workflow_events
            return {
                "status": "success",
                "enhanced_result": final_result
            }
        elif workflow_events:
            print(f"[MCPBackendService] No final result but got {len(workflow_events)} events")
            # Try to extract result from workflow events
            for event in reversed(workflow_events):
                if event.get('type') == 'workflow_completed':
                    final_result = event.get('data', {})
                    final_result["workflow_events"] = workflow_events
                    return {
                        "status": "success",
                        "enhanced_result": final_result
                    }
        
        print(f"[MCPBackendService] No usable result from stream")
        return {
            "status": "error",
            "message": "No result received from streaming endpoint"
        }
    
    def send_query_fallback(
        self, 
        query: str, 
        session_id: str,
        conversation_id: str = None,
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Fallback method for sending queries without streaming.
        
        Args:
            query: The user query
            session_id: Current session ID
            conversation_id: Current conversation ID
            conversation_history: Recent conversation history
            
        Returns:
            Response dictionary with results
        """
        try:
            payload = {
                "query": query,
                "session_id": session_id,
                "conversation_id": conversation_id,
                "conversation_history": conversation_history or [],
                "include_analysis": True,
                "max_steps": 5
            }
            
            print(f"[MCPBackendService] Sending fallback query with {len(conversation_history or [])} history messages")
            
            response = requests.post(
                f"{self.base_url}/enhanced/query",
                json=payload,
                timeout=self.timeout
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
    
    def send_query(
        self, 
        query: str, 
        session_id: str,
        conversation_id: str = None,
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Main query method - uses streaming by default.
        
        Args:
            query: The user query
            session_id: Current session ID
            conversation_id: Current conversation ID
            conversation_history: Recent conversation history
            
        Returns:
            Response dictionary with results
        """
        if STREAMING_CONFIG.get("enabled", True):
            return self.send_query_with_streaming(query, session_id, conversation_id, conversation_history)
        else:
            return self.send_query_fallback(query, session_id, conversation_id, conversation_history)
