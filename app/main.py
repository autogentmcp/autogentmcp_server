from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.responses import StreamingResponse
import uuid
import json
import asyncio
import time

from dotenv import load_dotenv

from app.langgraph_router import route_query
from app.auth_handler import auth_handler
from app.session_manager import session_manager
from app.registry import sync_registry
from app.vault_manager import vault_manager
from app.registry_auth_integration import registry_auth_integration
from app.data_agents_client import data_agents_client
from app.langgraph_orchestrator import convert_decimals_to_float
from app.orchestrator import simple_orchestrator
from app.workflow_streamer import workflow_streamer

load_dotenv()

app = FastAPI(title="MCP Registry Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class AuthCredential(BaseModel):
    key: str
    value: str

@app.get("/health")
def health():
    vault_health = vault_manager.health_check()
    return {
        "status": "ok",
        "vault": vault_health
    }

@app.post("/query")
def query_endpoint(request: QueryRequest):
    """Route a user query to the best agent/tool using LangGraph."""
    return route_query({"query": request.query}, request.session_id)

@app.post("/sync_registry")
def sync_registry_endpoint():
    """Manually sync the registry."""
    try:
        sync_registry()
        return {"status": "success", "message": "Registry synced successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/auth/set_credential")
def set_auth_credential(credential: AuthCredential):
    """Set an authentication credential."""
    auth_handler.set_auth_credential(credential.key, credential.value)
    return {"status": "success", "message": f"Credential {credential.key} updated"}

@app.get("/sessions")
def get_sessions():
    """Get all active sessions."""
    sessions = session_manager.get_all_sessions()
    return {"sessions": [session_manager.get_session_summary(sid) for sid in sessions]}

@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    """Clear a specific session."""
    session_manager.clear_session(session_id)
    return {"status": "success", "message": f"Session {session_id} cleared"}

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Get session details."""
    history = session_manager.get_session_history(session_id)
    return {
        "session_id": session_id,
        "history": history,
        "summary": session_manager.get_session_summary(session_id)
    }

@app.get("/vault/stats")
def get_vault_stats():
    """Get vault cache statistics."""
    return vault_manager.get_cache_stats()

@app.post("/vault/clear_cache")
def clear_vault_cache():
    """Clear vault cache."""
    vault_manager.clear_cache()
    return {"status": "success", "message": "Vault cache cleared"}

@app.post("/vault/preload_cache")
def preload_vault_cache():
    """Manually trigger vault cache preload from registry."""
    vault_manager.preload_cache_from_registry()
    cache_stats = vault_manager.get_cache_stats()
    return {
        "status": "success", 
        "message": "Vault cache preload completed",
        "cache_stats": cache_stats
    }

@app.post("/data-agents/preload_credentials")
def preload_data_agent_credentials():
    """Manually trigger data agent credential preloading into vault cache."""
    result = data_agents_client.preload_vault_credentials()
    vault_stats = vault_manager.get_cache_stats()
    
    return {
        "status": result.get("status", "unknown"),
        "message": result.get("message", "Data agent credential preload completed"),
        "preload_result": result,
        "vault_cache_stats": vault_stats
    }

class AuthHeaderRequest(BaseModel):
    application_id: str
    authentication_method: str
    endpoint_url: str = None
    request_method: str = "GET"
    request_body: str = None

class AuthHeaderVaultRequest(BaseModel):
    vault_key: str
    authentication_method: str
    endpoint_url: str = None
    request_method: str = "GET"
    request_body: str = None

@app.post("/auth/generate_headers")
def generate_auth_headers(request: AuthHeaderRequest):
    """Generate authentication headers for an application."""
    try:
        headers = auth_handler.generate_auth_headers(
            application_id=request.application_id,
            authentication_method=request.authentication_method,
            endpoint_url=request.endpoint_url,
            request_method=request.request_method,
            request_body=request.request_body
        )
        
        return {
            "status": "success",
            "application_id": request.application_id,
            "authentication_method": request.authentication_method,
            "generated_headers": headers,
            "header_count": len(headers)
        }
    except Exception as e:
        return {"status": "error", "message": f"Error generating headers: {str(e)}"}

@app.post("/auth/generate_headers_with_vault_key")
def generate_auth_headers_with_vault_key(request: AuthHeaderVaultRequest):
    """Generate authentication headers using vault key directly."""
    try:
        headers = auth_handler.generate_auth_headers_with_vault_key(
            vault_key=request.vault_key,
            authentication_method=request.authentication_method,
            endpoint_url=request.endpoint_url,
            request_method=request.request_method,
            request_body=request.request_body
        )
        
        return {
            "status": "success",
            "vault_key": request.vault_key,
            "authentication_method": request.authentication_method,
            "generated_headers": headers,
            "header_count": len(headers)
        }
    except Exception as e:
        return {"status": "error", "message": f"Error generating headers: {str(e)}"}

@app.get("/auth/validate/{application_id}/{authentication_method}")
def validate_auth_credentials(application_id: str, authentication_method: str):
    """Validate that required credentials are available for the authentication method."""
    try:
        is_valid = auth_handler.validate_auth_credentials(application_id, authentication_method)
        
        return {
            "status": "success",
            "application_id": application_id,
            "authentication_method": authentication_method,
            "is_valid": is_valid,
            "message": "Credentials are valid" if is_valid else "Credentials are missing or invalid"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error validating credentials: {str(e)}"}

@app.get("/auth/validate_with_vault_key/{vault_key}/{authentication_method}")
def validate_auth_credentials_with_vault_key(vault_key: str, authentication_method: str):
    """Validate that required credentials are available for the authentication method using vault key."""
    try:
        is_valid = auth_handler.validate_auth_credentials_with_vault_key(vault_key, authentication_method)
        
        return {
            "status": "success",
            "vault_key": vault_key,
            "authentication_method": authentication_method,
            "is_valid": is_valid,
            "message": "Credentials are valid" if is_valid else "Credentials are missing or invalid"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error validating credentials: {str(e)}"}

@app.get("/auth/supported_methods")
def get_supported_auth_methods():
    """Get list of supported authentication methods."""
    from app.auth_header_generator import auth_header_generator
    
    return {
        "status": "success",
        "supported_methods": list(auth_header_generator.SUPPORTED_AUTH_METHODS),
        "total_methods": len(auth_header_generator.SUPPORTED_AUTH_METHODS),
        "description": "All supported authentication methods with vault-based credential processing"
    }

@app.get("/auth/registry/agents")
def get_agents_with_auth_info():
    """Get all agents with their authentication information from registry."""
    try:
        auth_info = registry_auth_integration.list_agents_with_auth_info()
        
        return {
            "status": "success",
            "agents": auth_info,
            "total_agents": len(auth_info),
            "message": "Registry agents with authentication information"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting agents: {str(e)}"}

@app.get("/auth/registry/agent/{app_key}")
def get_agent_auth_info(app_key: str):
    """Get authentication information for a specific agent from registry."""
    try:
        auth_info = registry_auth_integration.get_agent_auth_info(app_key)
        
        if auth_info:
            return {
                "status": "success",
                "auth_info": auth_info,
                "message": "Agent authentication information retrieved"
            }
        else:
            return {
                "status": "error",
                "message": f"Agent {app_key} not found or has no authentication configuration"
            }
    except Exception as e:
        return {"status": "error", "message": f"Error getting agent info: {str(e)}"}

class RegistryAuthRequest(BaseModel):
    app_key: str
    endpoint_url: str = None
    request_method: str = "GET"
    request_body: str = None

@app.post("/auth/registry/generate_headers")
def generate_auth_headers_from_registry(request: RegistryAuthRequest):
    """Generate authentication headers for an agent using registry data."""
    try:
        headers = registry_auth_integration.get_auth_headers_for_agent(
            app_key=request.app_key,
            endpoint_url=request.endpoint_url,
            request_method=request.request_method,
            request_body=request.request_body
        )
        
        return {
            "status": "success",
            "app_key": request.app_key,
            "generated_headers": headers,
            "header_count": len(headers),
            "message": "Headers generated from registry configuration"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error generating headers: {str(e)}"}

@app.get("/auth/registry/validate/{app_key}")
def validate_agent_credentials_from_registry(app_key: str):
    """Validate agent credentials using registry configuration."""
    try:
        is_valid = registry_auth_integration.validate_agent_credentials(app_key)
        
        return {
            "status": "success",
            "app_key": app_key,
            "is_valid": is_valid,
            "message": "Agent credentials are valid" if is_valid else "Agent credentials are invalid or missing"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error validating credentials: {str(e)}"}

@app.get("/test/llm")
def test_llm_connection():
    """Test LLM connection and basic functionality."""
    from app.ollama_client import ollama_client
    return ollama_client.test_connection()

@app.get("/data-agents/list")
def get_data_agents():
    """Get all data agents."""
    try:
        data_agents = data_agents_client.fetch_data_agents()
        
        return {
            "status": "success",
            "data_agents": list(data_agents.values()),
            "total_agents": len(data_agents),
            "message": "Data agents retrieved successfully"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting data agents: {str(e)}"}

@app.get("/data-agents/search")
def search_data_agents(keywords: str):
    """Search data agents by keywords."""
    try:
        keyword_list = [kw.strip() for kw in keywords.split(",")]
        results = data_agents_client.search_data_agents_by_keywords(keyword_list)
        
        return {
            "status": "success",
            "search_keywords": keyword_list,
            "results": results,
            "total_matches": len(results),
            "message": "Data agents search completed"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error searching data agents: {str(e)}"}

@app.get("/data-agents/{agent_id}")
def get_data_agent(agent_id: str):
    """Get specific data agent information."""
    try:
        agent_info = data_agents_client.get_data_agent_info(agent_id)
        
        if agent_info:
            return {
                "status": "success",
                "agent": agent_info,
                "message": "Data agent information retrieved"
            }
        else:
            return {
                "status": "error",
                "message": f"Data agent {agent_id} not found"
            }
    except Exception as e:
        return {"status": "error", "message": f"Error getting data agent: {str(e)}"}

@app.get("/routing/candidates")
def get_routing_candidates(query: str):
    """Get routing candidates for a query (for debugging/testing)."""
    try:
        from app.unified_router import unified_router
        
        # Extract keywords
        keywords = unified_router._extract_keywords(query)
        
        # Get candidates from both sources
        app_candidates = unified_router._get_application_candidates(query, keywords)
        data_agent_candidates = unified_router._get_data_agent_candidates(keywords)
        
        # Combine and sort
        all_candidates = app_candidates + data_agent_candidates
        all_candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "status": "success",
            "query": query,
            "extracted_keywords": keywords,
            "application_candidates": app_candidates,
            "data_agent_candidates": data_agent_candidates,
            "all_candidates": all_candidates,
            "best_candidate": all_candidates[0] if all_candidates else None,
            "message": "Routing candidates retrieved"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting routing candidates: {str(e)}"}

# Multi-Agent Orchestration Endpoints
from app.langgraph_orchestrator import langgraph_orchestrator
from app.enhanced_orchestrator import enhanced_orchestrator
from app.workflow_streamer import workflow_streamer
from app.workflow_endpoints import router as workflow_router

class WorkflowQueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class StreamSubscriptionRequest(BaseModel):
    workflow_id: str = None
    session_id: str = None

@app.post("/orchestration/query")
async def orchestration_query_endpoint(request: WorkflowQueryRequest):
    """Route a query using LangGraph orchestration."""
    try:
        result = await langgraph_orchestrator.execute_workflow(
            user_query=request.query,
            session_id=request.session_id
        )
        return {
            "status": "success",
            "workflow_result": result
        }
    except Exception as e:
        return {"status": "error", "message": f"Error in orchestration query: {str(e)}"}

@app.post("/orchestration/enhanced/resume")
async def enhanced_orchestration_resume(request: dict):
    """Resume enhanced orchestration workflow with user choice."""
    try:
        workflow_id = request.get("workflow_id")
        session_id = request.get("session_id")
        user_choice = request.get("user_choice", {})
        
        if not workflow_id or not session_id:
            return {"status": "error", "message": "Missing workflow_id or session_id"}
        
        result = await enhanced_orchestrator.resume_workflow_with_user_choice(
            workflow_id=workflow_id,
            session_id=session_id,
            user_choice=user_choice
        )
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        return {"status": "error", "message": f"Error resuming enhanced orchestration: {str(e)}"}


@app.post("/orchestration/enhanced")
async def enhanced_orchestration_query(request: WorkflowQueryRequest):
    """Route a query using Enhanced Multi-Step Orchestration with iterative agent selection."""
    try:
        result = await enhanced_orchestrator.execute_workflow(
            user_query=request.query,
            session_id=request.session_id
        )
        return {
            "status": "success",
            "workflow_result": result
        }
    except Exception as e:
        return {"status": "error", "message": f"Error in enhanced orchestration: {str(e)}"}

@app.post("/orchestration/enhanced/stream")
async def enhanced_orchestration_stream(request: WorkflowQueryRequest):
    """Enhanced multi-step orchestration with real-time streaming."""
    import json
    import asyncio
    import uuid
    import time
    from app.config import config
    
    async def generate_enhanced_streaming():
        """Generate enhanced streaming response with multi-step orchestration."""
        client_id = str(uuid.uuid4())
        workflow_task = None
        workflow_completed = False
        
        try:
            # Subscribe to workflow events
            workflow_streamer.add_client(client_id, session_id=request.session_id)
            
            # Send initial connection status
            yield f"data: {json.dumps({
                'type': 'connection_established',
                'message': '🚀 Enhanced multi-step workflow initiated',
                'timestamp': time.time()
            }, default=str)}\n\n"
            
            # Simple Orchestrator step mapping - 5-step workflow
            step_mapping = {
                # Core workflow events
                "workflow_started": {"name": "Workflow Started", "status": "loading", "message": "🚀 Simple Orchestrator workflow started", "next": "Understanding Request"},
                "workflow_completed": {"name": "Analysis Complete", "status": "complete", "message": "✅ Analysis completed successfully", "next": "Finished"},
                "error": {"name": "Error", "status": "error", "message": "❌ Error occurred", "next": "Review"},
                
                # Simple Orchestrator specific steps
                "step_started": {"name": "Processing", "status": "loading", "message": "▶️ Starting step", "next": "In Progress"},
                "step_completed": {"name": "Step Complete", "status": "complete", "message": "✅ Step completed", "next": "Next Step"},
                
                # Step-specific mappings by step ID
                "analyze": {"name": "Understanding Request", "status": "loading", "message": "🧠 Understanding your request", "next": "Planning Execution"},
                "execute": {"name": "Executing Plan", "status": "loading", "message": "⚡ Executing agent plan", "next": "Generating Response"},
                "respond": {"name": "Generating Response", "status": "loading", "message": "📝 Generating final response", "next": "Complete"},
                
                # Legacy Enhanced Orchestrator events (for backward compatibility)
                "agent_recommendation": {"name": "Agent Recommendation", "status": "loading", "message": "� Analyzing requirements and selecting agent", "next": "Agent Selection"},  
                "agent_selection": {"name": "Agent Selection", "status": "loading", "message": "🎯 Selecting optimal agent", "next": "Request Preparation"},
                "request_preparation": {"name": "Request Preparation", "status": "loading", "message": "📝 Preparing specific request", "next": "Agent Execution"},
                "agent_execution": {"name": "Agent Execution", "status": "loading", "message": "⚡ Executing agent query", "next": "Result Evaluation"},
                "evaluation": {"name": "Result Evaluation", "status": "loading", "message": "🤔 Evaluating results and planning next step", "next": "Next Step Decision"},
                "llm_routing_decision": {"name": "Agent Selected", "status": "complete", "message": None, "next": "Next Action"},
                "sql_generated": {"name": "Query Generated", "status": "complete", "message": "📝 SQL query generated", "next": "Execution"},
                "data_query": {"name": "Data Retrieved", "status": "complete", "message": "📊 Data successfully retrieved", "next": "Analysis"},
                "user_input_required": {"name": "User Input", "status": "waiting", "message": "❓ Additional input required", "next": "Waiting"}
            }
            
            # Start simple workflow AFTER we start streaming events
            workflow_task = asyncio.create_task(
                simple_orchestrator.execute_workflow(
                    user_query=request.query,
                    session_id=request.session_id
                )
            )
            
            # Track if user input was required
            user_input_required = False
            workflow_result = None
            
            # Stream enhanced workflow events - this now runs concurrently with the workflow
            events_processed = 0
            async for event_sse in workflow_streamer.stream_events(client_id):
                events_processed += 1
                print(f"[DEBUG] Enhanced workflow event {events_processed}: {event_sse[:150]}...")
                
                if event_sse.startswith("data: "):
                    try:
                        event_data = json.loads(event_sse[6:].strip())
                        print(f"[DEBUG] Parsed event data: {event_data}")
                        event_type = event_data.get("type") or event_data.get("event_type")
                        
                        # Handle timeout and heartbeat events
                        if event_type == "timeout":
                            yield f"data: {json.dumps({
                                'type': 'timeout',
                                'message': event_data.get("message", "Multi-step process taking longer than expected"),
                                'elapsed_seconds': event_data.get("elapsed_seconds", 0)
                            }, default=str)}\n\n"
                            break
                            
                        elif event_type == "heartbeat":
                            yield event_sse  # Forward heartbeat
                            continue
                        
                        # Map workflow events to UI steps
                        step_info = None
                        final_message = event_data.get('message', '')
                        
                        # Handle step-specific events
                        if event_type == "step_started" or event_type == "step_completed":
                            step_id = event_data.get("step_id", "unknown")
                            if step_id in step_mapping:
                                step_info = step_mapping[step_id]
                                # Use the actual message from the event
                                final_message = event_data.get('description', '') or step_info.get("message", "Processing...")
                        elif event_type in step_mapping:
                            step_info = step_mapping[event_type]
                            final_message = event_data.get('message', '') or step_info.get("message", "Processing...")
                        
                        # Special handling for workflow_completed - forward with final_answer
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
                            # Don't break here - continue to get the final workflow result
                            workflow_completed = True
                        
                        if step_info:
                            current_step = {
                                "name": step_info["name"],
                                "status": "complete" if event_type == "step_completed" else step_info.get("status", "loading"),
                                "message": final_message,
                                "event_type": event_type,
                                "step_id": event_data.get("step_id", ""),
                                "raw_data": event_data
                            }
                            
                            next_step = {
                                "name": step_info.get("next", "Unknown"),
                                "description": f"Next: {step_info.get('next', 'Processing')}"
                            }
                            
                            yield f"data: {json.dumps({
                                'type': 'enhanced_step_update',
                                'current_step': current_step,
                                'next_step': next_step,
                                'progress': {
                                    'current_step_number': events_processed,
                                    'estimated_total': 'Dynamic',
                                    'workflow_type': 'enhanced_multi_step'
                                },
                                'timestamp': time.time()
                            }, default=str)}\n\n"
                            
                            # CRITICAL: If user input is required, mark flag and wait for workflow completion to check status
                            if event_type == "user_input_required":
                                print(f"[DEBUG] User input required detected in streaming endpoint")
                                user_input_required = True
                                # Don't break here - let the workflow complete and check its return status
                        else:
                            # Forward unmapped events with enhanced context
                            yield f"data: {json.dumps({
                                'type': 'enhanced_event',
                                'event_type': event_type,
                                'message': event_data.get('message', f'Enhanced workflow event: {event_type}'),
                                'data': event_data,
                                'timestamp': time.time()
                            }, default=str)}\n\n"
                            
                    except json.JSONDecodeError:
                        continue
                
                # Check if we should exit the streaming loop
                if workflow_completed and workflow_task and workflow_task.done():
                    print(f"[DEBUG] Workflow completed and task done - exiting streaming loop")
                    break
            
            # Get workflow result and check if user input is required
            if workflow_task.done() and not workflow_task.exception():
                workflow_result = await workflow_task
                safe_result = convert_decimals_to_float(workflow_result)
                
                # Check if workflow returned waiting_input status
                if safe_result.get('status') == 'waiting_input':
                    print(f"[DEBUG] Workflow requires user input - sending user input required event")
                    yield f"data: {json.dumps({
                        'type': 'enhanced_user_input_required',
                        'input_request': safe_result.get('pending_input', {}),
                        'workflow_id': safe_result.get('workflow_id'),
                        'session_id': request.session_id,
                        'message': 'Workflow paused - user input required',
                        'timestamp': time.time()
                    }, default=str)}\n\n"
                else:
                    # Workflow completed normally - send complete result with available data
                    response_data = {
                        'type': 'enhanced_completed',
                        'final_answer': safe_result.get('greeting') or safe_result.get('final_answer'),
                        'greeting': safe_result.get('greeting'),  # Include both for compatibility
                        'workflow_type': 'enhanced_multi_step',
                        'timestamp': time.time()
                    }
                    
                    # Only include visualization data if present (execution_complete path)
                    if 'results' in safe_result:
                        response_data.update({
                            'execution_summary': safe_result.get('execution_summary'),
                            'agents_used': safe_result.get('agents_used', []),
                            'total_data_points': safe_result.get('total_data_points', 0),
                            'results': safe_result.get('results', []),
                            'visualization_ready': safe_result.get('visualization_ready', False),
                            'data_summary': safe_result.get('data_summary', {})
                        })
                        print(f"[DEBUG] Sending enhanced_completed with {len(safe_result.get('results', []))} results")
                    else:
                        print(f"[DEBUG] Sending enhanced_completed without results - workflow_result keys: {list(safe_result.keys()) if safe_result else 'None'}")
                    
                    yield f"data: {json.dumps(response_data, default=str)}\n\n"
                    print(f"[DEBUG] Enhanced_completed event sent successfully")
            elif workflow_task.cancelled():
                print(f"[DEBUG] Workflow was cancelled")
            elif workflow_task.exception():
                print(f"[DEBUG] Workflow failed with exception: {workflow_task.exception()}")
            
            yield f"data: {json.dumps({'type': 'enhanced_stream_end', 'timestamp': time.time()}, default=str)}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'enhanced_error', 'message': str(e), 'timestamp': time.time()}, default=str)}\n\n"
        finally:
            workflow_streamer.remove_client(client_id)
    
    return StreamingResponse(
        generate_enhanced_streaming(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

# Include the workflow endpoints
app.include_router(workflow_router)

@app.post("/orchestration/query/stream")
async def orchestration_query_stream(request: WorkflowQueryRequest):
    """Route a query with simplified step-based streaming progress."""
    import json
    import asyncio
    import uuid
    import time
    from app.config import config
    
    async def generate_streaming_response():
        """Generate simple step-based streaming response."""
        client_id = str(uuid.uuid4())
        
        try:
            # Step tracking with simple states
            current_step = {"name": "Initializing", "status": "loading", "message": "🚀 Starting analysis..."}
            next_step = {"name": "Query Analysis", "description": "AI will analyze your query"}
            
            # Send initial status
            yield f"data: {json.dumps({
                'type': 'step_update',
                'current_step': current_step,
                'next_step': next_step,
                'timestamp': time.time()
            }, default=str)}\n\n"
            
            # Subscribe to workflow events
            workflow_streamer.add_client(client_id, session_id=request.session_id)
            
            # Start workflow
            workflow_task = asyncio.create_task(
                langgraph_orchestrator.execute_workflow(
                    user_query=request.query,
                    session_id=request.session_id
                )
            )
            
            # Simple step mapping for clean UI
            step_mapping = {
                "workflow_started": {"name": "Analysis", "status": "loading", "message": "🤔 AI analyzing your query...", "next": "Planning"},
                "thinking": {"name": "Analysis", "status": "loading", "message": "🤔 AI thinking...", "next": "Planning"}, 
                "llm_planning": {"name": "Planning", "status": "loading", "message": "📋 Creating execution plan...", "next": "Agent Selection"},
                "plan_created": {"name": "Planning", "status": "complete", "message": "✅ Plan created", "next": "Agent Selection"},
                "agent_selected": {"name": "Agent Selection", "status": "complete", "message": "🎯 Agent selected", "next": "Query Generation"},
                "forming_query": {"name": "Query Generation", "status": "loading", "message": "🧠 Generating query...", "next": "Database Execution"},
                "sql_generated": {"name": "Query Generation", "status": "complete", "message": "✅ Query generated", "next": "Database Execution"},
                "executing_query": {"name": "Database Execution", "status": "loading", "message": "⚡ Running query...", "next": "Result Processing"},
                "data_received": {"name": "Database Execution", "status": "complete", "message": "📊 Data retrieved", "next": "Result Processing"},
                "workflow_completed": {"name": "Result Processing", "status": "complete", "message": "✅ Complete!", "next": "Finished"}
            }
            
            # Process workflow events with timeout-aware streaming
            events_processed = 0
            async for event_sse in workflow_streamer.stream_events(client_id):
                events_processed += 1
                print(f"[DEBUG] Processed event {events_processed}: {event_sse[:100]}...")  # Debug log
                
                if event_sse.startswith("data: "):
                    try:
                        event_data = json.loads(event_sse[6:].strip())
                        event_type = event_data.get("event_type")
                        
                        # Handle timeout and heartbeat events
                        if event_type == "timeout":
                            yield f"data: {json.dumps({
                                'type': 'timeout',
                                'message': event_data.get("message", "Process taking longer than expected"),
                                'elapsed_seconds': event_data.get("elapsed_seconds", 0)
                            }, default=str)}\n\n"
                            break
                            
                        elif event_type == "heartbeat":
                            # Forward heartbeat to client
                            yield event_sse
                            continue
                        
                        # Map workflow events to simple steps
                        if event_type in step_mapping:
                            step_info = step_mapping[event_type]
                            
                            # Update current step
                            current_step = {
                                "name": step_info["name"],
                                "status": step_info.get("status", "loading"),
                                "message": step_info.get("message", "Processing...")
                            }
                            
                            # Update next step
                            if "next" in step_info:
                                next_step = {
                                    "name": step_info["next"],
                                    "description": f"Next: {step_info['next']}"
                                }
                            
                            yield f"data: {json.dumps({
                                'type': 'step_update', 
                                'current_step': current_step,
                                'next_step': next_step,
                                'timestamp': time.time()
                            }, default=str)}\n\n"
                            
                            if event_type == "workflow_completed":
                                break
                                
                    except json.JSONDecodeError:
                        continue
            
            # Get final result
            if workflow_task.done() and not workflow_task.exception():
                final_result = await workflow_task
                safe_result = convert_decimals_to_float(final_result)
                
                # Build response with available data
                response_data = {
                    'type': 'completed',
                    'final_answer': safe_result.get('greeting') or safe_result.get('final_answer'),
                    'greeting': safe_result.get('greeting'),  # Include both for compatibility
                    'timestamp': time.time()
                }
                
                # Only include visualization data if present (execution_complete path)
                if 'results' in safe_result:
                    response_data.update({
                        'execution_summary': safe_result.get('execution_summary'),
                        'results': safe_result.get('results', []),
                        'visualization_ready': safe_result.get('visualization_ready', False),
                        'data_summary': safe_result.get('data_summary', {})
                    })
                
                yield f"data: {json.dumps(response_data, default=str)}\n\n"
            
            yield f"data: {json.dumps({'type': 'stream_end', 'timestamp': time.time()}, default=str)}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'timestamp': time.time()}, default=str)}\n\n"
        finally:
            workflow_streamer.remove_client(client_id)
    
    return StreamingResponse(
        generate_streaming_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive", 
            "Access-Control-Allow-Origin": "*"
        }
    )

# Real-time Streaming Endpoints
@app.get("/stream/workflow/{workflow_id}")
async def stream_workflow_events(workflow_id: str):
    """Stream real-time events for a specific workflow."""
    client_id = str(uuid.uuid4())
    
    def generate_events():
        # Subscribe to workflow events
        workflow_streamer.add_client(client_id, workflow_id=workflow_id)
        
        # Stream events
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def async_stream():
                async for event in workflow_streamer.stream_events(client_id):
                    yield event
            
            for event in loop.run_until_complete(async_stream()):
                yield event
                
        except Exception as e:
            yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
        finally:
            workflow_streamer.remove_client(client_id)
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.get("/stream/session/{session_id}")
async def stream_session_events(session_id: str):
    """Stream real-time events for a specific session."""
    client_id = str(uuid.uuid4())
    
    def generate_events():
        # Subscribe to session events
        workflow_streamer.add_client(client_id, session_id=session_id)
        
        # Stream events
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def async_stream():
                async for event in workflow_streamer.stream_events(client_id):
                    yield event
            
            for event in loop.run_until_complete(async_stream()):
                yield event
                
        except Exception as e:
            yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
        finally:
            workflow_streamer.remove_client(client_id)
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.get("/stream")
async def stream_general():
    """Stream general workflow events (alias for /stream/all)."""
    return await stream_all_events()

@app.get("/stream/all")
async def stream_all_events():
    """Stream all workflow events."""
    client_id = str(uuid.uuid4())
    
    def generate_events():
        # Subscribe to all events
        workflow_streamer.add_client(client_id)
        
        # Stream events
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def async_stream():
                async for event in workflow_streamer.stream_events(client_id):
                    yield event
            
            for event in loop.run_until_complete(async_stream()):
                yield event
                
        except Exception as e:
            yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
        finally:
            workflow_streamer.remove_client(client_id)
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.get("/stream/stats")
def get_stream_stats():
    """Get streaming statistics."""
    try:
        return {
            "status": "success",
            "stats": workflow_streamer.get_client_stats()
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting stream stats: {str(e)}"}

@app.get("/stream/events/{workflow_id}")
def get_workflow_events(workflow_id: str, limit: int = 100):
    """Get recent events for a workflow."""
    try:
        events = workflow_streamer.get_workflow_events(workflow_id, limit)
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "events": events,
            "total_events": len(events)
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting workflow events: {str(e)}"}

# Database Testing Endpoints
from app.database_query_executor import database_query_executor

class DatabaseTestRequest(BaseModel):
    vault_key: str
    connection_type: str
    test_query: str = None

@app.post("/database/test_connection")
def test_database_connection(request: DatabaseTestRequest):
    """Test database connection for debugging."""
    try:
        result = database_query_executor.test_connection(
            vault_key=request.vault_key,
            connection_type=request.connection_type
        )
        return {
            "status": "success",
            "test_result": result
        }
    except Exception as e:
        return {"status": "error", "message": f"Error testing connection: {str(e)}"}

@app.post("/database/test_query")
def test_database_query(request: DatabaseTestRequest):
    """Test database query for debugging."""
    try:
        if not request.test_query:
            request.test_query = "SELECT 1 as test_column"
            
        result = database_query_executor.execute_query(
            vault_key=request.vault_key,
            connection_type=request.connection_type,
            sql_query=request.test_query,
            limit=10
        )
        return {
            "status": "success",
            "query_result": result
        }
    except Exception as e:
        return {"status": "error", "message": f"Error testing query: {str(e)}"}

@app.get("/database/drivers")
def get_database_drivers():
    """Get available database drivers for debugging."""
    try:
        drivers_info = {}
        
        # Check ODBC drivers (for MSSQL)
        try:
            import pyodbc
            drivers_info["odbc_drivers"] = pyodbc.drivers()
            drivers_info["sql_server_drivers"] = [d for d in pyodbc.drivers() if 'SQL Server' in d]
        except ImportError:
            drivers_info["odbc_drivers"] = "pyodbc not installed"
        
        # Check other database libraries
        libraries = {
            "psycopg2": "PostgreSQL",
            "mysql.connector": "MySQL", 
            "google.cloud.bigquery": "BigQuery",
            "databricks.sql": "Databricks",
            "ibm_db": "DB2"
        }
        
        for lib, db_name in libraries.items():
            try:
                __import__(lib)
                drivers_info[f"{db_name.lower()}_available"] = True
            except ImportError:
                drivers_info[f"{db_name.lower()}_available"] = False
        
        return {
            "status": "success",
            "drivers": drivers_info
        }
    except Exception as e:
        return {"status": "error", "message": f"Error checking drivers: {str(e)}"}

# Enhanced LLM Orchestrator Endpoints
from app.enhanced_llm_orchestrator import EnhancedLLMOrchestrator
from app.orchestrator import simple_orchestrator

# Initialize the enhanced LLM orchestrator (singleton) - keeping for legacy features
enhanced_llm_orchestrator = EnhancedLLMOrchestrator()

class EnhancedQueryRequest(BaseModel):
    query: str
    session_id: str = None
    include_analysis: bool = True
    max_steps: int = 5
    browser_fingerprint: dict = None  # Optional browser fingerprint for session generation

class BrowserFingerprint(BaseModel):
    user_agent: str = None
    screen_resolution: str = None
    timezone: str = None
    language: str = None
    platform: str = None
    viewport_size: str = None
    color_depth: int = None
    device_memory: float = None
    hardware_concurrency: int = None
    connection_type: str = None
    cookie_enabled: bool = None
    do_not_track: str = None
    canvas_fingerprint: str = None  # Hash of canvas rendering
    webgl_fingerprint: str = None   # WebGL renderer info

class SessionRequest(BaseModel):
    browser_fingerprint: BrowserFingerprint = None

class NewConversationRequest(BaseModel):
    current_session_id: str

class FeedbackRequest(BaseModel):
    workflow_id: str
    feedback: str  # "thumbs_up", "thumbs_down", "positive", "negative"
    comments: str = None

@app.post("/enhanced/session/create")
def create_session_with_fingerprint(request: SessionRequest):
    """
    Create a new session using browser fingerprint.
    This endpoint is designed to work with Streamlit's client-side JavaScript.
    """
    try:
        fingerprint_data = request.browser_fingerprint.dict() if request.browser_fingerprint else {}
        
        # Generate session using fingerprint
        user_id, session_id = enhanced_llm_orchestrator.session_manager.generate_session_from_fingerprint(
            fingerprint_data
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "session_id": session_id,
            "conversation_id": session_id,  # For compatibility
            "fingerprint_quality": enhanced_llm_orchestrator.session_manager._calculate_fingerprint_quality(fingerprint_data),
            "message": "Session created successfully"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to create session"
        }

@app.post("/enhanced/session/{session_id}/new_conversation")
def create_new_conversation(session_id: str):
    """
    Create a new conversation within an existing session.
    This is like starting a new chat thread.
    """
    try:
        new_conversation_id = enhanced_llm_orchestrator.session_manager.create_new_conversation(session_id)
        
        return {
            "status": "success",
            "session_id": session_id,
            "new_conversation_id": new_conversation_id,
            "conversation_title": "New Chat",  # Placeholder title, will be updated with first query
            "message": "New conversation created"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to create new conversation"
        }

@app.get("/enhanced/session/{session_id}/conversations")
def get_user_conversations(session_id: str):
    """
    Get all conversations for a user session.
    """
    try:
        conversations = enhanced_llm_orchestrator.session_manager.get_user_conversations(session_id)
        
        return {
            "status": "success",
            "session_id": session_id,
            "conversations": conversations,
            "total_conversations": len(conversations)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to get conversations"
        }

async def generate_conversation_title_from_query(query: str) -> str:
    """
    Generate a concise conversation title from the first query using LLM.
    """
    try:
        # Use LLM to generate a concise title
        title_prompt = f"""
        Generate a concise, descriptive title (2-6 words) for a conversation that starts with this query:
        "{query}"
        
        The title should be:
        - Brief and clear
        - Descriptive of the main topic
        - Professional but friendly
        - No quotes or special characters
        
        Examples:
        Query: "Show me sales data for Q3" → Title: "Q3 Sales Analysis"
        Query: "What products are out of stock?" → Title: "Inventory Stock Check" 
        Query: "Generate a customer report" → Title: "Customer Report Generation"
        Query: "yes please" → Title: "Follow-up Request"
        
        Return only the title, nothing else.
        """
        
        title_response = enhanced_llm_orchestrator.llm_client.invoke(title_prompt, timeout=10)
        title = title_response.content.strip().strip('"').strip("'")
        
        # Fallback if title is too long or empty
        if len(title) > 50 or len(title) < 3:
            # Generate simple title from query
            words = query.split()[:4]  # First 4 words
            title = " ".join(words).title()
            if len(title) > 50:
                title = title[:47] + "..."
        
        return title
        
    except Exception as e:
        print(f"❌ LLM title generation failed: {e}")
        # Simple fallback title
        words = query.split()[:3]
        title = " ".join(words).title() if words else "New Conversation"
        return title

@app.post("/enhanced/query")
async def enhanced_query_endpoint(request: EnhancedQueryRequest):
    """
    Enhanced multi-hop query endpoint with LLM planning, result analysis, and feedback learning.
    This endpoint provides:
    - Advanced intent analysis and query refinement
    - Dynamic agent selection based on capabilities
    - Multi-step reasoning with follow-up actions
    - LLM-powered result analysis and insights
    - Learning from user feedback
    - Session management with browser fingerprinting
    """
    try:
        # Simple session handling - Simple Orchestrator manages sessions internally
        session_id = request.session_id
        if not session_id:
            # Generate simple session ID if not provided
            session_id = str(uuid.uuid4())
            print(f"🆔 Generated session: {session_id}")
        
        result = await simple_orchestrator.execute_workflow(
            user_query=request.query,
            session_id=session_id
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "result": result,
            "endpoint_info": {
                "type": "simple_orchestration",
                "features": [
                    "intent_analysis",
                    "single_llm_call", 
                    "agent_execution",
                    "conversation_memory"
                ]
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Enhanced query execution failed"
        }

@app.post("/enhanced/feedback")
async def record_feedback_endpoint(request: FeedbackRequest):
    """
    Record user feedback for completed workflows to improve future recommendations.
    Supports thumbs up/down feedback and detailed comments.
    """
    try:
        result = await enhanced_llm_orchestrator.record_feedback(
            workflow_id=request.workflow_id,
            feedback=request.feedback
        )
        
        return {
            "status": "success",
            "feedback_recorded": result,
            "workflow_id": request.workflow_id,
            "feedback_type": request.feedback
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "message": "Failed to record feedback"
        }

@app.post("/enhanced/resume")
async def enhanced_llm_orchestration_resume(request: dict):
    """Resume enhanced LLM orchestration workflow with user choice."""
    try:
        workflow_id = request.get("workflow_id")
        session_id = request.get("session_id")
        user_choice = request.get("user_choice", {})
        
        if not workflow_id or not session_id:
            return {"status": "error", "message": "Missing workflow_id or session_id"}
        
        result = await enhanced_llm_orchestrator.resume_enhanced_workflow_with_user_choice(
            workflow_id=workflow_id,
            session_id=session_id,
            user_choice=user_choice
        )
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        return {"status": "error", "message": f"Error resuming enhanced LLM orchestration: {str(e)}"}

@app.get("/enhanced/feedback/analytics")
def get_feedback_analytics():
    """Get analytics on user feedback and system performance."""
    try:
        import sqlite3
        
        with sqlite3.connect(enhanced_llm_orchestrator.feedback_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Get feedback distribution
            cursor.execute("""
                SELECT user_feedback, COUNT(*) as count
                FROM interaction_logs
                WHERE user_feedback IS NOT NULL
                GROUP BY user_feedback
            """)
            feedback_dist = dict(cursor.fetchall())
            
            # Get most successful agents
            cursor.execute("""
                SELECT json_extract(agent_calls, '$[0].agent_id') as agent_id,
                       COUNT(*) as usage_count,
                       AVG(execution_time) as avg_time
                FROM interaction_logs
                WHERE user_feedback IN ('thumbs_up', 'positive')
                GROUP BY agent_id
                ORDER BY usage_count DESC
                LIMIT 10
            """)
            top_agents = [
                {"agent_id": row[0], "usage_count": row[1], "avg_time": row[2]}
                for row in cursor.fetchall()
            ]
            
            # Get query patterns
            cursor.execute("""
                SELECT COUNT(*) as total_interactions,
                       COUNT(CASE WHEN user_feedback IN ('thumbs_up', 'positive') THEN 1 END) as positive_feedback,
                       AVG(execution_time) as avg_execution_time
                FROM interaction_logs
            """)
            stats = cursor.fetchone()
            
            return {
                "status": "success",
                "analytics": {
                    "feedback_distribution": feedback_dist,
                    "top_performing_agents": top_agents,
                    "overall_stats": {
                        "total_interactions": stats[0],
                        "positive_feedback_count": stats[1],
                        "satisfaction_rate": stats[1] / stats[0] if stats[0] > 0 else 0,
                        "avg_execution_time": stats[2]
                    }
                }
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to get feedback analytics"
        }

@app.get("/enhanced/schema/{agent_id}")
def get_agent_schema(agent_id: str, force_refresh: bool = False):
    """Get schema information for a data agent from cached registry data (no DB connection needed)."""
    try:
        schema_summary = enhanced_llm_orchestrator.schema_introspector.get_schema_summary(
            agent_id=agent_id,
            force_refresh=force_refresh
        )
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "schema_summary": schema_summary,
            "source": "cached_registry_data",
            "refreshed": force_refresh
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to get schema for agent {agent_id}"
        }

@app.get("/enhanced/patterns/similar")
def get_similar_query_patterns(query: str, limit: int = 5):
    """Find similar successful query patterns for reuse."""
    try:
        similar_patterns = enhanced_llm_orchestrator.feedback_manager.get_similar_queries(
            query=query,
            limit=limit
        )
        
        return {
            "status": "success",
            "query": query,
            "similar_patterns": similar_patterns,
            "pattern_count": len(similar_patterns)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to find similar patterns"
        }

@app.get("/enhanced/capabilities")
def get_enhanced_capabilities():
    """Get information about enhanced orchestrator capabilities."""
    return {
        "status": "success",
        "capabilities": {
            "multi_hop_reasoning": {
                "description": "Execute complex workflows with multiple agent calls",
                "features": ["intent_analysis", "dynamic_planning", "context_awareness"]
            },
            "feedback_learning": {
                "description": "Learn from user feedback to improve recommendations",
                "features": ["thumbs_up_down", "pattern_recognition", "agent_performance_tracking"]
            },
            "schema_introspection": {
                "description": "Automatically discover and cache database schemas",
                "supported_databases": ["mssql", "postgresql", "mysql", "bigquery"]
            },
            "result_analysis": {
                "description": "LLM-powered analysis of query results with insights",
                "features": ["trend_detection", "anomaly_identification", "follow_up_suggestions"]
            },
            "dialect_awareness": {
                "description": "Generate database-specific SQL syntax",
                "supported_dialects": ["mssql", "postgresql", "mysql", "bigquery", "spark"]
            },
            "sql_safety_validation": {
                "description": "Comprehensive SQL safety checks to prevent data modification",
                "features": [
                    "dangerous_keyword_detection",
                    "injection_pattern_detection", 
                    "read_only_enforcement",
                    "multi_statement_prevention",
                    "schema_modification_blocking"
                ],
                "blocked_operations": [
                    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", 
                    "TRUNCATE", "MERGE", "BULK", "EXEC", "EXECUTE"
                ]
            }
        },
        "endpoints": {
            "/enhanced/query": "Main enhanced query endpoint with multi-hop reasoning",
            "/enhanced/feedback": "Record user feedback for workflow improvement",
            "/enhanced/feedback/analytics": "Get analytics on system performance and feedback",
            "/enhanced/schema/{agent_id}": "Get dynamic schema information for data agents",
            "/enhanced/patterns/similar": "Find similar successful query patterns",
            "/enhanced/sql/validate": "Validate SQL query safety without execution"
        }
    }

class SQLValidationRequest(BaseModel):
    sql_query: str
    database_type: str = "mssql"

@app.post("/enhanced/sql/validate")
def validate_sql_safety(request: SQLValidationRequest):
    """
    Validate SQL query safety without executing it.
    This endpoint allows testing the SQL safety validation system.
    """
    try:
        safety_check = enhanced_llm_orchestrator.sql_validator.validate_sql_safety(request.sql_query)
        
        return {
            "status": "success",
            "original_query": request.sql_query,
            "database_type": request.database_type,
            "validation_result": safety_check,
            "recommendation": "Query is safe to execute" if safety_check["is_safe"] else "Query is NOT safe - contains dangerous operations"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to validate SQL query"
        }

# Streaming endpoint for enhanced workflows
@app.get("/enhanced/stream/{workflow_id}")
async def stream_enhanced_workflow(workflow_id: str):
    """Stream real-time updates for an enhanced workflow execution."""
    try:
        from fastapi.responses import StreamingResponse
        import json
        
        def generate_stream():
            # This would connect to your workflow streaming system
            # For now, return a simple demo stream
            yield f"data: {json.dumps({'type': 'workflow_started', 'workflow_id': workflow_id})}\n\n"
            yield f"data: {json.dumps({'type': 'status', 'message': 'Enhanced orchestration streaming available'})}\n\n"
            yield f"data: {json.dumps({'type': 'workflow_completed', 'workflow_id': workflow_id})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to start workflow stream"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
