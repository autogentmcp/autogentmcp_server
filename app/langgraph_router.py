from app.llm_client import llm_client
from app.session_manager import session_manager
from app.tool_selector import tool_selector
from app.endpoint_invoker import endpoint_invoker
from app.unified_router import unified_router
from typing import Dict, Any

def route_query(query: dict, session_id: str = "default") -> Dict[str, Any]:
    """Route a user query to the best agent/tool using unified routing with confidence scoring."""
    
    # Use the new unified router that handles both applications and data agents
    try:
        return unified_router.route_query(query["query"], session_id)
    except Exception as e:
        print(f"[LangGraphRouter] Error in unified routing: {e}")
        # Fallback to original application-only routing
        return _fallback_application_route(query, session_id)

def _fallback_application_route(query: dict, session_id: str = "default") -> Dict[str, Any]:
    """Fallback to original application-only routing logic."""
    
    # Get conversation history
    history = session_manager.get_history_string(session_id)
    
    # Select agent and tool
    app_key, tool_name, agent_info, tool_info = tool_selector.select_agent_and_tool(
        query["query"], history
    )
    
    if not app_key or not tool_name:
        final_answer = "I couldn't find an appropriate agent or tool to handle your request."
        session_manager.add_to_session(session_id, query["query"], final_answer)
        return {
            "query": query["query"],
            "route_type": "fallback",
            "agents": [],
            "selected_agent": None,
            "agent_reason": "No suitable agent found",
            "selected_tool": None,
            "reason": "No suitable tool found",
            "call_result": None,
            "final_answer": final_answer
        }
    
    # Prepare endpoint invocation
    method = tool_info.get("method", "GET").upper()
    resolved_endpoint = tool_info.get("resolved_endpoint")
    
    # Fallback to base domain + endpoint if no resolved endpoint
    if not resolved_endpoint:
        base_domain = agent_info.get("base_domain", "")
        endpoint_path = tool_info.get("path", "")
        resolved_endpoint = base_domain + endpoint_path
    
    # Invoke endpoint
    call_result = endpoint_invoker.invoke_registry_endpoint(
        app_key=app_key,
        agent_info=agent_info,
        tool_info=tool_info,
        resolved_endpoint=resolved_endpoint,
        method=method,
        query_params=tool_info.get("query_params", {}),
        body_params=tool_info.get("body_params", {}),
        headers=tool_info.get("headers", {})
    )
    
    # Generate final answer
    final_prompt = llm_client.create_final_answer_prompt(query["query"], call_result)
    final_answer = llm_client.invoke_with_text_response(final_prompt, allow_diagrams=True)
    
    # Update session
    session_manager.add_to_session(session_id, query["query"], final_answer)
    
    return {
        "query": query["query"],
        "route_type": "fallback",
        "agents": [app_key],
        "selected_agent": app_key,
        "agent_reason": "Selected by fallback LLM",
        "selected_tool": tool_name,
        "reason": "Selected by fallback LLM",
        "call_result": call_result,
        "final_answer": final_answer
    }
