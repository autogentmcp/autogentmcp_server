"""
Admin controller - handles administrative endpoints.
"""
from fastapi import APIRouter
from app.registry.client import sync_registry
from app.auth.vault_manager import vault_manager
from app.utils.session_manager import session_manager
from app.utils.data_agents_client import data_agents_client
from app.registry.registry_auth_integration import registry_auth_integration
from app.models.responses import BaseResponse

router = APIRouter(tags=["admin"])


@router.post("/sync_registry", response_model=BaseResponse)
def sync_registry_endpoint():
    """Manually sync the registry."""
    try:
        sync_registry()
        return {"status": "success", "message": "Registry synced successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.delete("/sessions/{session_id}", response_model=BaseResponse)
def clear_session(session_id: str):
    """Clear a specific session."""
    session_manager.clear_session(session_id)
    return {"status": "success", "message": f"Session {session_id} cleared"}


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Get session details."""
    history = session_manager.get_session_history(session_id)
    return {
        "session_id": session_id,
        "history": history,
        "summary": session_manager.get_session_summary(session_id)
    }


@router.get("/vault/stats")
def get_vault_stats():
    """Get vault cache statistics."""
    return vault_manager.get_cache_stats()


@router.post("/vault/clear_cache", response_model=BaseResponse)
def clear_vault_cache():
    """Clear vault cache."""
    vault_manager.clear_cache()
    return {"status": "success", "message": "Vault cache cleared"}


@router.post("/vault/preload_cache")
def preload_vault_cache():
    """Manually trigger vault cache preload from registry."""
    vault_manager.preload_cache_from_registry()
    cache_stats = vault_manager.get_cache_stats()
    return {
        "status": "success", 
        "message": "Vault cache preload completed",
        "cache_stats": cache_stats
    }


@router.post("/data-agents/preload_credentials")
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


@router.get("/auth/registry/agents")
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


@router.get("/auth/registry/agent/{app_key}")
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


@router.get("/data-agents/list")
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


@router.get("/data-agents/search")
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


@router.get("/data-agents/{agent_id}")
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
