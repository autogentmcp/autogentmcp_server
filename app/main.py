from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from app.langgraph_router import route_query
from app.auth_handler import auth_handler
from app.session_manager import session_manager
from app.registry import sync_registry
from app.vault_manager import vault_manager
from app.registry_auth_integration import registry_auth_integration
from app.data_agents_client import data_agents_client

app = FastAPI(title="MCP Registry Server", version="1.0.0")

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
    from app.llm_client import llm_client
    return llm_client.test_connection()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
