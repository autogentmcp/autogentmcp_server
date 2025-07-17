from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from app.langgraph_router import route_query
from app.auth_handler import auth_handler
from app.session_manager import session_manager
from app.registry import sync_registry
from app.vault_manager import vault_manager
from app.registry_auth_integration import registry_auth_integration

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
