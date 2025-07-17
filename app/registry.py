import time
import httpx
import threading
import os
from typing import Dict, Any, Optional

from app.vault_manager import vault_manager

def fetch_agents_and_tools_from_registry(force_refresh=False):
    """Fetches the agent (app) and endpoint (tool) metadata from the registry endpoint and returns a dict of agents. Uses cache unless expired or force_refresh=True."""
    if not hasattr(fetch_agents_and_tools_from_registry, "_cache"):
        fetch_agents_and_tools_from_registry._cache = {
            "agents": None,
            "last_fetch": 0,
            "ttl": 300  # seconds (5 minutes)
        }
    _registry_cache = fetch_agents_and_tools_from_registry._cache
    now = time.time()
    if not force_refresh and _registry_cache["agents"] is not None and (now - _registry_cache["last_fetch"] < _registry_cache["ttl"]):
        return _registry_cache["agents"]
    
    # Get registry configuration from environment
    registry_url = os.getenv("REGISTRY_URL", "http://localhost:8000")
    registry_admin_key = os.getenv("REGISTRY_ADMIN_KEY", "9e2b7c1e-4f3a-4b8e-9c2d-7a1e5b6c8d2f")
    environment = os.getenv("REGISTRY_ENVIRONMENT", "production")
    
    REGISTRY_URL = f"{registry_url}/applications/with-endpoints?environment={environment}"
    
    try:
        headers = {"X-Admin-Key": registry_admin_key}
        resp = httpx.get(REGISTRY_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        agents = {}
        for app in data:
            app_key = app.get("appKey")
            if not app_key:
                continue
                
            # Extract environment information
            environment_info = app.get("environment", {})
            base_domain = environment_info.get("baseDomain", "")
            security_config = environment_info.get("security")
            
            # Create agent structure
            agents[app_key] = {
                "id": app.get("id"),
                "name": app.get("name"),
                "description": app.get("description", app_key),
                "base_domain": base_domain,
                "authentication_method": app.get("authenticationMethod"),
                "health_check_url": app.get("healthCheckUrl"),
                "status": app.get("status"),
                "environment": environment_info,
                "security_config": security_config,
                "tools": []
            }
            
            # Process endpoints as tools
            for endpoint in app.get("endpoints", []):
                tool = {
                    "id": endpoint.get("id"),
                    "name": endpoint.get("name"),
                    "path": endpoint.get("path"),
                    "method": endpoint.get("method", "GET"),
                    "description": endpoint.get("description", endpoint.get("name")),
                    "is_public": endpoint.get("isPublic", False),
                    "path_params": endpoint.get("pathParams", {}),
                    "query_params": endpoint.get("queryParams", {}),
                    "request_body": endpoint.get("requestBody"),
                    "response_body": endpoint.get("responseBody"),
                    "full_name": f"{app_key}:{endpoint.get('name')}",
                    "full_url": f"{base_domain}{endpoint.get('path')}"
                }
                agents[app_key]["tools"].append(tool)
        
        _registry_cache["agents"] = agents
        _registry_cache["last_fetch"] = now
        return agents
    except Exception as e:
        print(f"[fetch_agents_and_tools_from_registry] Error: {e}")
        return _registry_cache["agents"] or {}

def sync_registry():
    """Manually sync the registry cache and preload vault cache."""
    agents = fetch_agents_and_tools_from_registry(force_refresh=True)
    print(f"[Registry] Manual sync completed: {len(agents)} agents loaded")
    
    # Preload vault cache after manual sync
    try:
        from app.vault_manager import vault_manager
        vault_manager.preload_cache_from_registry()
    except Exception as e:
        print(f"[Registry] Error preloading vault cache during manual sync: {e}")
    
    return agents

def get_agent_auth_headers(app_key: str) -> Dict[str, str]:
    """Get authentication headers for a specific agent based on its authentication method."""
    agents = fetch_agents_and_tools_from_registry()
    agent = agents.get(app_key)
    
    if not agent:
        return {}
    
    auth_method = agent.get("authentication_method")
    if not auth_method:
        return {}
    
    security_config = agent.get("security_config")
    if not security_config:
        return {}
    
    vault_key = security_config.get("vaultKey")
    if not vault_key:
        return {}
    
    # Get credentials from vault
    credentials = vault_manager.get_auth_credentials(vault_key)
    if not credentials:
        print(f"[get_agent_auth_headers] Failed to get credentials for vault key: {vault_key}")
        return {}
    
    # Generate headers based on authentication method
    headers = {}
    
    if auth_method == "api_key":
        api_key = credentials.get("api_key") or credentials.get("value")
        if api_key:
            headers["X-API-Key"] = api_key
    
    elif auth_method == "bearer_token":
        token = credentials.get("token") or credentials.get("value")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    
    elif auth_method == "basic_auth":
        username = credentials.get("username")
        password = credentials.get("password")
        if username and password:
            import base64
            encoded_creds = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {encoded_creds}"
    
    elif auth_method == "oauth2":
        access_token = credentials.get("access_token") or credentials.get("token")
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
    
    elif auth_method == "jwt":
        jwt_token = credentials.get("jwt_token") or credentials.get("token")
        if jwt_token:
            headers["Authorization"] = f"Bearer {jwt_token}"
    
    elif auth_method == "azure_subscription":
        subscription_key = credentials.get("subscription_key") or credentials.get("value")
        if subscription_key:
            headers["Ocp-Apim-Subscription-Key"] = subscription_key
    
    elif auth_method == "azure_apim":
        apim_key = credentials.get("apim_key") or credentials.get("value")
        if apim_key:
            headers["Ocp-Apim-Subscription-Key"] = apim_key
    
    elif auth_method == "gcp_service_account":
        access_token = credentials.get("access_token") or credentials.get("token")
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
    
    elif auth_method == "signature_auth":
        signature = credentials.get("signature")
        timestamp = credentials.get("timestamp")
        if signature and timestamp:
            headers["X-Signature"] = signature
            headers["X-Timestamp"] = timestamp
    
    # Handle custom headers
    custom_headers = credentials.get("custom_headers", {})
    if custom_headers:
        headers.update(custom_headers)
    
    return headers

def _auto_reload_registry():
    """Background thread to auto-refresh the registry cache every ttl seconds."""
    while True:
        try:
            agents = fetch_agents_and_tools_from_registry(force_refresh=True)
            print(f"[Registry] Auto-reload completed: {len(agents)} agents loaded")
            
            # Preload vault cache after registry refresh
            try:
                from app.vault_manager import vault_manager
                vault_manager.preload_cache_from_registry()
            except Exception as e:
                print(f"[Registry] Error preloading vault cache during auto-reload: {e}")
                
        except Exception as e:
            print(f"[Registry] Auto-reload error: {e}")
        ttl = fetch_agents_and_tools_from_registry._cache["ttl"]
        time.sleep(ttl)

# Initial load on startup
agents = fetch_agents_and_tools_from_registry(force_refresh=True)
print(f"[Registry] Initial load completed: {len(agents)} agents loaded")

# Preload vault cache with secrets from registry
try:
    from app.vault_manager import vault_manager
    vault_manager.preload_cache_from_registry()
except Exception as e:
    print(f"[Registry] Error preloading vault cache: {e}")

# Start background auto-reload thread
_thread = threading.Thread(target=_auto_reload_registry, daemon=True)
_thread.start()
