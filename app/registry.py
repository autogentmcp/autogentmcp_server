import time
import httpx
import threading

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
    REGISTRY_URL = "http://localhost:8000/apps_with_endpoints"
    try:
        resp = httpx.get(REGISTRY_URL)
        resp.raise_for_status()
        data = resp.json()
        agents = {}
        for app in data.get("applications", []):
            app_key = app.get("app_key")
            agents[app_key] = {
                "description": app.get("app_description", app_key),
                "base_domain": app.get("base_domain"),
                "tools": []
            }
            for ep in app.get("endpoints", []):
                tool = {
                    "name": f"{app_key}:{ep.get('endpoint_uri')}",
                    "endpoint_uri": ep.get("endpoint_uri"),
                    "description": ep.get("endpoint_description", ep.get("endpoint_uri")),
                    "method": ep.get("method", "GET"),
                    "path_params": list(ep.get("path_params", {}).keys()),
                    "query_params": list(ep.get("query_params", {}).keys()),
                    "request_body": ep.get("request_body", {}),
                }
                agents[app_key]["tools"].append(tool)
        _registry_cache["agents"] = agents
        _registry_cache["last_fetch"] = now
        return agents
    except Exception as e:
        print(f"[fetch_agents_and_tools_from_registry] Error: {e}")
        return _registry_cache["agents"] or {}

def _auto_reload_registry():
    """Background thread to auto-refresh the registry cache every ttl seconds."""
    while True:
        try:
            fetch_agents_and_tools_from_registry(force_refresh=True)
        except Exception as e:
            print(f"[registry] Auto-reload error: {e}")
        ttl = fetch_agents_and_tools_from_registry._cache["ttl"]
        time.sleep(ttl)

# Initial load on startup
fetch_agents_and_tools_from_registry(force_refresh=True)

# Start background auto-reload thread
_thread = threading.Thread(target=_auto_reload_registry, daemon=True)
_thread.start()
