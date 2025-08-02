import time
import httpx
import threading
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.auth.vault_manager import vault_manager

# Global cache for detailed data agent responses
_data_agent_details_cache = {
    "cache": {},
    "last_fetch": 0,
    "ttl": 300  # 5 minutes TTL
}

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
    
    # Fetch both applications and data agents
    applications_url = f"{registry_url}/applications/with-endpoints?environment={environment}"
    data_agents_url = f"{registry_url}/data-agents/with-environment-details?environment={environment}"
    
    try:
        headers = {"X-Admin-Key": registry_admin_key}
        agents = {}
        
        # Fetch applications first
        print(f"[Registry] Fetching applications from: {applications_url}")
        app_resp = httpx.get(applications_url, headers=headers, timeout=10)
        app_resp.raise_for_status()
        applications_data = app_resp.json()
        print(f"[Registry] Found {len(applications_data)} applications")
        
        # Process applications
        for app in applications_data:
            app_key = app.get("appKey")
            if not app_key:
                continue
                
            # Extract environment information
            environment_info = app.get("environment", {})
            base_domain = environment_info.get("baseDomain", "")
            security_config = environment_info.get("security")
            
            # Extract vault key from security config for applications
            vault_key = ""
            if security_config and security_config.get("vaultKey"):
                vault_key = security_config["vaultKey"]
            
            # Create agent structure
            agents[app_key] = {
                "id": app.get("id"),
                "name": app.get("name"),
                "appKey": app_key,  # Add appKey field explicitly
                "description": app.get("description", app_key),
                "base_domain": base_domain,
                "authentication_method": app.get("authenticationMethod"),
                "health_check_url": app.get("healthCheckUrl"),
                "status": app.get("status"),
                "environment": environment_info,
                "security_config": security_config,
                "vault_key": vault_key,  # Add vault key for applications
                "agent_type": "application",  # Explicitly set type
                "tools": [],
                "endpoints": []  # Add endpoints field for enhanced orchestrator
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
                # Also add to endpoints field for enhanced orchestrator
                agents[app_key]["endpoints"].append(endpoint)
                # Also add to endpoints field for enhanced orchestrator
                agents[app_key]["endpoints"].append(endpoint)
        
        # Fetch data agents
        print(f"[Registry] Fetching data agents from: {data_agents_url}")
        try:
            data_agent_resp = httpx.get(data_agents_url, headers=headers, timeout=10)
            data_agent_resp.raise_for_status()
            data_agents_data = data_agent_resp.json()
            print(f"[Registry] Found {len(data_agents_data)} data agents")
            
            # Process data agents
            for data_agent in data_agents_data:
                # Validate data_agent structure
                if not isinstance(data_agent, dict):
                    print(f"[Registry] WARNING: Expected dict but got {type(data_agent)}")
                    continue
                
                agent_key = data_agent.get("id")  # Use 'id' as the key
                if not agent_key:
                    print(f"[Registry] WARNING: Data agent missing 'id' field: {data_agent}")
                    continue
                
                # Cache the full data agent response for later detailed access
                _data_agent_details_cache["cache"][agent_key] = data_agent
                
                # Extract connection information from environments
                environments = data_agent.get("environments", [])
                if not isinstance(environments, list):
                    print(f"[Registry] WARNING: environments is not a list for agent {agent_key}: {type(environments)}")
                    environments = []
                
                connection_info = {}
                database_type = data_agent.get("connectionType", "unknown")
                
                # Get connection config from the first active environment
                for env in environments:
                    if not isinstance(env, dict):
                        print(f"[Registry] WARNING: environment is not a dict for agent {agent_key}: {type(env)}")
                        continue
                    if env.get("status") == "ACTIVE":
                        connection_info = env.get("connectionConfig", {})
                        break
                
                # If no active environment, use the first one
                if not connection_info and environments:
                    first_env = environments[0]
                    if isinstance(first_env, dict):
                        connection_info = first_env.get("connectionConfig", {})
                    else:
                        print(f"[Registry] WARNING: First environment is not a dict for agent {agent_key}: {type(first_env)}")
                
                # Create data agent structure - capture ALL available data
                agents[agent_key] = {
                    "id": data_agent.get("id"),
                    "name": data_agent.get("name"),
                    "description": data_agent.get("description", f"Data agent for {database_type}"),
                    "database_type": database_type,
                    "connection_info": connection_info,
                    "status": data_agent.get("status"),
                    "agent_type": "data_agent",  # Mark as data agent
                    "security_config": {},  # Will be populated from environments
                    
                    # Capture available data from the new structure
                    "schema": connection_info.get("schema", ""),
                    "tables": data_agent.get("tables", []),
                    "metadata": {},  # Not available in this structure
                    "capabilities": [],  # Not available in this structure
                    "sample_queries": [],  # Not available in this structure
                    "business_context": data_agent.get("description", ""),
                    "data_sources": [],  # Not available in this structure
                    "environments": environments,  # Keep all environment details
                    "relations": data_agent.get("relations", []),
                    "vault_key": "",  # Will be populated from environments
                    "created_at": data_agent.get("createdAt"),
                    "updated_at": data_agent.get("updatedAt"),
                    
                    # Store the complete original data for reference
                    "raw_data": data_agent,
                    
                    "tools": [
                        {
                            "id": f"{agent_key}_query",
                            "name": "execute_sql_query",
                            "description": f"Execute SQL queries on {database_type} database",
                            "method": "POST",
                            "database_type": database_type
                        },
                        {
                            "id": f"{agent_key}_schema",
                            "name": "get_database_schema",
                            "description": f"Get schema information for {database_type} database",
                            "method": "GET",
                            "database_type": database_type
                        }
                    ]
                }
                
                # Extract vault key from the active environment
                for env in environments:
                    if not isinstance(env, dict):
                        continue
                    if env.get("status") == "ACTIVE":
                        # For data agents, vault key is directly in the environment
                        if env.get("vaultKey"):
                            agents[agent_key]["vault_key"] = env["vaultKey"]
                            agents[agent_key]["security_config"] = {"vaultKey": env["vaultKey"]}
                        break
                
        except Exception as e:
            print(f"[Registry] Error fetching data agents: {e}")
            print("[Registry] Continuing with applications only...")
        
        print(f"[Registry] Total agents loaded: {len(agents)} (applications + data agents)")
        
        # Update data agent details cache timestamp
        _data_agent_details_cache["last_fetch"] = now
        
        _registry_cache["agents"] = agents
        _registry_cache["last_fetch"] = now
        return agents
    except Exception as e:
        print(f"[fetch_agents_and_tools_from_registry] Error: {e}")
        return _registry_cache["agents"] or {}

def get_cached_data_agent_details(agent_key: str) -> Optional[Dict[str, Any]]:
    """Get cached detailed data agent information for LLM query generation."""
    import time
    
    # Check if cache is still valid
    now = time.time()
    if (now - _data_agent_details_cache["last_fetch"]) > _data_agent_details_cache["ttl"]:
        print(f"[Registry] Data agent details cache expired, refreshing...")
        # Refresh the cache by fetching agents again
        fetch_agents_and_tools_from_registry(force_refresh=True)
    
    # Get the cached detailed data
    cached_data = _data_agent_details_cache["cache"].get(agent_key)
    if not cached_data:
        print(f"[Registry] No cached details found for data agent: {agent_key}")
        return None
    
    print(f"[Registry] Retrieved cached details for data agent: {agent_key}")
    print(f"[Registry] Available fields: {list(cached_data.keys())}")
    
    return cached_data

def get_enhanced_agent_details_for_llm(agent_key: str) -> Optional[Dict[str, Any]]:
    """Get enhanced agent details specifically formatted for LLM query generation."""
    # Get basic agent info from registry
    agents = fetch_agents_and_tools_from_registry()
    agent = agents.get(agent_key)
    # print(f"[Registry] Found agent: {agent_key} - {agent.get('tables', 'Unknown')}")
    
    
    if not agent:
        print(f"[Registry] Agent not found: {agent_key}")
        return None
    
    # If it's a data agent, get cached detailed information
    if agent.get("agent_type") == "data_agent":
        cached_details = get_cached_data_agent_details(agent_key)
        
        if cached_details:
            # Combine basic agent info with detailed cached data
            enhanced_details = agent.copy()
            
            # Extract rich information from the cached response
            tables = cached_details.get("tables", [])
            relations = cached_details.get("relations", [])
            environments = cached_details.get("environments", [])
            
            # Build comprehensive schema information
            schema_info_parts = []
            table_summaries = []
            sample_queries = []
            
            # Process tables for detailed schema info
            for table in tables:
                table_name = table.get("tableName", "")
                schema_name = table.get("schemaName", "public")
                description = table.get("description", "")
                row_count = table.get("rowCount", 0)
                
                if table_name:
                    table_info = f"\n-- Table: {schema_name}.{table_name}"
                    if description:
                        # Truncate long descriptions but keep meaningful content
                        desc_preview = description[:300] + "..." if len(description) > 300 else description
                        table_info += f"\n-- Description: {desc_preview}"
                    table_info += f"\n-- Row Count: {row_count}"
                    
                    # Add key column information
                    columns = table.get("columns", [])
                    if columns:
                        table_info += f"\n-- Key Columns:"
                        for col in columns:  # Limit to first 8 columns to avoid token overflow
                            col_name = col.get("columnName", "")
                            data_type = col.get("dataType", "")
                            nullable = "NULL" if col.get("isNullable", True) else "NOT NULL"
                            
                            if col.get("isPrimaryKey", False):
                                table_info += f"\n--   {col_name}: {data_type} PRIMARY KEY"
                            elif col.get("isForeignKey", False):
                                ref_table = col.get("referencedTable", "")
                                table_info += f"\n--   {col_name}: {data_type} FK -> {ref_table}"
                            else:
                                table_info += f"\n--   {col_name}: {data_type} {nullable}"
                    
                    schema_info_parts.append(table_info)
                    table_summaries.append({
                        "name": f"{schema_name}.{table_name}",
                        "description": description[:100] + "..." if len(description) > 100 else description,
                        "row_count": row_count,
                        "column_count": len(columns)
                    })
            print(f"[Registry] table summaries for LLM schema: {table_summaries}")
            # Extract sample queries from relations
            for relation in relations:
                example = relation.get("example", "")
                description = relation.get("description", "")
                if example and example not in sample_queries:
                    query_with_context = f"-- {description}\n{example}" if description else example
                    sample_queries.append(query_with_context)
            
            # Get connection details and vault key from active environment
            connection_details = {}
            vault_key = ""
            if environments:
                active_env = next((env for env in environments if env.get("status") == "ACTIVE"), environments[0] if environments else None)
                if active_env:
                    connection_details = active_env.get("connectionConfig", {})
                    vault_key = active_env.get("vaultKey", "")
            
            # Create enhanced details with comprehensive information
            enhanced_details.update({
                # Core information
                "name": cached_details.get("name", enhanced_details.get("name", "")),
                "description": cached_details.get("description", enhanced_details.get("description", "")),
                "database_type": cached_details.get("connectionType", enhanced_details.get("database_type", "")),
                "connection_type": cached_details.get("connectionType", enhanced_details.get("database_type", "")),  # Ensure connection_type is available
                
                # Security information from environment (correct location)
                "vault_key": vault_key,
                "security_config": {"vaultKey": vault_key} if vault_key else {},
                
                # Rich schema information for LLM
                "schema": "\n".join(schema_info_parts),
                "tables": table_summaries,  # Summarized tables for display
                "tables_with_columns": tables,  # Full table data with columns for LLM
                "table_relations": relations,
                "sample_queries": sample_queries,
                
                # Metadata for context
                "metadata": {
                    "total_tables": len(tables),
                    "total_relations": len(relations),
                    "database_type": cached_details.get("connectionType", "unknown"),
                    "status": cached_details.get("status", "unknown"),
                    "total_rows": sum(t.get("rowCount", 0) for t in tables)
                },
                
                # Capabilities and context
                "capabilities": [
                    "Complex SQL query execution",
                    "Multi-table joins with relationships",
                    "Schema introspection and analysis", 
                    "Data type validation and constraints"
                ],
                "business_context": cached_details.get("description", ""),
                "connection_details": connection_details,
                "environments": environments,
                
                # Store full response for debugging
                "raw_response": cached_details
            })
            
            print(f"[Registry] Enhanced details prepared for LLM:")
            print(f"  - Agent: {enhanced_details.get('name')}")
            print(f"  - Tables: {len(tables)}")
            print(f"  - Relations: {len(relations)}")
            print(f"  - Sample Queries: {len(sample_queries)}")
            print(f"  - Total Fields: {len(enhanced_details)}")
            
            return enhanced_details
    
    # For non-data agents, return basic agent info
    return agent

def sync_registry():
    """Manually sync the registry cache and preload vault cache."""
    global _data_agent_details_cache
    
    # Store current cache keys for comparison (don't clear yet)
    old_cache_keys = set(_data_agent_details_cache["cache"].keys())
    print(f"[Registry] Manual sync starting - current cache has {len(old_cache_keys)} entries")
    
    agents = fetch_agents_and_tools_from_registry(force_refresh=True)
    
    # Now that new data is loaded, remove any stale entries
    new_cache_keys = set(_data_agent_details_cache["cache"].keys())
    stale_keys = old_cache_keys - new_cache_keys
    if stale_keys:
        print(f"[Registry] Removing {len(stale_keys)} stale cache entries: {list(stale_keys)[:3]}...")
        for key in stale_keys:
            _data_agent_details_cache["cache"].pop(key, None)
    
    print(f"[Registry] Manual sync completed: {len(agents)} agents loaded")
    print(f"[Registry] Data agent details cache updated with {len(_data_agent_details_cache['cache'])} entries")
    
    # Preload vault cache after manual sync
    try:
        from app.auth.vault_manager import vault_manager
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
            # Store current cache keys for comparison (don't clear yet)
            old_cache_keys = set(_data_agent_details_cache["cache"].keys())
            
            agents = fetch_agents_and_tools_from_registry(force_refresh=True)
            
            # Now that new data is loaded, remove any stale entries
            new_cache_keys = set(_data_agent_details_cache["cache"].keys())
            stale_keys = old_cache_keys - new_cache_keys
            if stale_keys:
                print(f"[Registry] Auto-reload removing {len(stale_keys)} stale cache entries")
                for key in stale_keys:
                    _data_agent_details_cache["cache"].pop(key, None)
            
            print(f"[Registry] Auto-reload completed: {len(agents)} agents loaded")
            print(f"[Registry] Data agent details cache updated with {len(_data_agent_details_cache['cache'])} entries")
            
            # Preload vault cache after registry refresh
            try:
                from app.auth.vault_manager import vault_manager
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
    from app.auth.vault_manager import vault_manager
    vault_manager.preload_cache_from_registry()
except Exception as e:
    print(f"[Registry] Error preloading vault cache: {e}")

# Start background auto-reload thread
_thread = threading.Thread(target=_auto_reload_registry, daemon=True)
_thread.start()
