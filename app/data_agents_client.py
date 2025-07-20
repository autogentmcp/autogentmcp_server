"""
Data Agents client for fetching and managing database-connected agents.
"""
import time
import httpx
import os
from typing import Dict, Any, List, Optional

class DataAgentsClient:
    """Client for interacting with Data Agents API."""
    
    def __init__(self):
        self._cache = {
            "data_agents": None,
            "last_fetch": 0,
            "ttl": 300  # 5 minutes
        }
    
    def fetch_data_agents(self, force_refresh=False) -> Dict[str, Any]:
        """
        Fetch data agents from the Data Agents API.
        
        Returns:
            Dict of data agents keyed by agent ID
        """
        now = time.time()
        if (not force_refresh and 
            self._cache["data_agents"] is not None and 
            (now - self._cache["last_fetch"] < self._cache["ttl"])):
            return self._cache["data_agents"]
        
        # Get configuration from environment
        data_agents_url = os.getenv("DATA_AGENTS_URL", "http://localhost:8000")
        admin_key = os.getenv("DATA_AGENTS_ADMIN_KEY", "9e2b7c1e-4f3a-4b8e-9c2d-7a1e5b6c8d2f")
        environment = os.getenv("DATA_AGENTS_ENVIRONMENT", "production")
        
        url = f"{data_agents_url}/data-agents/with-environment-details?environment={environment}"
        
        try:
            headers = {"X-Admin-Key": admin_key}
            resp = httpx.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            data_agents = {}
            for agent in data:
                agent_id = agent.get("id")
                if not agent_id:
                    continue
                
                # Extract vault key from environment
                vault_key = None
                environments = agent.get("environments", [])
                if environments:
                    env = environments[0]  # Use first environment
                    vault_key = env.get("vaultKey")
                
                # Process tables and their descriptions for search keywords
                tables = agent.get("tables", [])
                table_keywords = []
                column_keywords = []
                
                for table in tables:
                    table_name = table.get("tableName", "")
                    table_desc = table.get("description", "")
                    table_keywords.append(table_name)
                    if table_desc:
                        table_keywords.extend(table_desc.lower().split())
                    
                    # Process columns
                    for column in table.get("columns", []):
                        column_name = column.get("columnName", "")
                        column_desc = column.get("description", "")
                        column_keywords.append(column_name)
                        if column_desc:
                            column_keywords.extend(column_desc.lower().split())
                
                data_agents[agent_id] = {
                    "id": agent_id,
                    "name": agent.get("name"),
                    "description": agent.get("description"),
                    "status": agent.get("status"),
                    "connection_type": agent.get("connectionType"),
                    "user_id": agent.get("userId"),
                    "vault_key": vault_key,
                    "environments": environments,
                    "tables": tables,
                    "relations": agent.get("relations", []),
                    "table_keywords": list(set(table_keywords)),
                    "column_keywords": list(set(column_keywords)),
                    "type": "data_agent"  # Identifier for routing logic
                }
            
            self._cache["data_agents"] = data_agents
            self._cache["last_fetch"] = now
            return data_agents
            
        except Exception as e:
            print(f"[DataAgentsClient] Error fetching data agents: {e}")
            return self._cache["data_agents"] or {}
    
    def get_data_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information for a specific data agent."""
        data_agents = self.fetch_data_agents()
        return data_agents.get(agent_id)
    
    def search_data_agents_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search data agents by keywords in their table/column names and descriptions.
        
        Returns list of (agent_info, confidence_score) tuples.
        """
        data_agents = self.fetch_data_agents()
        results = []
        
        for agent_id, agent in data_agents.items():
            score = self._calculate_keyword_confidence(agent, keywords)
            if score > 0:
                results.append({
                    "agent": agent,
                    "confidence": score,
                    "type": "data_agent"
                })
        
        # Sort by confidence score (descending)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
    
    def _calculate_keyword_confidence(self, agent: Dict[str, Any], keywords: List[str]) -> float:
        """
        Calculate confidence score for data agent based on keywords.
        
        Scoring logic:
        - Exact table name match: 50 points
        - Exact column name match: 30 points
        - Description keyword match: 10 points
        - Name/description match: 15 points
        """
        score = 0
        keywords_lower = [kw.lower() for kw in keywords]
        
        # Check agent name and description
        agent_name = agent.get("name", "").lower()
        agent_desc = agent.get("description", "").lower()
        
        for keyword in keywords_lower:
            if keyword in agent_name:
                score += 15
            if keyword in agent_desc:
                score += 10
        
        # Check table names (exact matches get higher score)
        table_keywords = [kw.lower() for kw in agent.get("table_keywords", [])]
        for keyword in keywords_lower:
            if keyword in table_keywords:
                # Check if it's an exact table name match
                for table in agent.get("tables", []):
                    if keyword == table.get("tableName", "").lower():
                        score += 50
                        break
                else:
                    score += 30  # Column name or description match
        
        # Check column names
        column_keywords = [kw.lower() for kw in agent.get("column_keywords", [])]
        for keyword in keywords_lower:
            if keyword in column_keywords:
                score += 20
        
        return score
    
    def get_connection_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get database connection information for a data agent."""
        agent = self.get_data_agent_info(agent_id)
        if not agent:
            return None
        
        environments = agent.get("environments", [])
        if not environments:
            return None
        
        env = environments[0]  # Use first environment
        return {
            "connection_type": agent.get("connection_type"),
            "vault_key": agent.get("vault_key"),
            "environment": env,
            "tables": agent.get("tables", [])
        }

# Global instance
data_agents_client = DataAgentsClient()
