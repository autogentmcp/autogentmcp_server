from app.registry import get_agents
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class Tool:
    app_key: str
    endpoint_uri: str
    endpoint_description: str
    path_params: Dict[str, Any] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    request_body: Dict[str, Any] = field(default_factory=dict)
    method: str = "GET"
    security: Dict[str, Any] = field(default_factory=dict)

    def to_langgraph_tool(self):
        # Convert to a format suitable for LangGraph tool selection
        return {
            "name": f"{self.app_key}:{self.endpoint_uri}:{self.method}",
            "description": self.endpoint_description,
            "parameters": {
                "path": self.path_params,
                "query": self.query_params,
                "body": self.request_body
            },
            "method": self.method,
            "security": self.security
        }

def build_tools_from_registry() -> List[Tool]:
    """Dynamically build Tool objects from registry metadata."""
    agents = get_agents()
    tools = []
    for agent in agents:
        tools.append(Tool(
            app_key=agent.get("app_key"),
            endpoint_uri=agent.get("endpoint_uri"),
            endpoint_description=agent.get("endpoint_description", ""),
            path_params=agent.get("path_params", {}),
            query_params=agent.get("query_params", {}),
            request_body=agent.get("request_body", {}),
            method=agent.get("method", "GET"),
            security=agent.get("security", {})
        ))
    return tools
