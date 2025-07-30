"""
Tool and agent selection logic.
"""
from typing import Dict, Any, Optional, Tuple
from app.ollama_client import ollama_client
from app.registry import fetch_agents_and_tools_from_registry

class ToolSelector:
    """Handle agent and tool selection using LLM."""
    
    def __init__(self):
        pass
    
    def select_agent_and_tool(
        self, 
        query: str, 
        history: str = ""
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Select the best agent and tool for a query.
        
        Args:
            query: User query
            history: Conversation history
            
        Returns:
            Tuple of (app_key, tool_name, agent_info, tool_info)
        """
        # Get available agents
        agents = fetch_agents_and_tools_from_registry()
        
        # Step 1: Select agent
        app_key = self._select_agent(query, agents, history)
        if not app_key:
            return None, None, None, None
        
        agent_info = agents[app_key]
        
        # Step 2: Select tool
        tool_name, tool_info = self._select_tool(query, app_key, agent_info)
        
        return app_key, tool_name, agent_info, tool_info
    
    def _select_agent(self, query: str, agents: Dict[str, Any], history: str) -> Optional[str]:
        """Select the best agent for the query."""
        prompt = ollama_client.create_agent_selection_prompt(query, agents, history)
        
        result = ollama_client.invoke_with_json_response(prompt)
        if not result:
            print("[ToolSelector] Failed to parse agent selection")
            return None
        
        agent_name = result.get("agent")
        reason = result.get("reason")
        
        print(f"[ToolSelector] Selected agent: {agent_name}, reason: {reason}")
        
        if agent_name not in agents:
            print(f"[ToolSelector] Selected agent '{agent_name}' not found in registry")
            return None
        
        return agent_name
    
    def _select_tool(self, query: str, app_key: str, agent_info: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Select the best tool for the query."""
        tools = agent_info['tools']
        
        if not tools:
            print(f"[ToolSelector] No tools available for agent {app_key}")
            return None, None
        
        # If only one tool, select it
        if len(tools) == 1:
            tool = tools[0]
            print(f"[ToolSelector] Only one tool available: {tool['name']}")
            return tool['name'], tool
        
        # Use LLM to select tool
        prompt = ollama_client.create_tool_selection_prompt(query, app_key, agent_info)
        
        result = ollama_client.invoke_with_json_response(prompt)
        if not result:
            print("[ToolSelector] Failed to parse tool selection")
            return None, None
        
        tool_name = result.get("tool")
        reason = result.get("reason")
        
        print(f"[ToolSelector] Selected tool: {tool_name}, reason: {reason}")
        
        # Find the selected tool
        selected_tool = next((t for t in tools if t["name"] == tool_name), None)
        if not selected_tool:
            print(f"[ToolSelector] Selected tool '{tool_name}' not found in agent tools")
            return None, None
        
        # Add LLM-generated parameters to tool info
        selected_tool = selected_tool.copy()
        selected_tool.update({
            'resolved_endpoint': result.get('resolved_endpoint'),
            'query_params': result.get('query_params', {}),
            'body_params': result.get('body_params', {}),
            'headers': result.get('headers', {})
        })
        
        return tool_name, selected_tool

# Global tool selector instance
tool_selector = ToolSelector()
