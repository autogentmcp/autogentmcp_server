"""
Application agent execution with real API calls and vault-based authentication
"""

import json
from typing import Dict, List, Any, Optional
from ..models import AgentResult
from app.registry import get_enhanced_agent_details_for_llm
from app.multimode_llm_client import get_global_llm_client, TaskType
from app.auth_handler import auth_handler
from app.endpoint_invoker import endpoint_invoker

class ApplicationAgentExecutor:
    """Executes application agents with custom prompt support"""
    
    def __init__(self):
        self.llm_client = get_global_llm_client()
        self.auth_handler = auth_handler
        self.endpoint_invoker = endpoint_invoker
    
    async def execute_application_agent(self, agent_id: str, query: str) -> AgentResult:
        """Execute application agent with real API calls"""
        
        print(f"[ApplicationAgentExecutor] Executing application agent {agent_id}")
        
        try:
            # Get full agent details including endpoints
            agent_details = get_enhanced_agent_details_for_llm(agent_id)
            agent_name = agent_details.get("name", agent_id)
            
            # Step 1: Use LLM to determine the API call parameters
            api_call_params = await self._determine_api_call(agent_details, query)
            
            if not api_call_params.get("success"):
                return AgentResult(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    success=False,
                    error=api_call_params.get("error", "Failed to determine API call parameters"),
                    metadata={"stage": "api_planning", "custom_prompt_used": bool(agent_details.get("environment", {}).get("customPrompt"))}
                )
            
            # Step 2: Make the actual API call
            api_response = await self._make_api_call(api_call_params)
            
            if not api_response.get("success"):
                return AgentResult(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    success=False,
                    error=api_response.get("error", "API call failed"),
                    metadata={
                        "stage": "api_execution",
                        "api_call": api_call_params,
                        "custom_prompt_used": bool(agent_details.get("environment", {}).get("customPrompt"))
                    }
                )
            
            # Return raw API response data, let response handler do final LLM processing
            return AgentResult(
                agent_id=agent_id,
                agent_name=agent_name,
                success=True,
                data=api_response.get("data"),  # Raw API response data
                metadata={
                    "execution_type": "application",
                    "api_call": api_call_params,
                    "api_response_status": api_response.get("status_code"),
                    "custom_prompt_used": bool(agent_details.get("environment", {}).get("customPrompt")),
                    "stages_completed": ["api_planning", "api_execution"]
                }
            )
                
        except Exception as e:
            error_msg = f"Application agent execution error: {str(e)}"
            print(f"[ApplicationAgentExecutor] {error_msg}")
            
            return AgentResult(
                agent_id=agent_id,
                agent_name=agent_details.get("name", agent_id) if 'agent_details' in locals() else agent_id,
                success=False,
                error=error_msg,
                metadata={
                    "execution_type": "application",
                    "exception": True
                }
            )
    
    async def _determine_api_call(self, agent_details: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Use LLM to determine the correct API call parameters"""
        
        print(f"[ApplicationAgentExecutor] Determining API call parameters")
        
        agent_name = agent_details.get("name", "Unknown")
        description = agent_details.get("description", "")
        endpoints = agent_details.get("endpoints", [])
        environment = agent_details.get("environment", {})
        custom_prompt = environment.get("customPrompt", "")
        
        # Build comprehensive prompt with all agent details
        prompt = f"""
You are an API integration specialist. Based on the user query and available API endpoints, determine the exact API call to make.

Agent: {agent_name}
Description: {description}
User Query: "{query}"

Available Endpoints:
{json.dumps(endpoints, indent=2)}

Environment Details:
{json.dumps(environment, indent=2)}
"""
        
        if custom_prompt:
            prompt += f"""

Custom Agent Instructions:
{custom_prompt}
"""
        
        prompt += """

INSTRUCTIONS:
1. Analyze the user query and match it to the most appropriate endpoint
2. Determine the HTTP method, URL, headers, and payload needed
3. Use the authentication details from the environment if required
4. Consider any custom instructions provided

Respond with this EXACT JSON structure:
{
    "success": true/false,
    "endpoint_selected": "name_of_selected_endpoint",
    "method": "GET|POST|PUT|DELETE|PATCH",
    "url": "full_url_to_call",
    "headers": {
        "Content-Type": "application/json",
        "Authorization": "Bearer token_if_needed"
    },
    "payload": {
        
    },
    "reasoning": "Why this endpoint and parameters were chosen",
    "error": "error_message_if_cannot_determine"
}

RULES:
- Return ONLY valid JSON without any comments or extra text
- Use exact field names as shown in the structure
- Do not include JavaScript-style comments (//) in the JSON
- Set success=false if you cannot determine the appropriate endpoint
- Include reasoning for your endpoint selection
- Only include necessary headers
- Leave payload empty for GET requests
- If no suitable endpoint found, set success=false and explain in error
- Include all necessary authentication headers
- Build proper payload based on endpoint requirements
- Use environment variables for secrets/tokens
- Follow the endpoint specification exactly
"""
        
        response = self.llm_client.invoke_with_json_response(prompt, task_type=TaskType.TOOL_SELECTION)
        
        if not response:
            return {"success": False, "error": "LLM failed to determine API call parameters"}
        
        return response
    
    async def _make_api_call(self, api_params: Dict[str, Any]) -> Dict[str, Any]:
        """Make the actual HTTP API call using endpoint invoker with vault authentication"""
        
        print(f"[ApplicationAgentExecutor] Making API call to {api_params.get('url')}")
        
        try:
            method = api_params.get("method", "GET").upper()
            url = api_params.get("url")
            headers = api_params.get("headers", {})
            payload = api_params.get("payload")
            
            if not url:
                return {"success": False, "error": "No URL provided for API call"}
            
            # Generate authentication headers using vault (with error handling)
            try:
                auth_headers = self.auth_handler.generate_auth_headers_with_vault_key(vault_key="default_key")
                if auth_headers:
                    headers.update(auth_headers)
            except Exception as auth_error:
                print(f"[ApplicationAgentExecutor] Auth header generation failed: {str(auth_error)}")
                # Continue without auth headers
            
            # Prepare parameters for endpoint invoker
            agent_info = {"security": {}}
            tool_info = {
                "query_params": {},
                "body_params": payload if payload else {},
                "headers": {}
            }
            
            print(f"[ApplicationAgentExecutor] Calling endpoint invoker for {method} {url}")
            
            # Make the API call using endpoint invoker (synchronous)
            response = self.endpoint_invoker.invoke_registry_endpoint(
                app_key="default_app",
                agent_info=agent_info,
                tool_info=tool_info,
                resolved_endpoint=url,
                method=method,
                query_params={},
                body_params=payload,
                headers=headers
            )
            
            print(f"[ApplicationAgentExecutor] API call completed, response type: {type(response)}")
            
            if response is None:
                return {"success": False, "error": "Endpoint invoker returned None"}
            
            # Process the response from endpoint invoker
            return self._process_endpoint_response(response)
                
        except Exception as e:
            error_msg = f"API call failed: {str(e)}"
            print(f"[ApplicationAgentExecutor] {error_msg}")
            return {"success": False, "error": error_msg}
    
    def _process_endpoint_response(self, response: Any) -> Dict[str, Any]:
        """Process the response from the endpoint invoker"""
        
        try:
            # The existing invoke_registry_endpoint returns raw response data (JSON, text, or error string)
            # Check if it's an error string
            if isinstance(response, str) and response.startswith("HTTP "):
                # Error response format: "HTTP 404: Not Found"
                status_code = int(response.split(":")[0].split(" ")[1])
                return {
                    "success": False,
                    "status_code": status_code,
                    "error": response,
                    "data": {},
                    "headers": {}
                }
            elif isinstance(response, str) and "Error calling endpoint:" in response:
                # Generic error response
                return {
                    "success": False,
                    "status_code": 500,
                    "error": response,
                    "data": {},
                    "headers": {}
                }
            else:
                # Successful response - could be JSON dict or text
                return {
                    "success": True,
                    "status_code": 200,  # Assume 200 for successful responses
                    "data": response,
                    "headers": {}
                }
                
        except Exception as e:
            return {"success": False, "error": f"Failed to process endpoint response: {str(e)}"}
