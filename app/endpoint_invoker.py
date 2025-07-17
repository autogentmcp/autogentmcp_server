"""
HTTP endpoint invocation with authentication support.
"""
import httpx
from typing import Dict, Any, Optional
from app.auth_handler import auth_handler

class EndpointInvoker:
    """Handle HTTP requests to external APIs with authentication."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def invoke_endpoint(
        self, 
        app_key: str,
        agent_info: Dict[str, Any],
        tool_info: Dict[str, Any],
        resolved_endpoint: str,
        method: str = "GET",
        query_params: Optional[Dict[str, Any]] = None,
        body_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Invoke an external API endpoint with authentication.
        
        Args:
            app_key: Application key from registry
            agent_info: Agent configuration from registry
            tool_info: Tool configuration from registry
            resolved_endpoint: Full URL to invoke
            method: HTTP method
            query_params: Query parameters
            body_params: Request body parameters
            headers: Additional headers
            
        Returns:
            Response data (JSON or text)
        """
        try:
            # Start with provided headers
            request_headers = headers.copy() if headers else {}
            
            # Add authentication headers using new registry-based auth
            auth_headers = auth_handler.get_auth_headers_for_agent(app_key)
            request_headers.update(auth_headers)
            
            # Add authentication query parameters (if any from legacy security info)
            security_info = agent_info.get('security', {})
            if security_info:
                # Fallback to legacy auth if needed
                legacy_auth_headers = auth_handler.get_auth_headers(app_key, security_info)
                for key, value in legacy_auth_headers.items():
                    if key not in request_headers:  # Don't override registry auth
                        request_headers[key] = value
            
            # Add default headers
            if 'User-Agent' not in request_headers:
                request_headers['User-Agent'] = 'MCP-Registry-Server/1.0'
            
            print(f"[EndpointInvoker] Invoking {method} {resolved_endpoint}")
            print(f"[EndpointInvoker] Headers: {request_headers}")
            print(f"[EndpointInvoker] Query params: {query_params}")
            print(f"[EndpointInvoker] Body params: {body_params}")
            
            # Make the request
            response = self._make_request(
                method=method,
                url=resolved_endpoint,
                headers=request_headers,
                query_params=query_params,
                body_params=body_params
            )
            
            # Process response
            result = self._process_response(response)
            print(f"[EndpointInvoker] Response: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Error calling endpoint: {e}"
            print(f"[EndpointInvoker] {error_msg}")
            return error_msg
    
    def _make_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        query_params: Optional[Dict[str, Any]] = None,
        body_params: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        """Make the actual HTTP request."""
        method = method.upper()
        
        if method == "GET":
            response = httpx.get(url, params=query_params, headers=headers, timeout=self.timeout)
        elif method == "POST":
            response = httpx.post(url, params=query_params, json=body_params, headers=headers, timeout=self.timeout)
        elif method == "PUT":
            response = httpx.put(url, params=query_params, json=body_params, headers=headers, timeout=self.timeout)
        elif method == "DELETE":
            response = httpx.delete(url, params=query_params, headers=headers, timeout=self.timeout)
        elif method == "PATCH":
            response = httpx.patch(url, params=query_params, json=body_params, headers=headers, timeout=self.timeout)
        else:
            # Generic request for other methods
            response = httpx.request(method, url, params=query_params, json=body_params, headers=headers, timeout=self.timeout)
        
        return response
    
    def _process_response(self, response: httpx.Response) -> Any:
        """Process the HTTP response."""
        # Check if response is successful
        if response.status_code >= 400:
            return f"HTTP {response.status_code}: {response.text}"
        
        # Try to parse as JSON
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            try:
                return response.json()
            except Exception:
                return response.text
        else:
            return response.text

# Global endpoint invoker instance
endpoint_invoker = EndpointInvoker()
