"""
Authentication handler for different API authorization types.
"""
import os
import base64
from typing import Dict, Optional, Any
import json

from app.registry import get_agent_auth_headers
from app.auth_header_generator import auth_header_generator

class AuthHandler:
    """Handle different types of API authentication."""
    
    def __init__(self):
        self.auth_cache = {}
        self.load_auth_config()
    
    def load_auth_config(self):
        """Load authentication configuration from environment and config files."""
        # Load from environment variables
        self.auth_cache.update({
            'default_api_key': os.getenv('MCP_DEFAULT_API_KEY'),
            'default_bearer_token': os.getenv('MCP_DEFAULT_BEARER_TOKEN'),
            'default_basic_user': os.getenv('MCP_DEFAULT_BASIC_USER'),
            'default_basic_pass': os.getenv('MCP_DEFAULT_BASIC_PASS'),
        })
        
        # Try to load from config file
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'auth_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.auth_cache.update(config)
        except Exception as e:
            print(f"[AuthHandler] Could not load auth config: {e}")
    
    def get_auth_headers_for_agent(self, app_key: str) -> Dict[str, str]:
        """
        Get authentication headers for a specific agent using registry and vault.
        
        Args:
            app_key: Application key from registry
            
        Returns:
            Dictionary of headers to add to the request
        """
        try:
            # Use registry-based authentication
            headers = get_agent_auth_headers(app_key)
            if headers:
                return headers
            
            # Fallback to cached auth if registry auth fails
            return self._get_fallback_auth_headers(app_key)
        except Exception as e:
            print(f"[AuthHandler] Error getting auth headers for {app_key}: {e}")
            return self._get_fallback_auth_headers(app_key)
    
    def _get_fallback_auth_headers(self, app_key: str) -> Dict[str, str]:
        """Fallback authentication using cached credentials."""
        headers = {}
        
        # Try agent-specific API key first
        api_key = (
            self.auth_cache.get(f'{app_key}_api_key') or
            self.auth_cache.get('default_api_key')
        )
        
        if api_key:
            headers['X-API-Key'] = api_key
        
        # Try bearer token
        bearer_token = (
            self.auth_cache.get(f'{app_key}_bearer_token') or
            self.auth_cache.get('default_bearer_token')
        )
        
        if bearer_token:
            headers['Authorization'] = f'Bearer {bearer_token}'
        
        return headers
    
    def set_auth_credential(self, key: str, value: str):
        """Set an authentication credential in the cache."""
        self.auth_cache[key] = value
    
    def generate_auth_headers(self, application_id: str, authentication_method: str,
                             endpoint_url: str = None, request_method: str = 'GET',
                             request_body: str = None) -> Dict[str, str]:
        """
        Generate authentication headers using the enhanced authentication header generator.
        
        This method supports all 11 authentication types with proper credential processing:
        - api_key, bearer_token, basic_auth, oauth2, jwt
        - azure_subscription, azure_apim, aws_iam, gcp_service_account
        - signature_auth, custom
        
        Args:
            application_id: The application ID for credential lookup
            authentication_method: Type of authentication
            endpoint_url: Target endpoint URL (required for some auth types)
            request_method: HTTP method (GET, POST, etc.)
            request_body: Request body for signature-based auth
            
        Returns:
            Dictionary of headers to add to the request
        """
        try:
            return auth_header_generator.generate_headers(
                application_id=application_id,
                authentication_method=authentication_method,
                endpoint_url=endpoint_url,
                request_method=request_method,
                request_body=request_body
            )
        except Exception as e:
            print(f"[AuthHandler] Error generating auth headers: {e}")
            return {}
    
    def generate_auth_headers_with_vault_key(self, vault_key: str, authentication_method: str,
                                           endpoint_url: str = None, request_method: str = 'GET',
                                           request_body: str = None) -> Dict[str, str]:
        """
        Generate authentication headers using vault key directly.
        
        This method supports all 11 authentication types with proper credential processing:
        - api_key, bearer_token, basic_auth, oauth2, jwt
        - azure_subscription, azure_apim, aws_iam, gcp_service_account
        - signature_auth, custom
        
        Args:
            vault_key: The vault key to retrieve credentials from
            authentication_method: Type of authentication
            endpoint_url: Target endpoint URL (required for some auth types)
            request_method: HTTP method (GET, POST, etc.)
            request_body: Request body for signature-based auth
            
        Returns:
            Dictionary of headers to add to the request
        """
        try:
            return auth_header_generator.generate_headers(
                vault_key=vault_key,
                authentication_method=authentication_method,
                endpoint_url=endpoint_url,
                request_method=request_method,
                request_body=request_body
            )
        except Exception as e:
            print(f"[AuthHandler] Error generating auth headers: {e}")
            return {}
    
    def validate_auth_credentials(self, application_id: str, authentication_method: str) -> bool:
        """
        Validate that required credentials are available for the authentication method.
        
        Args:
            application_id: The application ID
            authentication_method: Type of authentication
            
        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            return auth_header_generator.validate_credentials(application_id, authentication_method)
        except Exception as e:
            print(f"[AuthHandler] Error validating credentials: {e}")
            return False
    
    def validate_auth_credentials_with_vault_key(self, vault_key: str, authentication_method: str) -> bool:
        """
        Validate that required credentials are available for the authentication method using vault key.
        
        Args:
            vault_key: The vault key to retrieve credentials from
            authentication_method: Type of authentication
            
        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            return auth_header_generator.validate_credentials(vault_key, authentication_method)
        except Exception as e:
            print(f"[AuthHandler] Error validating credentials: {e}")
            return False
    
    def get_auth_headers(self, agent_name: str, security_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Legacy method for backwards compatibility.
        Generate authentication headers based on security configuration.
        
        Args:
            agent_name: Name of the agent/service
            security_info: Security configuration from registry
            
        Returns:
            Dictionary of headers to add to the request
        """
        headers = {}
        
        if not security_info:
            return headers
        
        auth_type = security_info.get('type', '').lower()
        
        # Standard auth type handlers
        if auth_type == 'apikey':
            headers.update(self._handle_api_key_auth(agent_name, security_info))
        elif auth_type == 'bearer':
            headers.update(self._handle_bearer_auth(agent_name, security_info))
        elif auth_type == 'basic':
            headers.update(self._handle_basic_auth(agent_name, security_info))
        elif auth_type == 'oauth2':
            headers.update(self._handle_oauth2_auth(agent_name, security_info))
        elif auth_type == 'azure_apim':
            headers.update(self._handle_azure_apim_auth(agent_name, security_info))
        elif auth_type == 'aws_sigv4':
            headers.update(self._handle_aws_sigv4_auth(agent_name, security_info))
        elif auth_type == 'gcp_service_account':
            headers.update(self._handle_gcp_service_account_auth(agent_name, security_info))
        elif auth_type == 'custom_signature':
            headers.update(self._handle_custom_signature_auth(agent_name, security_info))
        else:
            print(f"[AuthHandler] Unsupported auth type: {auth_type}")
        
        # Always process custom_headers if present
        if 'custom_headers' in security_info:
            headers.update(self._handle_custom_headers(agent_name, security_info))
        
        return headers
    
    def _handle_api_key_auth(self, agent_name: str, security_info: Dict[str, Any]) -> Dict[str, str]:
        """Handle API key authentication."""
        headers = {}
        
        # Get API key from various sources
        api_key = (
            self.auth_cache.get(f'{agent_name}_api_key') or
            self.auth_cache.get('default_api_key') or
            security_info.get('api_key')
        )
        
        if not api_key:
            print(f"[AuthHandler] No API key found for {agent_name}")
            return headers
        
        # Determine header name and format
        header_name = security_info.get('header_name', 'X-API-Key')
        key_format = security_info.get('format', 'direct')  # direct, Bearer {key}, etc.
        
        if key_format == 'bearer':
            headers['Authorization'] = f'Bearer {api_key}'
        elif key_format == 'direct':
            headers[header_name] = api_key
        else:
            headers[header_name] = key_format.format(key=api_key)
        
        return headers
    
    def _handle_bearer_auth(self, agent_name: str, security_info: Dict[str, Any]) -> Dict[str, str]:
        """Handle Bearer token authentication."""
        headers = {}
        
        token = (
            self.auth_cache.get(f'{agent_name}_bearer_token') or
            self.auth_cache.get('default_bearer_token') or
            security_info.get('token')
        )
        
        if token:
            headers['Authorization'] = f'Bearer {token}'
        else:
            print(f"[AuthHandler] No bearer token found for {agent_name}")
        
        return headers
    
    def _handle_basic_auth(self, agent_name: str, security_info: Dict[str, Any]) -> Dict[str, str]:
        """Handle Basic authentication."""
        headers = {}
        
        username = (
            self.auth_cache.get(f'{agent_name}_basic_user') or
            self.auth_cache.get('default_basic_user') or
            security_info.get('username')
        )
        
        password = (
            self.auth_cache.get(f'{agent_name}_basic_pass') or
            self.auth_cache.get('default_basic_pass') or
            security_info.get('password')
        )
        
        if username and password:
            credentials = base64.b64encode(f'{username}:{password}'.encode()).decode()
            headers['Authorization'] = f'Basic {credentials}'
        else:
            print(f"[AuthHandler] No basic auth credentials found for {agent_name}")
        
        return headers
    
    def _handle_oauth2_auth(self, agent_name: str, security_info: Dict[str, Any]) -> Dict[str, str]:
        """Handle OAuth2 authentication."""
        headers = {}
        
        # For OAuth2, we'll use a stored access token
        access_token = (
            self.auth_cache.get(f'{agent_name}_oauth2_token') or
            self.auth_cache.get('default_oauth2_token') or
            security_info.get('access_token')
        )
        
        if access_token:
            headers['Authorization'] = f'Bearer {access_token}'
        else:
            print(f"[AuthHandler] No OAuth2 token found for {agent_name}")
        
        return headers
    
    def _handle_azure_apim_auth(self, agent_name: str, security_info: Dict[str, Any]) -> Dict[str, str]:
        """Handle Azure APIM subscription key authentication."""
        headers = {}
        
        sub_key = (
            self.auth_cache.get(f'{agent_name}_azure_apim_key') or
            self.auth_cache.get('default_azure_apim_key') or
            security_info.get('subscription_key')
        )
        
        header_name = security_info.get('header_name', 'Ocp-Apim-Subscription-Key')
        
        if sub_key:
            headers[header_name] = sub_key
        else:
            print(f"[AuthHandler] No Azure APIM subscription key found for {agent_name}")
        
        return headers
    
    def _handle_aws_sigv4_auth(self, agent_name: str, security_info: Dict[str, Any]) -> Dict[str, str]:
        """Handle AWS SigV4 authentication (stub)."""
        # TODO: Implement AWS SigV4 signing (use 'requests-aws4auth' or 'botocore')
        print(f"[AuthHandler] AWS SigV4 auth not implemented. Returning empty headers.")
        return {}
    
    def _handle_gcp_service_account_auth(self, agent_name: str, security_info: Dict[str, Any]) -> Dict[str, str]:
        """Handle GCP Service Account authentication (stub)."""
        # TODO: Implement GCP service account JWT or OAuth2 token
        print(f"[AuthHandler] GCP Service Account auth not implemented. Returning empty headers.")
        return {}
    
    def _handle_custom_signature_auth(self, agent_name: str, security_info: Dict[str, Any]) -> Dict[str, str]:
        """Handle custom signature authentication (stub)."""
        # TODO: Implement custom signature logic as needed
        print(f"[AuthHandler] Custom signature auth not implemented. Returning empty headers.")
        return {}
    
    def set_auth_credential(self, key: str, value: str):
        """Set an authentication credential at runtime."""
        self.auth_cache[key] = value
        print(f"[AuthHandler] Updated credential: {key}")
    
    def get_auth_query_params(self, agent_name: str, security_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate query parameters for authentication (e.g., API key in query string).
        
        Args:
            agent_name: Name of the agent/service
            security_info: Security configuration from registry
            
        Returns:
            Dictionary of query parameters to add to the request
        """
        params = {}
        
        if not security_info:
            return params
        
        auth_type = security_info.get('type', '').lower()
        location = security_info.get('location', 'header').lower()
        
        if auth_type == 'apikey' and location == 'query':
            api_key = (
                self.auth_cache.get(f'{agent_name}_api_key') or
                self.auth_cache.get('default_api_key') or
                security_info.get('api_key')
            )
            
            if api_key:
                param_name = security_info.get('param_name', 'api_key')
                params[param_name] = api_key
        
        return params
    
    def _handle_custom_headers(self, agent_name: str, security_info: Dict[str, Any]) -> Dict[str, str]:
        """Inject custom headers from security_info['custom_headers'] (resolve secrets if needed)."""
        headers = {}
        custom_headers = security_info.get('custom_headers', {})
        for key, value in custom_headers.items():
            if isinstance(value, str) and value.startswith('use_secret:'):
                secret_key = value[len('use_secret:')]
                secret_val = self.auth_cache.get(secret_key) or os.getenv(secret_key)
                if secret_val:
                    headers[key] = secret_val
                else:
                    print(f"[AuthHandler] Secret {secret_key} not found for header {key}")
            else:
                headers[key] = value
        return headers

# Global auth handler instance
auth_handler = AuthHandler()
