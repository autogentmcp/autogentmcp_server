"""
Enhanced authentication header generator for all supported authentication types.
This module generates appropriate headers based on application authentication method and vault-stored credentials.
"""

import base64
import binascii
import json
import logging
import time
from typing import Dict, Optional, Any, List
import hashlib
import hmac
import urllib.parse
from datetime import datetime, timezone

from app.utils.credential_processor import CredentialProcessor
from .vault_manager import vault_manager

logger = logging.getLogger(__name__)

class AuthenticationHeaderGenerator:
    """Generates authentication headers for all supported authentication types."""
    
    # Supported authentication methods
    SUPPORTED_AUTH_METHODS = {
        'api_key', 'bearer_token', 'basic_auth', 'oauth2', 'jwt',
        'azure_subscription', 'azure_apim', 'aws_iam', 'gcp_service_account',
        'signature_auth', 'custom'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _process_custom_headers(self, credentials: Dict[str, Any]) -> Dict[str, str]:
        """
        Process custom headers from credentials.
        
        Custom headers can be:
        1. Direct dictionary in 'customHeaders' field
        2. Base64 encoded JSON array in 'customHeaders' field
        3. Base64 encoded JSON object in 'customHeaders' field
        
        Args:
            credentials: Processed credentials dictionary
            
        Returns:
            Dictionary of custom headers
        """
        custom_headers = {}
        
        # Get custom headers from credentials
        custom_headers_raw = credentials.get('customHeaders')
        if not custom_headers_raw:
            return custom_headers
        
        try:
            # If it's already a dict, use it directly
            if isinstance(custom_headers_raw, dict):
                custom_headers.update(custom_headers_raw)
            elif isinstance(custom_headers_raw, str):
                # Try to decode as Base64 first
                try:
                    decoded = base64.b64decode(custom_headers_raw).decode('utf-8')
                    parsed = json.loads(decoded)
                    
                    if isinstance(parsed, list):
                        # Handle array format: [{"name": "Header-Name", "value": "Header-Value"}]
                        for header_item in parsed:
                            if isinstance(header_item, dict) and 'name' in header_item and 'value' in header_item:
                                custom_headers[header_item['name']] = header_item['value']
                    elif isinstance(parsed, dict):
                        # Handle object format: {"Header-Name": "Header-Value"}
                        custom_headers.update(parsed)
                except (base64.binascii.Error, json.JSONDecodeError):
                    # If Base64 decoding fails, try direct JSON parsing
                    try:
                        parsed = json.loads(custom_headers_raw)
                        if isinstance(parsed, dict):
                            custom_headers.update(parsed)
                        elif isinstance(parsed, list):
                            for header_item in parsed:
                                if isinstance(header_item, dict) and 'name' in header_item and 'value' in header_item:
                                    custom_headers[header_item['name']] = header_item['value']
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to parse custom headers as JSON")
            
        except Exception as e:
            self.logger.error(f"Error processing custom headers: {e}")
        
        return custom_headers
    
    def generate_headers(self, vault_key: str, authentication_method: str, 
                        endpoint_url: str = None, request_method: str = 'GET',
                        request_body: str = None) -> Dict[str, str]:
        """
        Generate authentication headers based on vault key and authentication method.
        
        Args:
            vault_key: The vault key to retrieve credentials from
            authentication_method: Type of authentication (api_key, bearer_token, etc.)
            endpoint_url: Target endpoint URL (required for some auth types)
            request_method: HTTP method (GET, POST, etc.)
            request_body: Request body for signature-based auth
            
        Returns:
            Dictionary of headers to add to the request
        """
        if not authentication_method or authentication_method not in self.SUPPORTED_AUTH_METHODS:
            self.logger.warning(f"Unsupported authentication method: {authentication_method}")
            return {}
        
        if not vault_key:
            self.logger.warning("No vault key provided")
            return {}
        
        # Get credentials from vault using provided key
        credentials = vault_manager.get_auth_credentials(vault_key)
        
        if not credentials:
            self.logger.warning(f"No credentials found for vault key: {vault_key}")
            return {}
        
        # Process credentials (Base64 decode sensitive fields)
        processed_credentials = CredentialProcessor.process_credentials_from_storage(credentials)
        
        # Generate headers based on authentication method
        method_handlers = {
            'api_key': self._generate_api_key_headers,
            'bearer_token': self._generate_bearer_token_headers,
            'basic_auth': self._generate_basic_auth_headers,
            'oauth2': self._generate_oauth2_headers,
            'jwt': self._generate_jwt_headers,
            'azure_subscription': self._generate_azure_subscription_headers,
            'azure_apim': self._generate_azure_apim_headers,
            'aws_iam': self._generate_aws_iam_headers,
            'gcp_service_account': self._generate_gcp_service_account_headers,
            'signature_auth': self._generate_signature_auth_headers,
            'custom': self._generate_custom_headers
        }
        
        handler = method_handlers.get(authentication_method)
        if handler:
            return handler(processed_credentials, endpoint_url, request_method, request_body)
        
        return {}
    
    def _generate_api_key_headers(self, credentials: Dict[str, Any], 
                                 endpoint_url: str = None, request_method: str = 'GET',
                                 request_body: str = None) -> Dict[str, str]:
        """Generate API key authentication headers."""
        headers = {}
        
        api_key = credentials.get('apiKey') or credentials.get('api_key')
        if not api_key:
            self.logger.warning("No API key found in credentials")
            return headers
        
        # Get header configuration
        header_name = credentials.get('headerName', 'X-API-Key')
        key_format = credentials.get('format', 'direct')  # direct, Bearer {key}, etc.
        
        if key_format == 'Bearer':
            headers['Authorization'] = f'Bearer {api_key}'
        elif key_format == 'direct':
            headers[header_name] = api_key
        else:
            # Custom format with placeholder
            headers[header_name] = key_format.replace('{key}', api_key)
        
        # Add custom headers if present
        custom_headers = self._process_custom_headers(credentials)
        headers.update(custom_headers)
        
        return headers
    
    def _generate_bearer_token_headers(self, credentials: Dict[str, Any], 
                                      endpoint_url: str = None, request_method: str = 'GET',
                                      request_body: str = None) -> Dict[str, str]:
        """Generate Bearer token authentication headers."""
        headers = {}
        
        token = credentials.get('token') or credentials.get('bearerToken') or credentials.get('access_token')
        if not token:
            self.logger.warning("No bearer token found in credentials")
            return headers
        
        headers['Authorization'] = f'Bearer {token}'
        
        # Add additional headers if specified (legacy support)
        if 'additionalHeaders' in credentials:
            headers.update(credentials['additionalHeaders'])
        
        # Add custom headers if present
        custom_headers = self._process_custom_headers(credentials)
        headers.update(custom_headers)
        
        return headers
    
    def _generate_basic_auth_headers(self, credentials: Dict[str, Any], 
                                    endpoint_url: str = None, request_method: str = 'GET',
                                    request_body: str = None) -> Dict[str, str]:
        """Generate Basic authentication headers."""
        headers = {}
        
        username = credentials.get('username') or credentials.get('user')
        password = credentials.get('password') or credentials.get('pass')
        
        if not username or not password:
            self.logger.warning("Username or password missing for basic auth")
            return headers
        
        # Encode credentials
        auth_string = f"{username}:{password}"
        auth_bytes = auth_string.encode('utf-8')
        auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
        
        headers['Authorization'] = f'Basic {auth_b64}'
        
        # Add custom headers if present
        custom_headers = self._process_custom_headers(credentials)
        headers.update(custom_headers)
        
        return headers
    
    def _generate_oauth2_headers(self, credentials: Dict[str, Any], 
                                endpoint_url: str = None, request_method: str = 'GET',
                                request_body: str = None) -> Dict[str, str]:
        """Generate OAuth2 authentication headers."""
        headers = {}
        
        access_token = credentials.get('accessToken') or credentials.get('access_token')
        if not access_token:
            self.logger.warning("No access token found for OAuth2")
            return headers
        
        token_type = credentials.get('tokenType', 'Bearer')
        headers['Authorization'] = f'{token_type} {access_token}'
        
        # Add additional OAuth2 headers if present
        if 'scope' in credentials:
            headers['X-OAuth-Scope'] = credentials['scope']
        
        # Add custom headers if present
        custom_headers = self._process_custom_headers(credentials)
        headers.update(custom_headers)
        
        return headers
    
    def _generate_jwt_headers(self, credentials: Dict[str, Any], 
                             endpoint_url: str = None, request_method: str = 'GET',
                             request_body: str = None) -> Dict[str, str]:
        """Generate JWT authentication headers."""
        headers = {}
        
        jwt_token = credentials.get('jwtToken') or credentials.get('jwt') or credentials.get('token')
        if not jwt_token:
            self.logger.warning("No JWT token found in credentials")
            return headers
        
        # JWT tokens are typically sent as Bearer tokens
        headers['Authorization'] = f'Bearer {jwt_token}'
        
        # Add custom headers if present
        custom_headers = self._process_custom_headers(credentials)
        headers.update(custom_headers)
        
        return headers
    
    def _generate_azure_subscription_headers(self, credentials: Dict[str, Any], 
                                           endpoint_url: str = None, request_method: str = 'GET',
                                           request_body: str = None) -> Dict[str, str]:
        """Generate Azure Subscription authentication headers."""
        headers = {}
        
        subscription_key = credentials.get('subscriptionKey') or credentials.get('subscription_key')
        if not subscription_key:
            self.logger.warning("No subscription key found for Azure subscription auth")
            return headers
        
        # Azure typically uses Ocp-Apim-Subscription-Key header
        headers['Ocp-Apim-Subscription-Key'] = subscription_key
        
        # Add additional Azure headers if present
        if 'tenantId' in credentials:
            headers['x-ms-tenant-id'] = credentials['tenantId']
        
        # Add custom headers if present
        custom_headers = self._process_custom_headers(credentials)
        headers.update(custom_headers)
        
        return headers
    
    def _generate_azure_apim_headers(self, credentials: Dict[str, Any], 
                                    endpoint_url: str = None, request_method: str = 'GET',
                                    request_body: str = None) -> Dict[str, str]:
        """Generate Azure API Management authentication headers."""
        headers = {}
        
        subscription_key = credentials.get('subscriptionKey') or credentials.get('subscription_key')
        if subscription_key:
            headers['Ocp-Apim-Subscription-Key'] = subscription_key
        
        # Azure APIM can also use access tokens
        access_token = credentials.get('accessToken') or credentials.get('access_token')
        if access_token:
            headers['Authorization'] = f'Bearer {access_token}'
        
        # Add trace header if specified
        if 'traceEnabled' in credentials and credentials['traceEnabled']:
            headers['Ocp-Apim-Trace'] = 'true'
        
        # Add custom headers if present
        custom_headers = self._process_custom_headers(credentials)
        headers.update(custom_headers)
        
        return headers
    
    def _generate_aws_iam_headers(self, credentials: Dict[str, Any], 
                                 endpoint_url: str = None, request_method: str = 'GET',
                                 request_body: str = None) -> Dict[str, str]:
        """Generate AWS IAM Signature V4 authentication headers."""
        headers = {}
        
        access_key = credentials.get('accessKey') or credentials.get('access_key_id')
        secret_key = credentials.get('secretKey') or credentials.get('secret_access_key')
        
        if not access_key or not secret_key:
            self.logger.warning("AWS access key or secret key missing")
            return headers
        
        # AWS Signature V4 is complex, this is a simplified version
        # In production, you'd use boto3 or implement full SigV4
        region = credentials.get('region', 'us-east-1')
        service = credentials.get('service', 'execute-api')
        
        # Create basic AWS headers
        headers['Authorization'] = f'AWS4-HMAC-SHA256 Credential={access_key}'
        headers['x-amz-date'] = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        
        # Add session token if present (for temporary credentials)
        if 'sessionToken' in credentials:
            headers['x-amz-security-token'] = credentials['sessionToken']
        
        # Add custom headers if present
        custom_headers = self._process_custom_headers(credentials)
        headers.update(custom_headers)
        
        return headers
    
    def _generate_gcp_service_account_headers(self, credentials: Dict[str, Any], 
                                            endpoint_url: str = None, request_method: str = 'GET',
                                            request_body: str = None) -> Dict[str, str]:
        """Generate GCP Service Account authentication headers."""
        headers = {}
        
        # GCP typically uses service account key or access token
        access_token = credentials.get('accessToken') or credentials.get('access_token')
        if access_token:
            headers['Authorization'] = f'Bearer {access_token}'
        
        # Add GCP-specific headers
        if 'projectId' in credentials:
            headers['x-goog-user-project'] = credentials['projectId']
        
        # Add custom headers if present
        custom_headers = self._process_custom_headers(credentials)
        headers.update(custom_headers)
        
        return headers
    
    def _generate_signature_auth_headers(self, credentials: Dict[str, Any], 
                                        endpoint_url: str = None, request_method: str = 'GET',
                                        request_body: str = None) -> Dict[str, str]:
        """Generate signature-based authentication headers."""
        headers = {}
        
        secret_key = credentials.get('secretKey') or credentials.get('secret')
        if not secret_key:
            self.logger.warning("No secret key found for signature auth")
            return headers
        
        # Create signature components
        timestamp = str(int(time.time()))
        nonce = credentials.get('nonce', timestamp)
        
        # Create signature string
        signature_string = f"{request_method.upper()}\n{endpoint_url}\n{timestamp}\n{nonce}"
        if request_body:
            signature_string += f"\n{request_body}"
        
        # Generate HMAC signature
        signature = hmac.new(
            secret_key.encode('utf-8'),
            signature_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Add signature headers
        headers['X-Timestamp'] = timestamp
        headers['X-Nonce'] = nonce
        headers['X-Signature'] = signature
        
        # Add key ID if present
        if 'keyId' in credentials:
            headers['X-Key-Id'] = credentials['keyId']
        
        # Add custom headers if present
        custom_headers = self._process_custom_headers(credentials)
        headers.update(custom_headers)
        
        return headers
    
    def _generate_custom_headers(self, credentials: Dict[str, Any], 
                                endpoint_url: str = None, request_method: str = 'GET',
                                request_body: str = None) -> Dict[str, str]:
        """Generate custom authentication headers."""
        headers = {}
        
        # Process custom headers from credentials (supports Base64 encoding)
        custom_headers = self._process_custom_headers(credentials)
        headers.update(custom_headers)
        
        # Process legacy custom headers format
        legacy_custom_headers = credentials.get('legacyCustomHeaders', {})
        if isinstance(legacy_custom_headers, dict):
            headers.update(legacy_custom_headers)
        
        # Process header templates with credential substitution
        header_templates = credentials.get('headerTemplates', {})
        if isinstance(header_templates, dict):
            for header_name, template in header_templates.items():
                # Replace placeholders with credential values
                processed_value = template
                for key, value in credentials.items():
                    if isinstance(value, str):
                        processed_value = processed_value.replace(f'{{{key}}}', value)
                headers[header_name] = processed_value
        
        return headers
    
    def validate_credentials(self, vault_key: str, authentication_method: str) -> bool:
        """
        Validate that required credentials are available for the authentication method.
        
        Args:
            vault_key: The vault key to retrieve credentials from
            authentication_method: Type of authentication
            
        Returns:
            True if credentials are valid, False otherwise
        """
        if not vault_key:
            return False
        
        credentials = vault_manager.get_auth_credentials(vault_key)
        
        if not credentials:
            return False
        
        processed_credentials = CredentialProcessor.process_credentials_from_storage(credentials)
        
        # Check required fields for each auth method
        required_fields = {
            'api_key': ['apiKey', 'api_key'],
            'bearer_token': ['token', 'bearerToken', 'access_token'],
            'basic_auth': ['username', 'password'],
            'oauth2': ['accessToken', 'access_token'],
            'jwt': ['jwtToken', 'jwt', 'token'],
            'azure_subscription': ['subscriptionKey', 'subscription_key'],
            'azure_apim': ['subscriptionKey', 'subscription_key'],
            'aws_iam': ['accessKey', 'secretKey'],
            'gcp_service_account': ['accessToken', 'access_token'],
            'signature_auth': ['secretKey', 'secret'],
            'custom': []  # Custom auth is flexible
        }
        
        required = required_fields.get(authentication_method, [])
        if not required:  # Custom auth or unknown method
            return True
        
        # Check if at least one required field is present
        return any(field in processed_credentials for field in required)

# Global instance
auth_header_generator = AuthenticationHeaderGenerator()
