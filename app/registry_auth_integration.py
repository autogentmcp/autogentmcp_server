"""
Enhanced registry integration that demonstrates how to use vault keys from registry
for authentication header generation.
"""

import logging
from typing import Dict, Any, Optional

from .registry import fetch_agents_and_tools_from_registry
from .auth_handler import auth_handler

logger = logging.getLogger(__name__)

class RegistryAuthIntegration:
    """Integration class that uses registry data to generate authentication headers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_auth_headers_for_agent(self, app_key: str, endpoint_url: str = None, 
                                  request_method: str = 'GET', 
                                  request_body: str = None) -> Dict[str, str]:
        """
        Get authentication headers for an agent using registry data and vault credentials.
        
        Args:
            app_key: The application key from registry
            endpoint_url: Target endpoint URL (required for some auth types)
            request_method: HTTP method (GET, POST, etc.)
            request_body: Request body for signature-based auth
            
        Returns:
            Dictionary of headers to add to the request
        """
        try:
            # Get agents from registry
            agents = fetch_agents_and_tools_from_registry()
            
            if app_key not in agents:
                self.logger.warning(f"Agent {app_key} not found in registry")
                return {}
            
            agent = agents[app_key]
            
            # Get security configuration
            security_config = agent.get("security_config")
            if not security_config:
                self.logger.warning(f"No security configuration found for agent {app_key}")
                return {}
            
            # Get vault key and authentication method
            vault_key = security_config.get("vaultKey")
            authentication_method = security_config.get("authenticationMethod")
            
            if not vault_key:
                self.logger.warning(f"No vault key found for agent {app_key}")
                return {}
            
            if not authentication_method:
                self.logger.warning(f"No authentication method found for agent {app_key}")
                return {}
            
            # Generate headers using vault key
            headers = auth_handler.generate_auth_headers_with_vault_key(
                vault_key=vault_key,
                authentication_method=authentication_method,
                endpoint_url=endpoint_url,
                request_method=request_method,
                request_body=request_body
            )
            
            self.logger.info(f"Generated {len(headers)} auth headers for agent {app_key}")
            return headers
            
        except Exception as e:
            self.logger.error(f"Error getting auth headers for agent {app_key}: {e}")
            return {}
    
    def validate_agent_credentials(self, app_key: str) -> bool:
        """
        Validate that an agent has valid credentials for its authentication method.
        
        Args:
            app_key: The application key from registry
            
        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            # Get agents from registry
            agents = fetch_agents_and_tools_from_registry()
            
            if app_key not in agents:
                self.logger.warning(f"Agent {app_key} not found in registry")
                return False
            
            agent = agents[app_key]
            
            # Get security configuration
            security_config = agent.get("security_config")
            if not security_config:
                self.logger.warning(f"No security configuration found for agent {app_key}")
                return False
            
            # Get vault key and authentication method
            vault_key = security_config.get("vaultKey")
            authentication_method = security_config.get("authenticationMethod")
            
            if not vault_key or not authentication_method:
                return False
            
            # Validate credentials using vault key
            is_valid = auth_handler.validate_auth_credentials_with_vault_key(
                vault_key=vault_key,
                authentication_method=authentication_method
            )
            
            self.logger.info(f"Credential validation for agent {app_key}: {'valid' if is_valid else 'invalid'}")
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Error validating credentials for agent {app_key}: {e}")
            return False
    
    def get_agent_auth_info(self, app_key: str) -> Optional[Dict[str, Any]]:
        """
        Get authentication information for an agent.
        
        Args:
            app_key: The application key from registry
            
        Returns:
            Dictionary containing auth info or None if not found
        """
        try:
            # Get agents from registry
            agents = fetch_agents_and_tools_from_registry()
            
            if app_key not in agents:
                return None
            
            agent = agents[app_key]
            security_config = agent.get("security_config")
            
            if not security_config:
                return None
            
            return {
                "app_key": app_key,
                "agent_name": agent.get("name"),
                "vault_key": security_config.get("vaultKey"),
                "authentication_method": security_config.get("authenticationMethod"),
                "has_credentials": self.validate_agent_credentials(app_key)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting auth info for agent {app_key}: {e}")
            return None
    
    def list_agents_with_auth_info(self) -> Dict[str, Dict[str, Any]]:
        """
        List all agents with their authentication information.
        
        Returns:
            Dictionary mapping app_key to authentication info
        """
        try:
            agents = fetch_agents_and_tools_from_registry()
            auth_info = {}
            
            for app_key in agents:
                info = self.get_agent_auth_info(app_key)
                if info:
                    auth_info[app_key] = info
            
            return auth_info
            
        except Exception as e:
            self.logger.error(f"Error listing agents with auth info: {e}")
            return {}

# Global instance
registry_auth_integration = RegistryAuthIntegration()
