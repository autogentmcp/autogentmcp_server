import os
import json
import base64
import time
import threading
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.utils.credential_processor import CredentialProcessor

class VaultClient(ABC):
    """Abstract base class for vault clients. Read-only interface for retrieving secrets."""
    
    @abstractmethod
    def get_secret(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a secret from the vault. Does not create or modify secrets."""
        pass

class HashicorpVaultClient(VaultClient):
    """Hashicorp Vault client implementation. Read-only - retrieves secrets only."""
    
    def __init__(self):
        self.vault_url = os.getenv("VAULT_URL")
        self.vault_token = os.getenv("VAULT_TOKEN")
        self.vault_namespace = os.getenv("VAULT_NAMESPACE", "")
        self.vault_path = os.getenv("VAULT_PATH", "")
        self.vault_mount = os.getenv("VAULT_MOUNT", "secret")
        self.vault_verify_ssl = os.getenv("VAULT_VERIFY_SSL", "true").lower() == "true"
    
    def get_secret(self, key: str) -> Optional[Dict[str, Any]]:
        """Get secret from Hashicorp Vault."""
        if not self.vault_url or not self.vault_token:
            return None
        
        try:
            import hvac
            client = hvac.Client(
                url=self.vault_url, 
                token=self.vault_token, 
                namespace=self.vault_namespace,
                verify=self.vault_verify_ssl
            )
            
            # Build the full path
            if self.vault_path:
                full_path = f"{self.vault_path}/{key}"
            else:
                full_path = key
            
            # Try KV v2 first, then v1
            try:
                secret_response = client.secrets.kv.v2.read_secret_version(
                    path=full_path,
                    mount_point=self.vault_mount
                )
                return secret_response['data']['data']
            except Exception as e:
                print(f"[HashicorpVault] KV v2 failed, trying v1: {e}")
                try:
                    secret_response = client.secrets.kv.v1.read_secret(
                        path=full_path,
                        mount_point=self.vault_mount
                    )
                    return secret_response['data']
                except Exception as e2:
                    print(f"[HashicorpVault] KV v1 also failed: {e2}")
                    return None
        except Exception as e:
            print(f"[HashicorpVault] Error getting secret {key}: {e}")
            return None

class AkeylessVaultClient(VaultClient):
    """Akeyless Vault client implementation."""
    
    def __init__(self):
        self.akeyless_url = os.getenv("AKEYLESS_URL", "https://api.akeyless.io")
        self.akeyless_token = os.getenv("AKEYLESS_TOKEN")
        self.akeyless_access_id = os.getenv("AKEYLESS_ACCESS_ID")
        self.akeyless_access_key = os.getenv("AKEYLESS_ACCESS_KEY")
    
    def get_secret(self, key: str) -> Optional[Dict[str, Any]]:
        """Get secret from Akeyless Vault."""
        if not self.akeyless_token and not (self.akeyless_access_id and self.akeyless_access_key):
            return None
        
        try:
            import akeyless
            
            # Authenticate if needed
            if not self.akeyless_token:
                auth_response = akeyless.auth(
                    access_id=self.akeyless_access_id,
                    access_key=self.akeyless_access_key
                )
                self.akeyless_token = auth_response.token
            
            # Get secret
            secret_response = akeyless.get_secret_value(
                names=[key],
                token=self.akeyless_token
            )
            
            if key in secret_response:
                # Parse JSON if it's a JSON string
                secret_value = secret_response[key]
                try:
                    return json.loads(secret_value)
                except:
                    return {"value": secret_value}
            return None
        except Exception as e:
            print(f"[AkeylessVault] Error getting secret {key}: {e}")
            return None

class AzureKeyVaultClient(VaultClient):
    """Azure Key Vault client implementation."""
    
    def __init__(self):
        self.vault_url = os.getenv("AZURE_KEYVAULT_URL")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
    
    def get_secret(self, key: str) -> Optional[Dict[str, Any]]:
        """Get secret from Azure Key Vault."""
        if not self.vault_url:
            return None
        
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import ClientSecretCredential
            
            credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            
            client = SecretClient(vault_url=self.vault_url, credential=credential)
            secret = client.get_secret(key)
            
            # Parse JSON if it's a JSON string
            try:
                return json.loads(secret.value)
            except:
                return {"value": secret.value}
        except Exception as e:
            print(f"[AzureKeyVault] Error getting secret {key}: {e}")
            return None

class GCPSecretManagerClient(VaultClient):
    """GCP Secret Manager client implementation."""
    
    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    def get_secret(self, key: str) -> Optional[Dict[str, Any]]:
        """Get secret from GCP Secret Manager."""
        if not self.project_id:
            return None
        
        try:
            from google.cloud import secretmanager
            
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.project_id}/secrets/{key}/versions/latest"
            
            response = client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            
            # Parse JSON if it's a JSON string
            try:
                return json.loads(secret_value)
            except:
                return {"value": secret_value}
        except Exception as e:
            print(f"[GCPSecretManager] Error getting secret {key}: {e}")
            return None

class AWSSecretsManagerClient(VaultClient):
    """AWS Secrets Manager client implementation."""
    
    def __init__(self):
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    def get_secret(self, key: str) -> Optional[Dict[str, Any]]:
        """Get secret from AWS Secrets Manager."""
        try:
            import boto3
            
            client = boto3.client(
                'secretsmanager',
                region_name=self.region,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key
            )
            
            response = client.get_secret_value(SecretId=key)
            secret_value = response['SecretString']
            
            # Parse JSON if it's a JSON string
            try:
                return json.loads(secret_value)
            except:
                return {"value": secret_value}
        except Exception as e:
            print(f"[AWSSecretsManager] Error getting secret {key}: {e}")
            return None

class VaultManager:
    """
    Manages multiple vault clients based on environment configuration.
    
    This is a READ-ONLY vault interface that only retrieves secrets.
    Secrets must be created and managed outside of this system using 
    vault-specific tools (vault CLI, web UI, etc.).
    
    Features smart caching with configurable TTL and security considerations.
    """
    
    def __init__(self):
        self.vault_type = os.getenv("VAULT_TYPE", "hashicorp").lower()
        self.clients = {
            "hashicorp": HashicorpVaultClient(),
            "akeyless": AkeylessVaultClient(),
            "azure": AzureKeyVaultClient(),
            "gcp": GCPSecretManagerClient(),
            "aws": AWSSecretsManagerClient()
        }
        self.active_client = self.clients.get(self.vault_type)
        
        # Cache configuration
        self.cache_enabled = os.getenv("VAULT_CACHE_ENABLED", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("VAULT_CACHE_TTL", "300"))  # 5 minutes default
        self.max_cache_size = int(os.getenv("VAULT_MAX_CACHE_SIZE", "100"))  # Max 100 secrets
        self.preload_cache = os.getenv("VAULT_PRELOAD_CACHE", "true").lower() == "true"
        
        # Cache storage
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        self._validate_dependencies()
        
        # Start cache cleanup thread if caching is enabled
        if self.cache_enabled:
            self._start_cache_cleanup_thread()
    
    def preload_cache_from_registry(self):
        """
        Preload cache with vault keys found in the registry during startup.
        This ensures authentication credentials are cached before first API call.
        """
        if not self.cache_enabled or not self.preload_cache:
            print("[VaultManager] Cache preloading disabled")
            return
        
        if not self.active_client:
            print("[VaultManager] No active vault client, skipping preload")
            return
        
        print("[VaultManager] Starting cache preload from registry...")
        
        try:
            # Import here to avoid circular imports
            from app.registry.client import fetch_agents_and_tools_from_registry
            
            agents = fetch_agents_and_tools_from_registry()
            vault_keys = set()
            
            # Extract vault keys from all agents
            for app_key, agent_info in agents.items():
                security_config = agent_info.get("security_config")
                if security_config and isinstance(security_config, dict):
                    vault_key = security_config.get("vaultKey")
                    if vault_key:
                        vault_keys.add(vault_key)
            
            # Also include data agent vault keys
            try:
                from app.utils.data_agents_client import data_agents_client
                data_agents = data_agents_client.fetch_data_agents()
                
                for agent_id, agent_info in data_agents.items():
                    vault_key = agent_info.get("vault_key")
                    if vault_key:
                        vault_keys.add(vault_key)
                
                print(f"[VaultManager] Found {len(data_agents)} data agents with vault keys")
            except Exception as e:
                print(f"[VaultManager] Error loading data agent vault keys: {e}")
            
            print(f"[VaultManager] Found {len(vault_keys)} unique vault keys to preload")
            
            # Preload each vault key
            preloaded_count = 0
            for vault_key in vault_keys:
                try:
                    secret_data = self.active_client.get_secret(vault_key)
                    if secret_data:
                        with self._cache_lock:
                            self._cache[vault_key] = (secret_data, time.time())
                        preloaded_count += 1
                        print(f"[VaultManager] Preloaded: {vault_key}")
                    else:
                        print(f"[VaultManager] Failed to preload: {vault_key} (secret not found)")
                except Exception as e:
                    print(f"[VaultManager] Error preloading {vault_key}: {e}")
            
            print(f"[VaultManager] Cache preload completed: {preloaded_count}/{len(vault_keys)} secrets cached")
            
        except Exception as e:
            print(f"[VaultManager] Error during cache preload: {e}")
        
        # Show final cache stats
        cache_stats = self.get_cache_stats()
        print(f"[VaultManager] Cache stats after preload: {cache_stats}")
    
    def preload_data_agent_credentials(self, data_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Preload data agent vault credentials into cache for better performance.
        
        Args:
            data_agents: List of data agent configurations
            
        Returns:
            Dictionary with preload results
        """
        if not self.cache_enabled:
            return {
                "status": "disabled",
                "message": "Vault caching is disabled"
            }
        
        if not self.active_client:
            return {
                "status": "error",
                "message": "No active vault client"
            }
        
        try:
            preloaded_count = 0
            failed_count = 0
            vault_keys = set()
            
            # Extract unique vault keys from data agents
            for agent in data_agents:
                # Check vault_key at agent level (from environments)
                vault_key = agent.get("vault_key")
                if vault_key:
                    vault_keys.add(vault_key)
                
                # Also check in environments array
                environments = agent.get("environments", [])
                for env in environments:
                    vault_key = env.get("vaultKey")
                    if vault_key:
                        vault_keys.add(vault_key)
            
            print(f"[VaultManager] Preloading {len(vault_keys)} data agent vault credentials")
            
            # Preload each vault key
            for vault_key in vault_keys:
                try:
                    # Use the get_secret method which handles caching
                    credentials = self.get_secret(vault_key)
                    if credentials:
                        preloaded_count += 1
                        print(f"[VaultManager] Preloaded credentials for vault key: {vault_key}")
                    else:
                        failed_count += 1
                        print(f"[VaultManager] Failed to preload vault key: {vault_key}")
                except Exception as e:
                    failed_count += 1
                    print(f"[VaultManager] Error preloading vault key {vault_key}: {e}")
            
            return {
                "status": "success",
                "preloaded_count": preloaded_count,
                "failed_count": failed_count,
                "total_keys": len(vault_keys),
                "vault_keys": list(vault_keys)
            }
            
        except Exception as e:
            print(f"[VaultManager] Error during data agent credential preload: {e}")
            return {
                "status": "error",
                "message": f"Error during credential preload: {str(e)}"
            }
    
    def _start_cache_cleanup_thread(self):
        """Start background thread to clean up expired cache entries."""
        def cleanup_expired():
            while True:
                try:
                    with self._cache_lock:
                        now = time.time()
                        expired_keys = [
                            key for key, (_, timestamp) in self._cache.items() 
                            if now - timestamp > self.cache_ttl
                        ]
                        for key in expired_keys:
                            del self._cache[key]
                        
                        # If cache is too large, remove oldest entries
                        if len(self._cache) > self.max_cache_size:
                            sorted_cache = sorted(
                                self._cache.items(), 
                                key=lambda x: x[1][1]  # Sort by timestamp
                            )
                            excess_count = len(self._cache) - self.max_cache_size
                            for i in range(excess_count):
                                del self._cache[sorted_cache[i][0]]
                
                except Exception as e:
                    print(f"[VaultManager] Cache cleanup error: {e}")
                
                # Run cleanup every minute
                time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_expired, daemon=True)
        cleanup_thread.start()
    
    def _validate_dependencies(self):
        """Validate that required dependencies are available for the configured vault type."""
        if self.vault_type == "none" or not self.active_client:
            return
        
        dependency_checks = {
            "hashicorp": ("hvac", "pip install hvac>=1.2.1"),
            "akeyless": ("akeyless", "pip install akeyless>=3.0.0"),
            "azure": ("azure.keyvault.secrets", "pip install azure-keyvault-secrets>=4.7.0 azure-identity>=1.15.0"),
            "gcp": ("google.cloud.secretmanager", "pip install google-cloud-secret-manager>=2.16.0"),
            "aws": ("boto3", "pip install boto3>=1.34.0")
        }
        
        if self.vault_type in dependency_checks:
            module_name, install_cmd = dependency_checks[self.vault_type]
            try:
                __import__(module_name)
                print(f"[VaultManager] ✓ {self.vault_type} vault dependencies are available")
            except ImportError:
                print(f"[VaultManager] ⚠️  WARNING: {self.vault_type} vault configured but dependencies missing.")
                print(f"[VaultManager] Install with: {install_cmd}")
                print(f"[VaultManager] Will fallback to environment variables for authentication.")
    
    def get_secret(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get secret from the configured vault system.
        
        This method only retrieves existing secrets - it does not create them.
        Secrets must be created externally using vault-specific tools.
        
        Supports caching based on VAULT_CACHE_ENABLED environment variable.
        """
        if not self.active_client:
            print(f"[VaultManager] No active client for vault type: {self.vault_type}")
            return None
        
        # Check cache first if enabled
        if self.cache_enabled:
            with self._cache_lock:
                if key in self._cache:
                    secret_data, timestamp = self._cache[key]
                    if time.time() - timestamp < self.cache_ttl:
                        print(f"[VaultManager] Cache hit for key: {key}")
                        # Make sure cached data is processed (in case it was cached raw)
                        if secret_data:
                            secret_data = CredentialProcessor.process_credentials_from_storage(secret_data)
                        return secret_data
                    else:
                        # Remove expired entry
                        del self._cache[key]
        
        # Get from vault
        secret_data = self.active_client.get_secret(key)
        
        # Process credentials if found (handle Base64 decoding of sensitive fields)
        if secret_data:
            secret_data = CredentialProcessor.process_credentials_from_storage(secret_data)
        
        # Cache the result if enabled and secret was found
        if self.cache_enabled and secret_data:
            with self._cache_lock:
                self._cache[key] = (secret_data, time.time())
                print(f"[VaultManager] Cached secret for key: {key}")
        
        return secret_data
    
    def get_auth_credentials(self, vault_key: str) -> Optional[Dict[str, Any]]:
        """
        Get authentication credentials from vault.
        
        Retrieves pre-stored authentication credentials for the given vault key.
        Credentials must be created externally in your vault system.
        
        Automatically processes Base64-encoded sensitive fields and parses JSON 
        values where appropriate.
        """
        return self.get_secret(vault_key)
    
    def health_check(self) -> Dict[str, Any]:
        """Check vault connectivity and return health status."""
        if not self.active_client:
            return {
                "status": "disabled",
                "vault_type": self.vault_type,
                "message": "No vault configured"
            }
        
        try:
            # Try to get a test key (this will fail but tells us if vault is reachable)
            self.active_client.get_secret("__health_check_test__")
            cache_info = {}
            if self.cache_enabled:
                with self._cache_lock:
                    cache_info = {
                        "cache_size": len(self._cache),
                        "cache_ttl": self.cache_ttl,
                        "max_cache_size": self.max_cache_size
                    }
            
            return {
                "status": "healthy",
                "vault_type": self.vault_type,
                "message": "Vault is accessible",
                "cache_enabled": self.cache_enabled,
                **cache_info
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "vault_type": self.vault_type,
                "message": f"Vault error: {str(e)}"
            }
    
    def clear_cache(self):
        """Clear all cached secrets. Useful for testing or security purposes."""
        if self.cache_enabled:
            with self._cache_lock:
                cleared_count = len(self._cache)
                self._cache.clear()
                print(f"[VaultManager] Cleared {cleared_count} cached secrets")
        else:
            print("[VaultManager] Cache is disabled")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_enabled:
            return {"cache_enabled": False}
        
        with self._cache_lock:
            now = time.time()
            expired_count = sum(
                1 for _, (_, timestamp) in self._cache.items() 
                if now - timestamp > self.cache_ttl
            )
            
            return {
                "cache_enabled": True,
                "cache_size": len(self._cache),
                "cache_ttl": self.cache_ttl,
                "max_cache_size": self.max_cache_size,
                "expired_entries": expired_count,
                "preload_enabled": self.preload_cache
            }

# Global vault manager instance
vault_manager = VaultManager()
