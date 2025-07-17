# MCP Registry Server with Vault Integration

This document explains how to use the updated MCP Registry Server with comprehensive authentication and vault integration.

## Overview

The system now supports the new registry endpoint structure from `/applications/with-endpoints?environment=production` and integrates with multiple vault systems for secure credential management.

## Registry Data Structure

The registry returns applications with the following structure:

```json
{
  "name": "app_x29iirgfycp9e92iqkb04",
  "description": "Order Tracking Service for MCP POC",
  "appKey": "app_x29iirgfycp9e92iqkb04",
  "authenticationMethod": "api_key",
  "environment": {
    "baseDomain": "http://localhost:8081",
    "security": {
      "vaultKey": "env_cmd72flj7000jn5hwd4zv148p_security_settings"
    }
  },
  "endpoints": [
    {
      "name": "getOrderStatus",
      "path": "/orders/{orderId}/status",
      "method": "GET",
      "description": "Get order status by order ID",
      "pathParams": { "orderId": "String" },
      "queryParams": { "status": "String" }
    }
  ]
}
```

## Authentication Methods

The system supports the following authentication methods:

### 1. API Key (`api_key`)
- **Header**: `X-API-Key: {api_key}`
- **Vault Format**: `{"api_key": "your-key"}` or `{"value": "your-key"}`

### 2. Bearer Token (`bearer_token`)
- **Header**: `Authorization: Bearer {token}`
- **Vault Format**: `{"token": "your-token"}` or `{"value": "your-token"}`

### 3. Basic Authentication (`basic_auth`)
- **Header**: `Authorization: Basic {base64(username:password)}`
- **Vault Format**: `{"username": "user", "password": "pass"}`

### 4. OAuth 2.0 (`oauth2`)
- **Header**: `Authorization: Bearer {access_token}`
- **Vault Format**: `{"access_token": "token"}` or `{"token": "token"}`

### 5. JWT Token (`jwt`)
- **Header**: `Authorization: Bearer {jwt_token}`
- **Vault Format**: `{"jwt_token": "token"}` or `{"token": "token"}`

### 6. Azure Subscription (`azure_subscription`)
- **Header**: `Ocp-Apim-Subscription-Key: {subscription_key}`
- **Vault Format**: `{"subscription_key": "key"}` or `{"value": "key"}`

### 7. Azure APIM (`azure_apim`)
- **Header**: `Ocp-Apim-Subscription-Key: {apim_key}`
- **Vault Format**: `{"apim_key": "key"}` or `{"value": "key"}`

### 8. GCP Service Account (`gcp_service_account`)
- **Header**: `Authorization: Bearer {access_token}`
- **Vault Format**: `{"access_token": "token"}` or `{"token": "token"}`

### 9. Signature Authentication (`signature_auth`)
- **Headers**: `X-Signature: {signature}`, `X-Timestamp: {timestamp}`
- **Vault Format**: `{"signature": "sig", "timestamp": "ts"}`

### 10. Custom Headers
- **Vault Format**: `{"custom_headers": {"X-Custom": "value"}}`

## Vault Integration

The system supports multiple vault providers for **retrieving** authentication credentials. This is a **read-only** integration - secrets must be created and managed externally using vault-specific tools.

### Smart Caching

The system includes intelligent caching to balance performance and security:

- **Configurable**: Enable/disable caching via `VAULT_CACHE_ENABLED`
- **TTL-based**: Secrets expire after `VAULT_CACHE_TTL` seconds (default: 5 minutes)  
- **Size-limited**: Maximum `VAULT_MAX_CACHE_SIZE` secrets cached (default: 100)
- **Startup Preload**: Automatically cache all vault keys found in registry during startup
- **Auto-refresh**: Cache is preloaded when registry is refreshed
- **Thread-safe**: Concurrent access protection
- **Auto-cleanup**: Expired entries removed automatically

### Cache Configuration

```env
VAULT_CACHE_ENABLED=true          # Enable/disable caching
VAULT_CACHE_TTL=300              # Cache TTL in seconds (5 minutes)
VAULT_MAX_CACHE_SIZE=100         # Maximum cached secrets
VAULT_PRELOAD_CACHE=true         # Preload cache with registry vault keys
```

### Startup Preload Process

1. **Registry Scan**: System scans all applications in registry for `vaultKey` fields
2. **Vault Retrieval**: Fetches each unique vault key from your vault system
3. **Cache Population**: Stores secrets in memory cache with TTL
4. **Ready State**: First API requests use cached credentials (fast response)

**Benefits:**
- **Zero Cold Start**: First API requests are fast (no vault round-trip)
- **Proactive Loading**: Identifies missing secrets during startup
- **Reduced Latency**: Authentication headers generated from cache
- **Fault Tolerance**: Cached secrets survive brief vault outages

**Security Considerations:**
- **Short TTL**: Default 5-minute expiry balances performance vs. freshness
- **Memory protection**: Cached secrets are cleared on process termination
- **Size limits**: Prevents memory exhaustion
- **Manual clearing**: Admin can clear cache via API endpoint

### Important: Secret Management

- **✅ The MCP server ONLY retrieves secrets**
- **❌ The MCP server does NOT create or modify secrets**
- **📝 Use vault-specific tools to create secrets:**
  - HashiCorp Vault: `vault kv put` command or web UI
  - Azure Key Vault: Azure portal or Azure CLI
  - AWS Secrets Manager: AWS console or AWS CLI
  - GCP Secret Manager: Google Cloud console or gcloud CLI
  - Akeyless: Akeyless console or CLI

### Hashicorp Vault
```env
VAULT_TYPE=hashicorp
VAULT_URL=http://localhost:8200
VAULT_TOKEN=your-vault-token
VAULT_NAMESPACE=your-namespace
VAULT_PATH=your-app-path          # Optional: base path for secrets
VAULT_MOUNT=secret               # Optional: mount point (default: secret)
```

**Path Structure:**
- If `VAULT_PATH` is set, secrets are stored at: `{VAULT_MOUNT}/{VAULT_PATH}/{vault_key}`
- If `VAULT_PATH` is not set, secrets are stored at: `{VAULT_MOUNT}/{vault_key}`
- Example: With `VAULT_PATH=autogentmcp` and `VAULT_MOUNT=secret`, the vault key `env_cmd72flj7000jn5hwd4zv148p_security_settings` becomes: `secret/autogentmcp/env_cmd72flj7000jn5hwd4zv148p_security_settings`

### Akeyless Vault
```env
VAULT_TYPE=akeyless
AKEYLESS_URL=https://api.akeyless.io
AKEYLESS_TOKEN=your-token
# OR
AKEYLESS_ACCESS_ID=your-access-id
AKEYLESS_ACCESS_KEY=your-access-key
```

### Azure Key Vault
```env
VAULT_TYPE=azure
AZURE_KEYVAULT_URL=https://your-keyvault.vault.azure.net/
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id
```

### GCP Secret Manager
```env
VAULT_TYPE=gcp
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### AWS Secrets Manager
```env
VAULT_TYPE=aws
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

## Setup Instructions

1. **Create Secrets in Your Vault System First**:
   
   Example for HashiCorp Vault:
   ```bash
   # Create authentication credentials for an application
   vault kv put secret/autogentmcp/env_cmd72flj7000jn5hwd4zv148p_security_settings \
     api_key=your-actual-api-key \
     custom_headers='{"X-Custom-Header": "value"}'
   ```

2. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your vault configuration
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   
   # Install vault-specific dependencies
   # For Hashicorp Vault:
   pip install hvac>=1.2.1
   
   # For Azure Key Vault:
   pip install azure-keyvault-secrets>=4.7.0 azure-identity>=1.15.0
   
   # For AWS Secrets Manager:
   pip install boto3>=1.34.0
   
   # For GCP Secret Manager:
   pip install google-cloud-secret-manager>=2.16.0
   
   # For Akeyless:
   pip install akeyless>=3.0.0
   ```

4. **Run the Server**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

## API Endpoints

### Health Check
```http
GET /health
```

### Query Processing
```http
POST /query
Content-Type: application/json

{
  "query": "Get order status for order 12345",
  "session_id": "user123"
}
```

### Manual Registry Sync
```http
POST /sync_registry
```

### Set Authentication Credential (Fallback)
```http
POST /auth/set_credential
Content-Type: application/json

{
  "key": "app_x29iirgfycp9e92iqkb04_api_key",
  "value": "your-api-key"
}
```

### Session Management
```http
GET /sessions
GET /sessions/{session_id}
DELETE /sessions/{session_id}
```

### Vault Cache Management
```http
GET /vault/stats
POST /vault/clear_cache
```

## Security Best Practices

1. **Use Vault Systems**: Always store credentials in a proper vault system rather than environment variables.

2. **Rotate Credentials**: Implement regular credential rotation in your vault system.

3. **Environment Isolation**: Use different vault keys for different environments (development, staging, production).

4. **Access Control**: Implement proper access control in your vault system.

5. **Audit Logging**: Enable audit logging in your vault system to track credential access.

## Error Handling

The system includes comprehensive error handling:

1. **Registry Errors**: Falls back to cached data if registry is unavailable.
2. **Vault Errors**: Falls back to environment variables and then cached credentials.
3. **Authentication Errors**: Logs errors and continues with available credentials.

## Monitoring

Monitor the following:

1. **Registry Sync Status**: Check logs for registry sync errors.
2. **Vault Connectivity**: Monitor vault connection health.
3. **Authentication Failures**: Track authentication errors in logs.
4. **API Response Times**: Monitor endpoint response times.

## Migration Guide

To migrate from the old system:

1. Update your registry endpoint to return the new format.
2. Set up your vault system with the appropriate credentials.
3. Update environment variables to point to your vault.
4. Test authentication with a few applications before full rollout.

## Troubleshooting

### Common Issues

1. **"Not Found" errors**: Ensure your registry URL is correct and the endpoint exists.
2. **Authentication failures**: Check that the vault key exists and contains the correct format.
3. **Vault connection errors**: Verify vault configuration and network connectivity.
4. **Missing credentials**: Ensure vault contains the required keys for your applications.

### Debug Mode

Enable debug logging by setting:
```env
LOG_LEVEL=DEBUG
```

This will show detailed information about registry calls, vault lookups, and authentication attempts.
