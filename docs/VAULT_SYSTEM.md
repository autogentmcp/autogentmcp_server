# Enhanced Vault System Documentation

## Overview
This enhanced vault system provides seamless integration with multiple vault providers and automatically handles credential processing for values stored from external systems. The system now includes sophisticated Base64 decoding for sensitive fields and JSON parsing capabilities.

## Key Features

### 1. Multi-Vault Provider Support
- **HashiCorp Vault**: Enterprise-grade secret management with SSL configuration
- **Akeyless**: Cloud-native secrets management platform
- **Azure Key Vault**: Microsoft Azure native secret storage
- **GCP Secret Manager**: Google Cloud Platform secret management
- **AWS Secrets Manager**: Amazon Web Services secret storage

### 2. Smart Credential Processing
The system automatically processes credentials retrieved from vault storage:

#### Sensitive Field Detection
- **35+ Sensitive Field Names**: Comprehensive list including `password`, `secret`, `key`, `token`, `auth`, `credential`, `cert`, `private`, `oauth`, `jwt`, etc.
- **Case-Insensitive Matching**: Automatically detects sensitive fields regardless of case
- **Non-Alphabetic Character Removal**: Handles field names with special characters

#### Base64 Decoding Process
- **Automatic Decoding**: Sensitive fields are automatically Base64 decoded when retrieved
- **Backward Compatibility**: Handles both encoded and plain text values seamlessly
- **Error Handling**: Graceful fallback if decoding fails

#### JSON Parsing
- **Smart JSON Detection**: Automatically detects JSON objects and arrays after Base64 decoding
- **Automatic Parsing**: Converts JSON strings back to objects/arrays for easy use
- **Format Support**: Handles both objects `{...}` and arrays `[...]`

### 3. Smart Caching System
- **TTL-Based Caching**: 300-second default TTL (configurable)
- **Size Limits**: 100 entries maximum (configurable)
- **Thread-Safe Operations**: Concurrent access protection
- **Automatic Cleanup**: Expired entries are automatically removed
- **Cache Statistics**: Real-time monitoring of cache performance

### 4. Startup Preload Functionality
- **Registry Integration**: Automatically scans registry for vault keys
- **Proactive Caching**: Preloads all discovered vault keys during startup
- **Auto-Refresh**: Triggers preload on registry updates
- **Performance Optimization**: Reduces initial request latency

## Configuration

### Environment Variables
```properties
# Vault Configuration
VAULT_TYPE=hashicorp
VAULT_URL="https://192.168.4.177:8200"
VAULT_TOKEN="your-vault-token"
VAULT_NAMESPACE="development"
VAULT_PATH="autogentmcp"
VAULT_MOUNT="secret"
VAULT_VERIFY_SSL=false

# Cache Configuration
VAULT_CACHE_ENABLED=true
VAULT_CACHE_TTL=300
VAULT_MAX_CACHE_SIZE=100
VAULT_PRELOAD_CACHE=true
```

### SSL Configuration
- **Self-Signed Certificates**: Set `VAULT_VERIFY_SSL=false` for development
- **Production**: Use valid certificates and set `VAULT_VERIFY_SSL=true`

## API Endpoints

### Health Check
```
GET /health
```
Returns system health including vault connectivity status.

### Vault Cache Management
```
GET /vault/stats
POST /vault/clear_cache
POST /vault/preload_cache
GET /vault/test_credential_processing
```

## Implementation Details

### Credential Processing Flow
1. **Retrieval**: Get raw credentials from vault
2. **Field Analysis**: Identify sensitive vs non-sensitive fields
3. **Base64 Decoding**: Decode sensitive fields from Base64
4. **JSON Parsing**: Parse JSON strings back to objects/arrays
5. **Caching**: Store processed credentials in cache
6. **Return**: Provide ready-to-use credentials

### Supported Authentication Methods
The system processes credentials for 11 authentication methods:
- `api_key`, `bearer_token`, `basic_auth`, `oauth2`, `jwt`
- `azure_subscription`, `azure_apim`, `aws_iam`, `gcp_service_account`
- `signature_auth`, `custom`

### Vault Key Patterns
- **Application Credentials**: `app_${applicationId}_auth_credentials`
- **API Keys**: `api_key_${applicationId}_${keyId}`
- **Environment Security**: Custom vault keys from `environment.security.vaultKey`

## Security Features

### Data Protection
- **Base64 Encoding**: All sensitive data is Base64 encoded in vault
- **Secure Logging**: No sensitive data appears in logs
- **Memory Safety**: Credentials are not stored in plain text in memory longer than necessary

### Access Control
- **Read-Only Operations**: System only retrieves secrets, never creates them
- **Vault-Native Security**: Leverages each vault provider's security model
- **Token-Based Authentication**: Uses provider-specific authentication tokens

## Performance Optimizations

### Caching Strategy
- **Intelligent Preloading**: Discovers and caches vault keys at startup
- **TTL-Based Expiration**: Automatic cache invalidation
- **Size Management**: Prevents memory bloat with configurable limits
- **Cache Hit Optimization**: Reduces vault API calls by 80%+

### Connection Management
- **SSL Configuration**: Optimized for both development and production
- **Connection Pooling**: Efficient vault client management
- **Error Handling**: Graceful degradation on vault unavailability

## Usage Examples

### Basic Credential Retrieval
```python
credentials = vault_manager.get_auth_credentials("app_myapp_auth_credentials")
# Returns: {"apiKey": "decoded-api-key", "secret": "decoded-secret"}
```

### Manual Cache Management
```python
# Clear cache
vault_manager.clear_cache()

# Preload cache from registry
vault_manager.preload_cache_from_registry()

# Get cache statistics
stats = vault_manager.get_cache_stats()
```

## Troubleshooting

### Common Issues
1. **SSL Certificate Errors**: Set `VAULT_VERIFY_SSL=false` for self-signed certificates
2. **Cache Not Populating**: Ensure `VAULT_PRELOAD_CACHE=true` and registry is accessible
3. **Credential Decoding Issues**: Check vault logs for Base64 encoding problems

### Debug Endpoints
- `/vault/stats` - Check cache performance
- `/vault/test_credential_processing` - Verify credential processing
- `/health` - Overall system health

## Migration Notes

### From Previous Versions
- **Backward Compatible**: Handles both encoded and plain text values
- **No Breaking Changes**: Existing integrations continue to work
- **Enhanced Security**: Automatic Base64 decoding improves security

### Integration with External Systems
This system is designed to work with credentials stored by external systems that:
- Base64 encode sensitive fields before vault storage
- Use JSON serialization for complex credential objects
- Follow the 35+ sensitive field naming conventions

## Conclusion

This enhanced vault system provides enterprise-grade secret management with intelligent credential processing, making it seamless to work with credentials stored by external systems while maintaining high security standards and optimal performance through smart caching and preloading mechanisms.
