# LLM Provider Configuration Guide

This system implements a **least-permissive approach** to LLM provider management. Only explicitly enabled providers will be initialized, providing better security and cost control.

## Overview

By default, **only OpenAI is enabled**. All other providers (DeepSeek, Ollama) are disabled by default and must be explicitly enabled.

## Configuration File

The `llm_config.json` file controls which LLM providers are enabled:

```json
{
  "enabled_providers": ["openai"],
  "provider_configs": {
    "openai": {
      "enabled": true,
      "default_model": "gpt-4o-mini",
      "required_env_vars": ["OPENAI_API_KEY"],
      "description": "OpenAI GPT models"
    },
    "deepseek": {
      "enabled": false,
      "default_model": "deepseek-chat", 
      "required_env_vars": ["DEEPSEEK_API_KEY"],
      "description": "DeepSeek models"
    },
    "ollama": {
      "enabled": false,
      "default_model": "qwen2.5:32b",
      "required_env_vars": [],
      "description": "Local Ollama models"
    }
  },
  "fallback_provider": "openai",
  "allow_disabled_fallback": false
}
```

## Key Features

### 🔒 Least-Permissive by Default
- Only OpenAI is enabled by default
- Providers must be explicitly enabled in both:
  1. `enabled_providers` array
  2. `provider_configs[provider].enabled = true`

### 🛡️ Environment Variable Validation
- System checks for required environment variables before initialization
- Missing variables prevent provider initialization
- Clear error messages for missing credentials

### 🔄 Smart Fallback System
- Configurable fallback provider
- Option to allow/disallow disabled provider fallbacks
- Graceful degradation to any available provider

### 📊 Provider Status Monitoring
- Real-time status of all providers
- Environment variable validation
- Initialization success tracking

## Quick Start

### 1. Check Current Status
```bash
python manage_llm_config.py status
```

### 2. Enable Additional Providers
```bash
# Enable DeepSeek (requires DEEPSEEK_API_KEY)
python manage_llm_config.py enable deepseek

# Enable Ollama (no API key needed)
python manage_llm_config.py enable ollama
```

### 3. Disable Providers
```bash
python manage_llm_config.py disable deepseek
```

### 4. Reset to Minimal Configuration
```bash
python manage_llm_config.py reset
```

## Environment Variables

### Required Variables by Provider:
- **OpenAI**: `OPENAI_API_KEY`
- **DeepSeek**: `DEEPSEEK_API_KEY` 
- **Ollama**: None (local installation)

### Optional Model Override Variables:
- `OPENAI_DEFAULT_MODEL` (default: gpt-4o-mini)
- `DEEPSEEK_DEFAULT_MODEL` (default: deepseek-chat)
- `OLLAMA_DEFAULT_MODEL` (default: qwen2.5:32b)

## Security Benefits

### 🚫 No Unwanted API Calls
- Disabled providers are never initialized
- No accidental API calls to unwanted services
- Zero network requests to disabled providers

### 💰 Cost Control
- Only pay for explicitly enabled services
- Prevent surprise charges from unused providers
- Clear visibility into active providers

### 🔐 Credential Management
- Only required credentials need to be provided
- Missing credentials don't break the system
- Clear feedback on credential requirements

## Task Routing

Tasks are routed to specific providers based on configuration:

```json
{
  "task_routing": {
    "code_generation": {"provider": "openai", "model": "gpt-4o-mini"},
    "data_analysis": {"provider": "openai", "model": "gpt-4o-mini"},
    "general_chat": {"provider": "openai", "model": "gpt-4o-mini"},
    "tool_selection": {"provider": "openai", "model": "gpt-4o-mini"},
    "agent_selection": {"provider": "openai", "model": "gpt-4o-mini"}
  }
}
```

## Example Configurations

### Minimal (OpenAI Only)
```json
{
  "enabled_providers": ["openai"]
}
```

### Multi-Provider Setup
```json
{
  "enabled_providers": ["openai", "ollama"],
  "task_routing": {
    "code_generation": {"provider": "openai", "model": "gpt-4o-mini"},
    "general_chat": {"provider": "ollama", "model": "qwen2.5:32b"}
  }
}
```

### Local Development (Ollama Only)
```json
{
  "enabled_providers": ["ollama"],
  "fallback_provider": "ollama"
}
```

## Error Handling

The system provides clear error messages for common issues:

- ❌ **Missing API Keys**: Clear indication of which environment variables are missing
- ❌ **Provider Not Available**: Automatic fallback to available providers
- ❌ **No Providers Enabled**: System refuses to start if no providers are available
- ❌ **Invalid Configuration**: Validation errors with specific details

## Monitoring and Debugging

### Check Provider Status Programmatically:
```python
from app.llm import MultiModeLLMClient

client = MultiModeLLMClient()
status = client.get_provider_status()
enabled = client.get_enabled_providers()
```

### View Initialization Logs:
Look for log messages during startup:
- ✅ `Initialized {provider} client: {description}`
- ❌ `Failed to initialize {provider} client: {error}`
- ⚠️ `Provider {provider} missing required env vars: {vars}`

## Best Practices

1. **Start Minimal**: Begin with only OpenAI enabled
2. **Test Incrementally**: Enable one provider at a time
3. **Validate Environment**: Ensure all required variables are set
4. **Monitor Costs**: Track usage across enabled providers
5. **Review Regularly**: Disable unused providers to reduce attack surface

## Troubleshooting

### Provider Not Available Error
```
ValueError: Provider deepseek not available for task code_generation
```
**Solution**: Either enable the provider or update task routing to use an available provider.

### Missing Environment Variables
```
Cannot initialize deepseek: missing required environment variables: ['DEEPSEEK_API_KEY']
```
**Solution**: Set the required environment variable or disable the provider.

### No Providers Available
```
RuntimeError: No LLM clients successfully initialized
```
**Solution**: Enable at least one provider and ensure its requirements are met.
