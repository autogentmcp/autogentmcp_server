#!/usr/bin/env python3
"""
LLM Configuration Management Utility

This script helps manage LLM provider configurations with a least-permissive approach.
Only explicitly enabled providers will be initialized.
"""

import json
import os
import sys
from typing import Dict, Any, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, environment variables from .env file may not be loaded")

def load_config(config_path: str = "llm_config.json") -> Dict[str, Any]:
    """Load current LLM configuration"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def save_config(config: Dict[str, Any], config_path: str = "llm_config.json"):
    """Save LLM configuration"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def enable_provider(provider: str, config_path: str = "llm_config.json"):
    """Enable a specific LLM provider"""
    config = load_config(config_path)
    
    # Add to enabled providers if not already there
    enabled_providers = config.get("enabled_providers", [])
    if provider not in enabled_providers:
        enabled_providers.append(provider)
        config["enabled_providers"] = enabled_providers
    
    # Mark as enabled in provider configs
    provider_configs = config.get("provider_configs", {})
    if provider in provider_configs:
        provider_configs[provider]["enabled"] = True
    else:
        print(f"Warning: Provider {provider} not found in provider_configs")
    
    save_config(config, config_path)
    print(f"✅ Enabled provider: {provider}")

def disable_provider(provider: str, config_path: str = "llm_config.json"):
    """Disable a specific LLM provider"""
    config = load_config(config_path)
    
    # Remove from enabled providers
    enabled_providers = config.get("enabled_providers", [])
    if provider in enabled_providers:
        enabled_providers.remove(provider)
        config["enabled_providers"] = enabled_providers
    
    # Mark as disabled in provider configs
    provider_configs = config.get("provider_configs", {})
    if provider in provider_configs:
        provider_configs[provider]["enabled"] = False
    
    save_config(config, config_path)
    print(f"❌ Disabled provider: {provider}")

def show_status(config_path: str = "llm_config.json"):
    """Show current provider status"""
    config = load_config(config_path)
    
    print("\n🔧 LLM Provider Configuration Status")
    print("=" * 50)
    
    enabled_providers = config.get("enabled_providers", [])
    provider_configs = config.get("provider_configs", {})
    
    print(f"Enabled Providers: {enabled_providers}")
    print(f"Fallback Provider: {config.get('fallback_provider', 'openai')}")
    print(f"Allow Disabled Fallback: {config.get('allow_disabled_fallback', False)}")
    
    print("\nProvider Details:")
    for provider, details in provider_configs.items():
        status_icon = "✅" if details.get("enabled", False) else "❌"
        env_vars = details.get("required_env_vars", [])
        missing_vars = [var for var in env_vars if not os.getenv(var)]
        env_status = "🔑" if not missing_vars else f"⚠️  Missing: {missing_vars}"
        
        print(f"  {status_icon} {provider:10} - {details.get('description', '')}")
        print(f"     Model: {details.get('default_model', 'N/A')}")
        print(f"     Env:   {env_status}")

def create_minimal_config(config_path: str = "llm_config.json"):
    """Create a minimal least-permissive configuration"""
    minimal_config = {
        "enabled_providers": ["openai"],
        "provider_configs": {
            "openai": {
                "enabled": True,
                "default_model": "gpt-4o-mini",
                "required_env_vars": ["OPENAI_API_KEY"],
                "description": "OpenAI GPT models"
            },
            "deepseek": {
                "enabled": False,
                "default_model": "deepseek-chat",
                "required_env_vars": ["DEEPSEEK_API_KEY"],
                "description": "DeepSeek models"
            },
            "ollama": {
                "enabled": False,
                "default_model": "qwen2.5:32b",
                "required_env_vars": [],
                "description": "Local Ollama models"
            }
        },
        "task_routing": {
            "code_generation": {"provider": "openai", "model": "gpt-4o-mini"},
            "data_analysis": {"provider": "openai", "model": "gpt-4o-mini"},
            "general_chat": {"provider": "openai", "model": "gpt-4o-mini"},
            "tool_selection": {"provider": "openai", "model": "gpt-4o-mini"},
            "agent_selection": {"provider": "openai", "model": "gpt-4o-mini"}
        },
        "fallback_provider": "openai",
        "allow_disabled_fallback": False
    }
    
    save_config(minimal_config, config_path)
    print(f"✅ Created minimal least-permissive configuration at {config_path}")

def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("🤖 LLM Configuration Manager")
        print("\nUsage:")
        print(f"  {sys.argv[0]} status                    # Show current status")
        print(f"  {sys.argv[0]} enable <provider>         # Enable a provider")
        print(f"  {sys.argv[0]} disable <provider>        # Disable a provider")
        print(f"  {sys.argv[0]} create-minimal            # Create minimal config")
        print(f"  {sys.argv[0]} reset                     # Reset to minimal config")
        print("\nSupported providers: openai, deepseek, ollama")
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        show_status()
    elif command == "enable" and len(sys.argv) > 2:
        enable_provider(sys.argv[2])
    elif command == "disable" and len(sys.argv) > 2:
        disable_provider(sys.argv[2])
    elif command in ["create-minimal", "reset"]:
        create_minimal_config()
    else:
        print(f"Unknown command: {command}")
        print("Use 'status', 'enable <provider>', 'disable <provider>', or 'create-minimal'")

if __name__ == "__main__":
    main()
