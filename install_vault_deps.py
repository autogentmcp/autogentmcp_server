#!/usr/bin/env python3
"""
Vault Dependencies Installation Script
Cross-platform Python version
"""

import os
import sys
import subprocess
import re

def read_env_file():
    """Read VAULT_TYPE from .env file."""
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Please create one from .env.example")
        print("   cp .env.example .env")
        print("   # Edit .env to configure your vault type")
        return None
    
    vault_type = None
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('VAULT_TYPE='):
                vault_type = line.split('=')[1].strip().strip('"').strip("'").lower()
                break
    
    if not vault_type:
        print("⚠️  VAULT_TYPE not found in .env file")
        print("   Please set VAULT_TYPE to one of: hashicorp, akeyless, azure, gcp, aws")
        return None
    
    return vault_type

def install_package(package):
    """Install a Python package using pip."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("MCP Registry Server - Vault Dependencies Installer")
    print("=================================================")
    
    vault_type = read_env_file()
    if not vault_type:
        sys.exit(1)
    
    print(f"🔍 Detected vault type: {vault_type}")
    
    # Install core dependencies first
    print("📦 Installing core dependencies...")
    if not install_package('-r requirements.txt'):
        print("❌ Failed to install core dependencies")
        sys.exit(1)
    
    # Install vault-specific dependencies
    vault_deps = {
        'hashicorp': ['hvac>=1.2.1'],
        'akeyless': ['akeyless>=3.0.0'],
        'azure': ['azure-keyvault-secrets>=4.7.0', 'azure-identity>=1.15.0'],
        'gcp': ['google-cloud-secret-manager>=2.16.0'],
        'aws': ['boto3>=1.34.0'],
        'none': []
    }
    
    if vault_type not in vault_deps:
        print(f"❌ Unknown vault type: {vault_type}")
        print("   Supported types: hashicorp, akeyless, azure, gcp, aws, none")
        sys.exit(1)
    
    deps = vault_deps[vault_type]
    
    if not deps:
        print("ℹ️  No vault configured - using environment variables only")
    else:
        print(f"📦 Installing {vault_type} Vault dependencies...")
        for dep in deps:
            if not install_package(dep):
                sys.exit(1)
        print(f"✅ {vault_type} Vault dependencies installed")
    
    print("\n🎉 Installation complete!")
    print("\nNext steps:")
    print("1. Configure your vault credentials in .env")
    print("2. Start the server: uvicorn app.main:app --reload --port 8001")
    print("3. Test health: curl http://localhost:8001/health")
    print("\nFor troubleshooting, see: README_VAULT_INTEGRATION.md")

if __name__ == "__main__":
    main()
