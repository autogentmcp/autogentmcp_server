#!/bin/bash
# Vault Dependencies Installation Script

set -e

echo "MCP Registry Server - Vault Dependencies Installer"
echo "================================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Please create one from .env.example"
    echo "   cp .env.example .env"
    echo "   # Edit .env to configure your vault type"
    exit 1
fi

# Read VAULT_TYPE from .env
VAULT_TYPE=$(grep "^VAULT_TYPE=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'" | tr '[:upper:]' '[:lower:]')

if [ -z "$VAULT_TYPE" ]; then
    echo "⚠️  VAULT_TYPE not found in .env file"
    echo "   Please set VAULT_TYPE to one of: hashicorp, akeyless, azure, gcp, aws"
    exit 1
fi

echo "🔍 Detected vault type: $VAULT_TYPE"

# Install core dependencies first
echo "📦 Installing core dependencies..."
pip install -r requirements.txt

# Install vault-specific dependencies
case $VAULT_TYPE in
    "hashicorp")
        echo "📦 Installing Hashicorp Vault dependencies..."
        pip install hvac>=1.2.1
        echo "✅ Hashicorp Vault dependencies installed"
        ;;
    "akeyless")
        echo "📦 Installing Akeyless Vault dependencies..."
        pip install akeyless>=3.0.0
        echo "✅ Akeyless Vault dependencies installed"
        ;;
    "azure")
        echo "📦 Installing Azure Key Vault dependencies..."
        pip install azure-keyvault-secrets>=4.7.0 azure-identity>=1.15.0
        echo "✅ Azure Key Vault dependencies installed"
        ;;
    "gcp")
        echo "📦 Installing GCP Secret Manager dependencies..."
        pip install google-cloud-secret-manager>=2.16.0
        echo "✅ GCP Secret Manager dependencies installed"
        ;;
    "aws")
        echo "📦 Installing AWS Secrets Manager dependencies..."
        pip install boto3>=1.34.0
        echo "✅ AWS Secrets Manager dependencies installed"
        ;;
    "none")
        echo "ℹ️  No vault configured - using environment variables only"
        ;;
    *)
        echo "❌ Unknown vault type: $VAULT_TYPE"
        echo "   Supported types: hashicorp, akeyless, azure, gcp, aws, none"
        exit 1
        ;;
esac

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Configure your vault credentials in .env"
echo "2. Start the server: uvicorn app.main:app --reload --port 8001"
echo "3. Test health: curl http://localhost:8001/health"
echo ""
echo "For troubleshooting, see: README_VAULT_INTEGRATION.md"
