# Vault Dependencies Installation Script for Windows
# PowerShell version

Write-Host "MCP Registry Server - Vault Dependencies Installer" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Check if .env exists
if (-not (Test-Path .env)) {
    Write-Host "⚠️  .env file not found. Please create one from .env.example" -ForegroundColor Yellow
    Write-Host "   Copy-Item .env.example .env"
    Write-Host "   # Edit .env to configure your vault type"
    exit 1
}

# Read VAULT_TYPE from .env
$vaultType = ""
if (Test-Path .env) {
    $envContent = Get-Content .env
    foreach ($line in $envContent) {
        if ($line.StartsWith("VAULT_TYPE=")) {
            $vaultType = $line.Split("=")[1].Replace('"', '').Replace("'", '').ToLower()
            break
        }
    }
}

if ([string]::IsNullOrEmpty($vaultType)) {
    Write-Host "⚠️  VAULT_TYPE not found in .env file" -ForegroundColor Yellow
    Write-Host "   Please set VAULT_TYPE to one of: hashicorp, akeyless, azure, gcp, aws"
    exit 1
}

Write-Host "🔍 Detected vault type: $vaultType" -ForegroundColor Cyan

# Install core dependencies first
Write-Host "📦 Installing core dependencies..." -ForegroundColor Blue
pip install -r requirements.txt

# Install vault-specific dependencies
switch ($vaultType) {
    "hashicorp" {
        Write-Host "📦 Installing Hashicorp Vault dependencies..." -ForegroundColor Blue
        pip install "hvac>=1.2.1"
        Write-Host "✅ Hashicorp Vault dependencies installed" -ForegroundColor Green
    }
    "akeyless" {
        Write-Host "📦 Installing Akeyless Vault dependencies..." -ForegroundColor Blue
        pip install "akeyless>=3.0.0"
        Write-Host "✅ Akeyless Vault dependencies installed" -ForegroundColor Green
    }
    "azure" {
        Write-Host "📦 Installing Azure Key Vault dependencies..." -ForegroundColor Blue
        pip install "azure-keyvault-secrets>=4.7.0" "azure-identity>=1.15.0"
        Write-Host "✅ Azure Key Vault dependencies installed" -ForegroundColor Green
    }
    "gcp" {
        Write-Host "📦 Installing GCP Secret Manager dependencies..." -ForegroundColor Blue
        pip install "google-cloud-secret-manager>=2.16.0"
        Write-Host "✅ GCP Secret Manager dependencies installed" -ForegroundColor Green
    }
    "aws" {
        Write-Host "📦 Installing AWS Secrets Manager dependencies..." -ForegroundColor Blue
        pip install "boto3>=1.34.0"
        Write-Host "✅ AWS Secrets Manager dependencies installed" -ForegroundColor Green
    }
    "none" {
        Write-Host "ℹ️  No vault configured - using environment variables only" -ForegroundColor Yellow
    }
    default {
        Write-Host "❌ Unknown vault type: $vaultType" -ForegroundColor Red
        Write-Host "   Supported types: hashicorp, akeyless, azure, gcp, aws, none"
        exit 1
    }
}

Write-Host ""
Write-Host "🎉 Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Configure your vault credentials in .env"
Write-Host "2. Start the server: uvicorn app.main:app --reload --port 8001"
Write-Host "3. Test health: curl http://localhost:8001/health"
Write-Host ""
Write-Host "For troubleshooting, see: README_VAULT_INTEGRATION.md"
