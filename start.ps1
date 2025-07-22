#!/usr/bin/env pwsh

Write-Host "🚀 Starting MCP Server with Streamlit UI..." -ForegroundColor Green
Write-Host ""

# Start FastAPI server in the background
Write-Host "[1/2] Starting FastAPI backend..." -ForegroundColor Yellow
$fastapi = Start-Process powershell -ArgumentList "-Command", "cd '$PSScriptRoot'; python -m uvicorn app.main:app --host localhost --port 8001 --reload" -PassThru
Start-Sleep -Seconds 3

# Start Streamlit UI
Write-Host "[2/2] Starting Streamlit UI..." -ForegroundColor Yellow
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  🚀 Services Starting:" -ForegroundColor White
Write-Host "  📡 FastAPI Backend: http://localhost:8001" -ForegroundColor Blue
Write-Host "  🎨 Streamlit UI:    http://localhost:8502" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Start Streamlit (this will block)
streamlit run ui/streamlit_app.py --server.port 8502

# If we get here, cleanup
Write-Host "Cleaning up..." -ForegroundColor Yellow
Stop-Process -Id $fastapi.Id -Force -ErrorAction SilentlyContinue
