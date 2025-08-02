"""
Health controller - handles health check endpoints.
"""
from fastapi import APIRouter
from app.auth.vault_manager import vault_manager
from app.models.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    vault_health = vault_manager.health_check()
    return {
        "status": "ok",
        "vault": vault_health
    }
