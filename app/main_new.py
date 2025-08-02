"""
Simplified MCP Server - Clean, modular FastAPI application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.core.config import app_config
from app.controllers import (
    health_controller,
    query_controller, 
    session_controller,
    admin_controller
)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title=app_config.APP_NAME,
    version=app_config.VERSION,
    description="Simplified MCP Server with modular architecture"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_controller.router)
app.include_router(query_controller.router)
app.include_router(session_controller.router)
app.include_router(admin_controller.router)

# Root endpoint
@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "app": app_config.APP_NAME,
        "version": app_config.VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "enhanced_query": "/enhanced/query", 
            "streaming": "/orchestration/enhanced/stream",
            "sessions": "/enhanced/session/create",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=app_config.HOST, 
        port=app_config.PORT
    )
