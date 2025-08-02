"""
Core configuration for the simplified MCP server.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    """Simplified application configuration."""
    
    # Basic settings
    APP_NAME = "MCP Server"
    VERSION = "2.0.0"
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8001"))
    
    # Orchestrator timeouts
    ORCHESTRATOR_TIMEOUT = int(os.getenv("ORCHESTRATOR_TIMEOUT", "600"))  # 10 minutes
    STREAM_TIMEOUT = int(os.getenv("STREAM_TIMEOUT", "120"))  # 2 minutes for streaming
    
    # CORS settings
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mcp_server.db")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# Global config instance
app_config = AppConfig()
