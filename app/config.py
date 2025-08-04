"""
Configuration settings for the MCP server.
"""
import os


class Config:
    """Application configuration with timeout management for streaming."""
    
    # LLM Processing Timeouts
    LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "600"))  # 10 minutes for LLM calls
    LLM_HEARTBEAT_INTERVAL = int(os.getenv("LLM_HEARTBEAT_INTERVAL", "10"))  # Send heartbeat every 10s during LLM wait
    
    # Streaming Configuration 
    STREAM_TIMEOUT_SECONDS = int(os.getenv("STREAM_TIMEOUT_SECONDS", "120"))  # 120s timeout for getting events from queue (increased for LLM processing)
    STREAM_MAX_WAIT_SECONDS = int(os.getenv("STREAM_MAX_WAIT_SECONDS", "900"))  # 15 minutes total max wait
    STREAM_HEARTBEAT_INTERVAL = int(os.getenv("STREAM_HEARTBEAT_INTERVAL", "10"))  # Send heartbeat every 10s
    
    # Workflow settings
    MAX_WORKFLOW_STEPS = int(os.getenv("MAX_WORKFLOW_STEPS", "10"))
    WORKFLOW_TIMEOUT_SECONDS = int(os.getenv("WORKFLOW_TIMEOUT_SECONDS", "720"))  # 12 minutes total workflow timeout
    
    # Registry and Database (if needed)
    REGISTRY_BASE_URL = os.getenv("REGISTRY_BASE_URL", "http://localhost:8001") 
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mcp_server.db")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# Global config instance
config = Config()
