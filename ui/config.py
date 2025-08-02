"""
Configuration and constants for MCP Chat Interface
"""

# Application configuration
APP_CONFIG = {
    "title": "🚀 Enhanced MCP Assistant",
    "icon": "🤖",
    "layout": "wide",
    "sidebar_state": "expanded"
}

# Backend configuration
BACKEND_CONFIG = {
    "url": "http://localhost:8001",
    "timeout": 600,
    "stream_timeout": 120
}

# UI Messages
UI_MESSAGES = {
    "welcome": {
        "title": "👋 Welcome to Enhanced MCP Assistant",
        "subtitle": "Ask me anything about your data and I'll help you analyze it using multiple database agents with intelligent orchestration."
    },
    "processing": "⏳ Processing your request... Please wait",
    "input_placeholder": "Ask me anything about your data...",
    "new_chat": "✏️ New chat",
    "stop_processing": "🛑 Stop Processing",
    "no_conversations": "No conversations yet"
}

# Debug and logging
DEBUG_CONFIG = {
    "show_debug_by_default": False,
    "log_level": "INFO",
    "enable_performance_tracking": True
}

# Session configuration
SESSION_CONFIG = {
    "auto_load_recent": True,
    "max_history_length": 10,
    "conversation_summary_threshold": 4
}

# Streaming configuration
STREAMING_CONFIG = {
    "enabled": True,
    "fallback_on_error": True,
    "max_events": 1000,
    "event_timeout": 60  # Increased from 5 to 60 seconds for LLM processing
}
