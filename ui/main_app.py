"""
Main MCP Chat Application
Orchestrates all UI components and services
"""

import streamlit as st

# Import local modules
from .config import APP_CONFIG
from .utils.session_manager import SessionManager
from .services.api_service import MCPBackendService
from .components.sidebar import Sidebar
from .components.chat_interface import ChatInterface

# Import external dependencies
from conversation_storage import ConversationStorage


class MCPChatApp:
    """Main application class that orchestrates the entire UI"""
    
    def __init__(self):
        """Initialize the application with all required services"""
        # Initialize services
        self.session_manager = SessionManager()
        self.conversation_storage = ConversationStorage()
        self.api_service = MCPBackendService()
        
        # Initialize components
        self.sidebar = Sidebar(self.conversation_storage, self.session_manager)
        self.chat_interface = ChatInterface(
            self.session_manager, 
            self.conversation_storage, 
            self.api_service
        )
    
    def run(self):
        """Main application entry point"""
        # Configure Streamlit
        self._configure_streamlit()
        
        # Initialize session state
        self.session_manager.initialize_session_state()
        
        # Initialize session early for proper conversation loading
        if not st.session_state.mcp_session_initialized:
            self.session_manager.get_or_create_session()
        
        # Render the application
        self._render_app()
    
    def _configure_streamlit(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=APP_CONFIG["title"],
            page_icon=APP_CONFIG["icon"],
            layout=APP_CONFIG["layout"],
            initial_sidebar_state=APP_CONFIG["sidebar_state"]
        )
    
    def _render_app(self):
        """Render the complete application"""
        # Render sidebar
        self.sidebar.render()
        
        # Render main chat interface
        self.chat_interface.render()


def main():
    """Application entry point"""
    app = MCPChatApp()
    app.run()


if __name__ == "__main__":
    main()
