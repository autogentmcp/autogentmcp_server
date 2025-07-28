"""
Streamlit Browser Fingerprinting Helper
Provides easy integration for browser fingerprinting in Streamlit apps
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import uuid
from datetime import datetime


class StreamlitSessionManager:
    """Helper class for managing browser fingerprinting and sessions in Streamlit."""
    
    def __init__(self, backend_url: str = "http://localhost:8001"):
        self.backend_url = backend_url
        self.js_file_path = Path(__file__).parent / "streamlit_browser_fingerprint.js"
    
    def get_fingerprint_js(self) -> str:
        """Get the JavaScript code for browser fingerprinting."""
        try:
            with open(self.js_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback JavaScript if file not found
            return '''
            function generateBrowserFingerprint() {
                return {
                    user_agent: navigator.userAgent,
                    screen_resolution: screen.width + "x" + screen.height,
                    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                    language: navigator.language,
                    platform: navigator.platform,
                    timestamp: Date.now()
                };
            }
            
            async function getOrCreateSession(backendUrl) {
                const fingerprint = generateBrowserFingerprint();
                
                try {
                    const response = await fetch(backendUrl + '/enhanced/session/create', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            browser_fingerprint: fingerprint
                        })
                    });
                    
                    if (response.ok) {
                        const sessionData = await response.json();
                        window.currentSession = sessionData;
                        return sessionData;
                    } else {
                        throw new Error('Backend session creation failed');
                    }
                } catch (error) {
                    console.warn('Failed to create session via backend:', error);
                    // Fallback to local session
                    const sessionId = 'streamlit_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                    const fallbackSession = {
                        session_id: sessionId, 
                        user_id: 'anonymous',
                        is_fallback: true,
                        fingerprint: fingerprint
                    };
                    window.currentSession = fallbackSession;
                    return fallbackSession;
                }
            }
            
            window.getOrCreateSession = getOrCreateSession;
            window.generateBrowserFingerprint = generateBrowserFingerprint;
            '''
    
    def initialize_session(self) -> Dict[str, Any]:
        """
        Initialize browser fingerprinting and session management in Streamlit.
        Call this once at the beginning of your Streamlit app.
        """
        # Check if session is already initialized
        if 'mcp_session_initialized' in st.session_state and 'mcp_session_data' in st.session_state:
            if st.session_state.mcp_session_data is not None:
                return st.session_state.mcp_session_data
        
        # Create the JavaScript component
        js_code = self.get_fingerprint_js()
        
        # HTML wrapper that includes session initialization
        html_wrapper = f'''
        <script>
        {js_code}
        
        // Initialize session when component loads
        (async function() {{
            try {{
                const sessionData = await getOrCreateSession('{self.backend_url}');
                console.log('Session initialized:', sessionData);
                
                // Store in a way Streamlit can access
                window.parent.streamlitSessionData = sessionData;
                
                // Try to communicate with Streamlit
                if (window.parent && window.parent.postMessage) {{
                    window.parent.postMessage({{
                        type: 'sessionInitialized',
                        data: sessionData
                    }}, '*');
                }}
                
                // Also store in global for direct access
                window.mcpSessionData = sessionData;
                
            }} catch (error) {{
                console.error('Session initialization failed:', error);
                // Fallback session
                const fallbackSession = {{
                    session_id: 'streamlit_fallback_' + Date.now(),
                    user_id: 'anonymous',
                    is_fallback: true,
                    error: error.message
                }};
                window.parent.streamlitSessionData = fallbackSession;
                window.mcpSessionData = fallbackSession;
            }}
        }})();
        </script>
        <div style="display:none;">Session manager initialized</div>
        '''
        
        # Render the component (hidden)
        components.html(html_wrapper, height=50)
        
        # Wait a moment for JavaScript to execute
        time.sleep(0.5)
        
        # Try to get session data from JavaScript or create fallback
        session_data = self._get_or_create_fallback_session()
        
        # Ensure we have valid session data
        if session_data is None:
            session_data = self._create_emergency_fallback_session()
        
        # Store in Streamlit session state
        st.session_state.mcp_session_initialized = True
        st.session_state.mcp_session_data = session_data
        
        return session_data
    
    def _get_or_create_fallback_session(self) -> Optional[Dict[str, Any]]:
        """Create a fallback session if JavaScript fingerprinting fails."""
        # Try to create session with basic info
        try:
            # Simple fingerprint based on Streamlit session
            basic_fingerprint = {
                "user_agent": "streamlit_user",
                "timestamp": int(time.time() * 1000),
                "streamlit_session_id": str(uuid.uuid4()),
                "is_streamlit_fallback": True
            }
            
            # Try to create session via API
            response = requests.post(
                f"{self.backend_url}/enhanced/session/create",
                json={"browser_fingerprint": basic_fingerprint},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "session_id": data.get("session_id"),
                    "user_id": data.get("user_id", "anonymous"),
                    "is_fallback": False,
                    "source": "api_created"
                }
            
        except Exception as e:
            # Silently handle connection errors
            pass
        
        # Return None to trigger emergency fallback
        return None
    
    def _create_emergency_fallback_session(self) -> Dict[str, Any]:
        """Create an emergency fallback session when all else fails."""
        fallback_session_id = f"streamlit_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        return {
            "session_id": fallback_session_id,
            "user_id": "anonymous",
            "is_fallback": True,
            "source": "emergency_fallback",
            "created_at": datetime.now().isoformat()
        }
    
    def get_session_id(self) -> str:
        """Get the current session ID."""
        if 'mcp_session_data' not in st.session_state or st.session_state.mcp_session_data is None:
            self.initialize_session()
        
        session_data = st.session_state.get('mcp_session_data', {})
        return session_data.get("session_id", "unknown")
    
    def create_new_conversation(self) -> Optional[str]:
        """Create a new conversation within the current session."""
        session_id = self.get_session_id()
        
        try:
            response = requests.post(
                f"{self.backend_url}/enhanced/session/{session_id}/new_conversation",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("new_conversation_id")
        except Exception as e:
            st.error(f"Failed to create new conversation: {e}")
        
        return None
    
    def get_conversations(self) -> list:
        """Get all conversations for the current session."""
        session_id = self.get_session_id()
        
        try:
            response = requests.get(
                f"{self.backend_url}/enhanced/session/{session_id}/conversations",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("conversations", [])
        except Exception as e:
            st.error(f"Failed to get conversations: {e}")
        
        return []
    
    def display_session_info(self):
        """Display session information for debugging."""
        if 'mcp_session_data' in st.session_state and st.session_state.mcp_session_data is not None:
            session_data = st.session_state.mcp_session_data
            
            with st.expander("🔍 Session Information", expanded=False):
                st.json(session_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refresh Session"):
                        # Clear session state and reinitialize
                        if 'mcp_session_initialized' in st.session_state:
                            del st.session_state.mcp_session_initialized
                        if 'mcp_session_data' in st.session_state:
                            del st.session_state.mcp_session_data
                        st.rerun()
                
                with col2:
                    if st.button("💬 New Conversation"):
                        new_conv_id = self.create_new_conversation()
                        if new_conv_id:
                            st.success(f"New conversation created: {new_conv_id}")
                        else:
                            st.error("Failed to create new conversation")


# Global instance for easy import
session_manager = StreamlitSessionManager()


def initialize_mcp_session(backend_url: str = "http://localhost:8001") -> Dict[str, Any]:
    """
    Quick function to initialize MCP session management in Streamlit.
    
    Usage:
        import streamlit as st
        from streamlit_session_helper import initialize_mcp_session
        
        session_data = initialize_mcp_session()
        st.write(f"Session ID: {session_data['session_id']}")
    """
    manager = StreamlitSessionManager(backend_url)
    return manager.initialize_session()


def get_mcp_session_id(backend_url: str = "http://localhost:8001") -> str:
    """
    Quick function to get the current MCP session ID.
    
    Usage:
        from streamlit_session_helper import get_mcp_session_id
        
        session_id = get_mcp_session_id()
    """
    manager = StreamlitSessionManager(backend_url)
    return manager.get_session_id()


# Example usage function
def demo_session_management():
    """Demo function showing how to use the session manager."""
    st.title("🚀 MCP Session Management Demo")
    
    # Initialize session
    session_data = initialize_mcp_session()
    
    st.success(f"Session initialized: {session_data['session_id']}")
    
    # Show session info
    session_manager.display_session_info()
    
    # Test query with session
    st.subheader("Test Query with Session")
    query = st.text_input("Enter a test query:")
    
    if st.button("Send Query") and query:
        try:
            response = requests.post(
                "http://localhost:8001/enhanced/query",
                json={
                    "query": query,
                    "session_id": session_data["session_id"]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success("Query successful!")
                st.json(result)
            else:
                st.error(f"Query failed: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error sending query: {e}")


if __name__ == "__main__":
    demo_session_management()
