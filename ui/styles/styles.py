"""
CSS Styles for MCP Chat Interface
Centralized styling following modern UI/UX principles
"""

# Main application styles
MAIN_APP_STYLES = """
<style>
/* Hide the default Streamlit header and footer */
header[data-testid="stHeader"] {
    height: 0px;
    background: transparent;
}

/* Root and body styling for full height */
html, body, #root {
    height: 100vh;
}

/* Main app container with flexbox */
.stApp {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

/* Main container styling - flexible to push footer down */
.main .block-container {
    max-width: 48rem;
    margin: 0 auto;
    padding: 1rem;
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: calc(100vh - 100px);
}

/* Chat content area - grows to fill space */
.chat-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 70vh;
}

/* Chat message styling */
.stChatMessage {
    margin: 1rem 0;
    background: transparent;
}

/* Fix input container styling */
.stChatInput {
    position: relative;
    background: transparent;
    padding: 1rem 0;
    z-index: 999;
}

.stChatInput > div {
    max-width: 48rem;
    margin: 0 auto;
    background: rgb(38, 39, 48);
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    min-height: 5rem;
}

/* Input field styling */
.stChatInput input {
    border: none;
    background: transparent;
    font-size: 1rem;
    padding: 0.75rem;
}

/* Custom title styling */
.chat-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 600;
    margin: 2rem 0 1rem 0;
    color: #1f2937;
}

/* Welcome message styling */
.welcome-container {
    text-align: center;
    margin: 4rem 0;
    padding: 2rem;
    min-height: 60vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.welcome-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 1rem;
}

.welcome-subtitle {
    font-size: 1rem;
    color: #6b7280;
    line-height: 1.5;
    max-width: 32rem;
    margin: 0 auto;
}

/* Better sidebar styling */
.css-1d391kg {
    background-color: #f8f9fa;
}

/* Responsive design */
@media (max-width: 768px) {
    .main .block-container {
        max-width: 100%;
        padding: 0.5rem;
        padding-bottom: 8rem;
    }
    
    .stChatInput > div {
        margin: 0 0.5rem;
        max-width: calc(100% - 1rem);
    }
}
</style>
"""

# Sidebar styles
SIDEBAR_STYLES = """
<style>
/* Modern sidebar styling similar to ChatGPT */
.css-1d391kg {
    background-color: #171717;
    color: white;
    border-right: 1px solid #333;
}

.sidebar-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #9ca3af;
    margin-bottom: 0.75rem;
    padding: 0 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* New chat button styling - clean and visible */
.css-1d391kg .stButton:first-child > button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 12px !important;
    background-color: transparent !important;
    border: 1px solid #4b5563 !important;
    border-radius: 8px !important;
    color: white !important;
    text-decoration: none !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    margin-bottom: 16px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    width: 100% !important;
    text-align: center !important;
}

.css-1d391kg .stButton:first-child > button:hover {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border-color: #6b7280 !important;
    color: white !important;
}

/* Chat history item styling - clean links */
.chat-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem;
    margin-bottom: 0.25rem;
    border-radius: 8px;
    background-color: transparent;
    color: white;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.875rem;
    width: 100%;
    border: none;
    outline: none;
}

.chat-item:hover {
    background-color: rgba(255, 255, 255, 0.1);
    color: white;
}

.chat-item.active {
    background-color: rgba(59, 130, 246, 0.3);
    color: white;
}

.chat-title-text {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-right: 0.5rem;
    color: white;
    text-align: left;
}

/* Section divider */
.section-divider {
    border-top: 1px solid #374151;
    margin: 1rem 0;
}

/* Settings expander styling */
.css-1d391kg .streamlit-expander {
    background-color: transparent;
    border: 1px solid #374151;
    border-radius: 6px;
}

.css-1d391kg .streamlit-expander .streamlit-expanderHeader {
    color: #9ca3af;
    font-size: 0.875rem;
}

/* Override Streamlit button styles in sidebar */
.css-1d391kg .stButton > button {
    background-color: transparent !important;
    border: none !important;
    color: #e5e7eb !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.2s !important;
    margin: 2px 0 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    outline: none !important;
}

.css-1d391kg .stButton > button:hover {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border: none !important;
    color: white !important;
    box-shadow: none !important;
}

.css-1d391kg .stButton > button:focus {
    box-shadow: none !important;
    border: none !important;
    outline: none !important;
    background-color: rgba(255, 255, 255, 0.1) !important;
}

.css-1d391kg .stButton > button:active {
    background-color: rgba(255, 255, 255, 0.15) !important;
    border: none !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Primary button for current conversation */
.css-1d391kg .stButton > button[kind="primary"] {
    background-color: rgba(59, 130, 246, 0.2) !important;
    color: white !important;
}

.css-1d391kg .stButton > button[kind="primary"]:hover {
    background-color: rgba(59, 130, 246, 0.3) !important;
}

/* Delete button styling */
.css-1d391kg .stButton > button[aria-label*="Delete"] {
    background-color: transparent !important;
    color: #ef4444 !important;
    font-size: 16px !important;
    padding: 8px !important;
    width: 32px !important;
    height: 32px !important;
    border-radius: 6px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.css-1d391kg .stButton > button[aria-label*="Delete"]:hover {
    background-color: rgba(239, 68, 68, 0.1) !important;
    color: #dc2626 !important;
}
</style>
"""

# Agent selection styles
AGENT_SELECTION_STYLES = """
<style>
.agent-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: white;
    transition: all 0.2s;
}

.agent-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
}

.agent-card.recommended {
    border-color: #10b981;
    background: #f0fdf4;
}

.agent-name {
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 0.5rem;
}

.agent-description {
    color: #6b7280;
    font-size: 0.875rem;
    line-height: 1.4;
    margin-bottom: 0.5rem;
}

.agent-confidence {
    font-size: 0.75rem;
    color: #9ca3af;
}
</style>
"""

# Progress and loading styles
PROGRESS_STYLES = """
<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-spinner {
    width: 20px;
    height: 20px;
    border: 2px solid #f3f3f3;
    border-top: 2px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    display: inline-block;
    margin-right: 10px;
}

.progress-container {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.5rem;
    background: #f8f9fa;
    border-radius: 6px;
    margin: 0.5rem 0;
}

.step-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.875rem;
    color: #374151;
}

.step-complete {
    color: #10b981;
}

.step-active {
    color: #3b82f6;
}
</style>
"""
