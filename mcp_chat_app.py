"""
Enhanced MCP Chat Interface - New Organized Entry Point
Professional modular structure following industry standards

Usage:
    streamlit run mcp_chat_app.py

Features:
- Modular component architecture
- Separated concerns (UI, services, utilities)
- Centralized configuration
- Clean CSS organization
- Professional code structure
"""

import sys
import os

# Add the project root to the Python path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and run the main application
from ui.main_app import main

if __name__ == "__main__":
    main()
