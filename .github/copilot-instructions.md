<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

This is an MCP server project using FastAPI and LangGraph. Please use the https://github.com/modelcontextprotocol/create-python-server SDK as a reference for MCP server patterns and best practices.

- The server fetches agent/tool metadata from a registry and dynamically builds tool objects.
- It uses LangGraph for LLM-driven tool selection and orchestration.
- Endpoints: /health, /sync_registry, /query
