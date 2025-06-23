# MCP LangGraph Server

A FastAPI-based MCP server that dynamically discovers agents/tools from a registry and uses LangGraph for LLM-driven tool selection and orchestration.

## Features
- Fetches application and endpoint metadata from the MCP registry
- Dynamically builds agent/tool objects
- Integrates with LangGraph for LLM-based tool selection
- FastAPI endpoints for health, registry sync, and user query

## Quickstart

1. Install dependencies (in your activated venv):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

## Project Structure
- `app/main.py`: FastAPI entrypoint
- `app/registry.py`: Registry sync logic
- `app/agents.py`: Dynamic agent/tool creation
- `app/langgraph_router.py`: LangGraph orchestration logic

## TODO
- Implement registry polling and agent creation
- Integrate with LangGraph for LLM-driven routing
- Add user query endpoint
