# MCP Registry Server

A FastAPI-based Model Context Protocol (MCP) server that dynamically fetches agent/tool metadata from a registry and uses LangGraph with Ollama for intelligent tool selection and orchestration.

## Features

- **Dynamic Agent/Tool Loading**: Fetches metadata from external registry
- **LLM-driven Tool Selection**: Uses Ollama to select the best agent and tool for each query
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Authentication Support**: Handles multiple authentication types (API Key, Bearer, Basic, OAuth2)
- **Session Management**: Maintains conversation context across multiple turns
- **Diagram Support**: Renders Mermaid and PlantUML diagrams in responses
- **Background Registry Sync**: Automatically refreshes agent metadata every 5 minutes

## Architecture

The server is built with a modular architecture:

```
app/
├── main.py                 # FastAPI application entry point
├── langgraph_router.py     # Main orchestration logic
├── auth_handler.py         # Authentication management
├── llm_client.py           # LLM interaction wrapper
├── session_manager.py      # Session context management
├── tool_selector.py        # Agent and tool selection
├── endpoint_invoker.py     # HTTP endpoint invocation
├── registry.py             # Registry sync and caching
└── agents.py               # Agent/tool data structures
```

## Authentication

The server supports multiple authentication methods:

### Supported Types

1. **API Key** (`apiKey`)
   - Header-based: `X-API-Key: your-key`
   - Query parameter: `?api_key=your-key`
   - Custom format: `Authorization: Bearer {key}`

2. **Bearer Token** (`bearer`)
   - `Authorization: Bearer your-token`

3. **Basic Auth** (`basic`)
   - `Authorization: Basic base64(username:password)`

4. **OAuth2** (`oauth2`)
   - `Authorization: Bearer access-token`

### Configuration

Authentication is configured through registry metadata:

```json
{
  "security": {
    "type": "apiKey",
    "location": "header",
    "header_name": "X-API-Key",
    "format": "direct"
  }
}
```

### Credential Management

#### Environment Variables
```bash
export MCP_DEFAULT_API_KEY="your-default-key"
export MCP_DEFAULT_BEARER_TOKEN="your-default-token"
export MCP_DEFAULT_BASIC_USER="username"
export MCP_DEFAULT_BASIC_PASS="password"
```

#### Configuration File
Copy `auth_config.json.example` to `auth_config.json` and update:

```json
{
  "credentials": {
    "jsonplaceholder_api_key": "your-api-key-here",
    "github_api_key": "your-github-token-here"
  },
  "agent_specific": {
    "JSONPlaceholder": {
      "api_key": "specific-key-for-jsonplaceholder"
    }
  }
}
```

#### Runtime API
```bash
# Set credentials via API
curl -X POST "http://localhost:8000/auth/set_credential" \
  -H "Content-Type: application/json" \
  -d '{"key": "github_api_key", "value": "github_pat_xxxxx"}'
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Ollama server:
```bash
ollama serve
ollama pull qwen3:14b
```

3. Configure authentication (optional):
```bash
cp auth_config.json.example auth_config.json
# Edit auth_config.json with your credentials
```

4. Start the server:
```bash
uvicorn app.main:app --reload
```

## Usage

### Query Endpoint

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Get user with ID 123",
    "session_id": "user-session-1"
  }'
```

### Session Management

```bash
# Get all sessions
curl "http://localhost:8000/sessions"

# Get specific session
curl "http://localhost:8000/sessions/user-session-1"

# Clear session
curl -X DELETE "http://localhost:8000/sessions/user-session-1"
```

### Registry Management

```bash
# Manually sync registry
curl -X POST "http://localhost:8000/sync_registry"
```

## API Endpoints

- `GET /health` - Health check
- `POST /query` - Process user query
- `POST /sync_registry` - Manually sync registry
- `POST /auth/set_credential` - Set authentication credential
- `GET /sessions` - List all sessions
- `GET /sessions/{session_id}` - Get session details
- `DELETE /sessions/{session_id}` - Clear session

## Registry Format

The server expects registry data in this format:

```json
{
  "agents": {
    "JSONPlaceholder": {
      "description": "JSONPlaceholder API for testing",
      "base_domain": "https://jsonplaceholder.typicode.com",
      "security": {
        "type": "apiKey",
        "location": "header",
        "header_name": "X-API-Key"
      },
      "tools": [
        {
          "name": "get_user",
          "description": "Get user by ID",
          "endpoint_uri": "/users/{id}",
          "method": "GET"
        }
      ]
    }
  }
}
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black app/
flake8 app/
```

### Building
```bash
docker build -t mcp-registry-server .
```

## Configuration

### Environment Variables

- `MCP_DEFAULT_API_KEY` - Default API key
- `MCP_DEFAULT_BEARER_TOKEN` - Default bearer token
- `MCP_DEFAULT_BASIC_USER` - Default basic auth username
- `MCP_DEFAULT_BASIC_PASS` - Default basic auth password
- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL` - Ollama model name (default: qwen3:14b)

### Registry Configuration

The registry URL is configured in `app/registry.py`. Update the `REGISTRY_URL` constant to point to your registry endpoint.

## Troubleshooting

### Common Issues

1. **LLM not responding**: Check Ollama server is running and model is available
2. **Authentication failures**: Verify credentials in auth_config.json or environment variables
3. **Registry sync errors**: Check network connectivity and registry URL
4. **JSON parsing errors**: Enable debug logging to see LLM responses

### Debug Logging

The server includes extensive logging. Look for log messages prefixed with:
- `[AuthHandler]` - Authentication issues
- `[LLMClient]` - LLM interaction problems
- `[SessionManager]` - Session management
- `[ToolSelector]` - Agent/tool selection
- `[EndpointInvoker]` - HTTP request issues

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License
