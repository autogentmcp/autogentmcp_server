from app.registry import fetch_agents_and_tools_from_registry
from langchain_ollama import ChatOllama
import json
import httpx
import re
import time

ollama = ChatOllama(model="qwen3:14b", base_url="http://localhost:11434", keep_alive="10m")  # 10 min keep-alive

# You can now use ollama as a LangChain-compatible chat model
# For LangGraph, you may need to wrap this or use as-is depending on your orchestration logic

# Caching logic for registry
_registry_cache = {
    "agents": None,
    "last_fetch": 0,
    "ttl": 300  # seconds (5 minutes)
}

# In-memory session context store
_session_contexts = {}

def summarize_history(history):
    if not history:
        return ""
    # Summarize all but the last 3 turns
    if len(history) <= 10:
        return None
    to_summarize = history[:-3]
    recent = history[-3:]
    summary_prompt = "/no_think\nSummarize the following conversation history for context, focusing on key facts, user goals, and important results.\n\n" + '\n'.join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in to_summarize])
    print("[route_query] Summarizing history:", summary_prompt)
    summary_response = ollama.invoke(summary_prompt)
    summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
    summary = re.sub(r'<think>[\s\S]*?</think>', '', summary, flags=re.IGNORECASE).strip()
    return summary, recent

def route_query(query: dict, session_id: str = "default"):
    """Route a user query to the best agent/tool using LangChain Ollama and invoke the endpoint, maintaining context by session_id."""
    agents = fetch_agents_and_tools_from_registry()
    # Retrieve or initialize session history
    history = _session_contexts.get(session_id, [])
    # Summarize history if too long
    summary = None
    if len(history) > 10:
        summary, recent = summarize_history(history)
        history = recent
    # Step 1: Agent selection
    agent_list_str = '\n'.join([f"- {name}: {info['description']} (base: {info['base_domain']})" for name, info in agents.items()])
    history_str = ''
    if summary:
        history_str += f"Summary of previous conversation:\n{summary}\n"
    history_str += '\n'.join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history])
    agent_prompt = f"""/no_think\nYou are an expert agent selector. Maintain context across turns.\n\nConversation history:\n{history_str}\n\nAvailable agents:\n{agent_list_str}\n\nUser query: {query['query']}\n\nRespond ONLY with a valid JSON object, with NO extra text, markdown, or explanation. The JSON must be on the first line of your response.\n\nExample:\n{{"agent": "<agent_name>", "reason": "<short explanation>"}}\n\nNow, respond with your selection:\n"""
    print("[route_query] Agent selection prompt:\n", agent_prompt)
    agent_response = ollama.invoke(agent_prompt)
    agent_llm_response = agent_response.content
    print("[route_query] Raw agent LLM response:\n", agent_llm_response)
    agent_llm_response_clean = re.sub(r'<think>[\s\S]*?</think>', '', agent_llm_response, flags=re.IGNORECASE).strip()
    print("[route_query] Cleaned agent LLM response (no <think>):\n", agent_llm_response_clean)
    selected_agent_name = None
    agent_reason = None
    try:
        parsed = json.loads(agent_llm_response_clean)
        selected_agent_name = parsed.get("agent")
        agent_reason = parsed.get("reason")
        print(f"[route_query] Parsed agent: {selected_agent_name}, reason: {agent_reason}")
    except Exception as e:
        print(f"[route_query] Agent JSON parsing error: {e}")
        selected_agent_name = None
        agent_reason = None

    # Step 2: Tool selection for the selected agent
    selected_agent = agents.get(selected_agent_name)
    selected_tool_name = None
    reason = None
    call_result = None
    if selected_agent:
        tools = selected_agent['tools']
        tool_list_str = '\n'.join([f"- {t['name']}: {t['description']}" for t in tools])
        tool_prompt = f"""/no_think
You are an expert tool selector. The selected agent is: {selected_agent_name} ({selected_agent['description']}).

Base domain: {selected_agent['base_domain']}

Available tools for this agent:
{tool_list_str}

User query: {query['query']}

If a tool requires path parameters (e.g., /users/{{id}}), extract the value from the user query and substitute it into the endpoint.

Respond ONLY with a valid JSON object, with NO extra text, markdown, or explanation.
The "tool" value MUST be copied exactly from the tool names in the list above.
The JSON must be on the first line of your response.

Example:
{{"tool": "<tool_name>", "reason": "<short explanation>", "resolved_endpoint": "https://example.com/users/123"}}

Now, respond with your selection:
"""
        print("[route_query] Tool selection prompt:\n", tool_prompt)
        response = ollama.invoke(tool_prompt)
        llm_response = response.content
        print("[route_query] Raw tool LLM response:\n", llm_response)
        llm_response_clean = re.sub(r'<think>[\s\S]*?</think>', '', llm_response, flags=re.IGNORECASE).strip()
        print("[route_query] Cleaned tool LLM response (no <think>):\n", llm_response_clean)
        # Directly parse the cleaned LLM response as JSON
        selected_tool_name = None
        reason = None
        resolved_endpoint = None
        try:
            parsed = json.loads(llm_response_clean)
            selected_tool_name = parsed.get("tool")
            reason = parsed.get("reason")
            resolved_endpoint = parsed.get("resolved_endpoint")
            print(f"[route_query] Parsed tool: {selected_tool_name}, reason: {reason}, resolved_endpoint: {resolved_endpoint}")
        except Exception as e:
            print(f"[route_query] Tool JSON parsing error: {e}")
        selected_tool = next((t for t in tools if t["name"] == selected_tool_name), None)
        # Determine method dynamically from tool metadata or LLM output
        method = selected_tool["method"].upper() if selected_tool and "method" in selected_tool else "GET"
        url = resolved_endpoint if resolved_endpoint else selected_agent['base_domain'] + selected_tool['endpoint_uri'] if selected_tool else None
        # Prepare request params, body, and headers if present in LLM output
        body_params = parsed.get("body_params", {}) if 'parsed' in locals() else {}
        query_params = parsed.get("query_params", {}) if 'parsed' in locals() else {}
        headers = parsed.get("headers", {}) if 'parsed' in locals() else {}
        call_result = None
        try:
            if method == "GET":
                r = httpx.get(url, params=query_params, headers=headers)
            elif method == "POST":
                r = httpx.post(url, params=query_params, json=body_params, headers=headers)
            elif method == "PUT":
                r = httpx.put(url, params=query_params, json=body_params, headers=headers)
            elif method == "DELETE":
                r = httpx.delete(url, params=query_params, headers=headers)
            else:
                r = httpx.request(method, url, params=query_params, json=body_params, headers=headers)
            call_result = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
            print(f"[route_query] Endpoint response: {call_result}")
        except Exception as e:
            call_result = f"Error calling endpoint: {e}"
            print(f"[route_query] Error calling endpoint: {e}")
    else:
        print("[route_query] No valid agent selected or found.")

    # Final LLM answer formatting step
    final_prompt = f"""/no_think\nYou are an expert assistant.\n\nUser query: {query['query']}\n\nRaw result from the service: {call_result}\n\nInstructions:\n- If the user is asking for a list (e.g., 'Give me list of users'), format the result as a markdown table.\n- Otherwise, summarize or present the result in the most appropriate and helpful way.\n- Do not add extra commentary or markdown unless formatting a table.\n"""
    print("[route_query] Final answer prompt:\n", final_prompt)
    final_response = ollama.invoke(final_prompt)
    final_answer = final_response.content if hasattr(final_response, 'content') else str(final_response)
    # Remove <think>...</think> blocks from the final answer
    final_answer = re.sub(r'<think>[\s\S]*?</think>', '', final_answer, flags=re.IGNORECASE).strip()
    print("[route_query] Final LLM answer:\n", final_answer)

    # After getting final_answer, update session context
    if session_id:
        history.append({"user": query["query"], "assistant": final_answer})
        if summary:
            # Store summary as a special turn
            _session_contexts[session_id] = [{"user": "__summary__", "assistant": summary}] + history[-10:]
        else:
            _session_contexts[session_id] = history[-10:]
    return {
        "query": query["query"],
        "agents": list(agents.keys()),
        "selected_agent": selected_agent_name,
        "agent_reason": agent_reason,
        "selected_tool": selected_tool_name,
        "reason": reason,
        "call_result": call_result,
        "final_answer": final_answer
    }
