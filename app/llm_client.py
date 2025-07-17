"""
LLM client wrapper for consistent LLM interactions.
"""
from langchain_ollama import ChatOllama
import re
import json
from typing import Dict, Any, Optional

class LLMClient:
    """Wrapper for LLM interactions with consistent prompting and parsing."""
    
    def __init__(self, model: str = "qwen3:14b", base_url: str = "http://localhost:11434"):
        self.ollama = ChatOllama(model=model, base_url=base_url, keep_alive="10m")
    
    def invoke_with_json_response(self, prompt: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Invoke LLM with a prompt expecting JSON response.
        
        Args:
            prompt: The main prompt
            context: Additional context to include
            
        Returns:
            Parsed JSON response or None if parsing fails
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[LLMClient] Prompt:\n{full_prompt}")
        
        response = self.ollama.invoke(full_prompt)
        llm_response = response.content if hasattr(response, 'content') else str(response)
        print(f"[LLMClient] Raw response:\n{llm_response}")
        
        # Clean response by removing <think> blocks
        clean_response = re.sub(r'<think>[\s\S]*?</think>', '', llm_response, flags=re.IGNORECASE).strip()
        print(f"[LLMClient] Cleaned response:\n{clean_response}")
        
        try:
            parsed = json.loads(clean_response)
            return parsed
        except json.JSONDecodeError as e:
            print(f"[LLMClient] JSON parsing error: {e}")
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return parsed
                except json.JSONDecodeError:
                    pass
            return None
    
    def invoke_with_text_response(self, prompt: str, context: str = "", allow_diagrams: bool = True) -> str:
        """
        Invoke LLM with a prompt expecting text response.
        
        Args:
            prompt: The main prompt
            context: Additional context to include
            allow_diagrams: Whether to preserve diagrams in the response
            
        Returns:
            Processed text response
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        print(f"[LLMClient] Prompt:\n{full_prompt}")
        
        response = self.ollama.invoke(full_prompt)
        llm_response = response.content if hasattr(response, 'content') else str(response)
        print(f"[LLMClient] Raw response:\n{llm_response}")
        
        if allow_diagrams:
            # Extract diagrams from <think> blocks and convert to markdown
            processed_response = self._extract_and_replace_diagrams(llm_response)
        else:
            # Remove <think> blocks entirely
            processed_response = re.sub(r'<think>[\s\S]*?</think>', '', llm_response, flags=re.IGNORECASE).strip()
        
        print(f"[LLMClient] Processed response:\n{processed_response}")
        return processed_response
    
    def _extract_and_replace_diagrams(self, text: str) -> str:
        """Extract diagrams from <think> blocks and convert to markdown code blocks."""
        def repl(match):
            content = match.group(1)
            # Try to detect mermaid or plantuml
            if 'mermaid' in content.lower():
                return f"```mermaid\n{content}\n```"
            elif '@startuml' in content.lower():
                return f"```plantuml\n{content}\n```"
            elif 'graph' in content.lower() and ('-->' in content or '->' in content):
                return f"```mermaid\n{content}\n```"
            else:
                return content  # fallback: just insert as-is
        
        # Replace <think>...</think> with code blocks if diagram detected
        return re.sub(r'<think>([\s\S]*?)</think>', repl, text, flags=re.IGNORECASE)
    
    def create_agent_selection_prompt(self, query: str, agents: Dict[str, Any], history: str = "") -> str:
        """Create a prompt for agent selection."""
        agent_list_str = '\n'.join([f"- {name}: {info['description']} (base: {info['base_domain']})" for name, info in agents.items()])
        
        return f"""/no_think
You are an expert agent selector. Maintain context across turns.

Conversation history:
{history}

Available agents:
{agent_list_str}

User query: {query}

Respond ONLY with a valid JSON object, with NO extra text, markdown, or explanation. The JSON must be on the first line of your response.

Example:
{{"agent": "<agent_name>", "reason": "<short explanation>"}}

Now, respond with your selection:
"""
    
    def create_tool_selection_prompt(self, query: str, agent_name: str, agent_info: Dict[str, Any]) -> str:
        """Create a prompt for tool selection."""
        tools = agent_info['tools']
        tool_list_str = '\n'.join([f"- {t['name']}: {t['description']}" for t in tools])
        
        return f"""/no_think
You are an expert tool selector. The selected agent is: {agent_name} ({agent_info['description']}).

Base domain: {agent_info['base_domain']}

Available tools for this agent:
{tool_list_str}

User query: {query}

If a tool requires path parameters (e.g., /users/{{id}}), extract the value from the user query and substitute it into the endpoint.

Respond ONLY with a valid JSON object, with NO extra text, markdown, or explanation.
The "tool" value MUST be copied exactly from the tool names in the list above.
The JSON must be on the first line of your response.

Example:
{{"tool": "<tool_name>", "reason": "<short explanation>", "resolved_endpoint": "https://example.com/users/123", "query_params": {{}}, "body_params": {{}}, "headers": {{}}}}

Now, respond with your selection:
"""
    
    def create_final_answer_prompt(self, query: str, call_result: Any) -> str:
        """Create a prompt for final answer formatting."""
        return f"""/no_think
You are an expert assistant.

User query: {query}

Raw result from the service: {call_result}

Instructions:
- If the user is asking for a list (e.g., 'Give me list of users'), format the result as a markdown table.
- If the answer includes a diagram, output it as a markdown code block with the correct language tag (e.g., ```mermaid for Mermaid diagrams, ```plantuml for PlantUML diagrams).
- Do NOT use <think> blocks for diagrams.
- Otherwise, summarize or present the result in the most appropriate and helpful way.
- Do not add extra commentary or markdown unless formatting a table or diagram.
"""

# Global LLM client instance
llm_client = LLMClient()
