"""
Intent analysis and conversation state management
"""

import json
from typing import Dict, List, Any
from ..models import ExecutionContext, IntentAnalysisResult
from app.multimode_llm_client import get_global_llm_client, TaskType
from app.registry import fetch_agents_and_tools_from_registry

class IntentAnalyzer:
    """Analyzes user intent and manages conversation state"""
    
    def __init__(self):
        self.llm_client = get_global_llm_client()
    
    async def analyze_intent(self, context: ExecutionContext, conversation_context: str) -> IntentAnalysisResult:
        """Analyze intent with LLM-driven conversation state management"""
        
        print(f"[IntentAnalyzer] Analyzing intent for query: '{context.user_query}'")
        print(f"[IntentAnalyzer] Conversation history length: {len(context.conversation_history)}")
        
        # Get all agents for context
        agents = fetch_agents_and_tools_from_registry()
        
        print(f"[IntentAnalyzer] Found {len(agents)} agents in cache:")
        for agent_id, agent in agents.items():
            agent_type = agent.get("agent_type", "unknown")
            name = agent.get("name", "Unknown")
            print(f"  - {agent_id}: {name} ({agent_type})")
        
        # Build simple agent list for LLM
        agent_list = []
        for agent_id, agent in agents.items():
            if agent.get("agent_type") in ["data_agent", "application"]:
                agent_info = {
                    "id": agent_id,
                    "name": agent.get("name"),
                    "type": agent.get("agent_type"),
                    "description": agent.get("description", "")
                }
                agent_list.append(agent_info)
        
        print(f"[IntentAnalyzer] Built agent list for LLM with {len(agent_list)} agents")
        
        prompt = f"""
You are a conversation state manager. Analyze the current user query in context and determine the appropriate next action.

Current Query: "{context.user_query}"

{conversation_context}

Available Agents:
{json.dumps(agent_list, indent=2)}

DO NOT refer to anything other than above agents. Consider above agents your whole world. You can only entertain queries related to these agents and greetings.
If the user is asking for general knowledge like where is something situated if you know you can answer it, otherwise you can say "I don't know" or "I can't help with that".

INSTRUCTIONS:
Analyze the conversation state and determine what should happen next. Consider:

1. **Previous agent usage** - Look at which agents were used previously and their success/failure
2. **If this is a continuation of a previous clarification request** - look for agent selection responses
3. **If this is a new data request** - determine if it needs clarification or can be executed directly  
4. **If this is a greeting or capability request** - handle appropriately
5. **If this is outside scope** - handle as general
6. **Agent preferences** - If user has had success with certain agents, prefer them for similar queries

Respond with this EXACT JSON structure:
{{
    "conversation_state": "new_request|clarification_response|continuing_conversation",
    "action": "execute|ask_clarification|greeting|capabilities|general",
    "confidence": 0.0-1.0,
    "message": "Response message to show user",
    "reasoning": "Brief explanation of decision including previous agent usage",
    "execution_plan": {{
        "strategy": "single|sequential|parallel",
        "selected_agent_id": "agent_id_if_selected",
        "query_to_execute": "actual query to run",
        "previous_agent_context": "information about previously used agents if relevant"
    }},
    "clarification_options": [
        {{
            "agent_id": "id",
            "name": "agent name",
            "description": "what this agent provides",
            "best_for": "scenarios where this excels",
            "previous_usage": "success|failure|never_used"
        }}
    ]
}}

DECISION RULES:
- If conversation history shows previous clarification AND current query looks like a selection → "clarification_response" + "execute"
- If new data request with multiple possible agents → "new_request" + "ask_clarification" 
- If new data request with clear single agent match → "new_request" + "execute"
- If greeting → "new_request" + "greeting"
- If asking about capabilities → "new_request" + "capabilities"
- If unclear or out of scope → "new_request" + "general"
- **If previous agents succeeded for similar queries, prefer them → higher confidence for "execute"**
- **If previous agents failed, avoid them or explain why trying again → mention in reasoning**

EXECUTION STRATEGY RULES:
- **single**: One agent can handle the request completely (e.g., "show sales data", "get inventory")
- **sequential**: Multiple agents needed in order, where later agents depend on earlier results (e.g., "get sales data then analyze trends", "fetch inventory then calculate turnover")
- **parallel**: Multiple independent agents for comprehensive analysis (e.g., "compare data across systems", "get both sales and inventory data")

For clarification_response: extract the selected agent and use the ORIGINAL data request from history
For execute: provide execution_plan with selected agent and query (consider previous agent success/failure)
For ask_clarification: provide clarification_options with 2-4 relevant agents (mark previous usage status)
"""
        
        print(f"[IntentAnalyzer] Sending prompt to LLM")
        response = self.llm_client.invoke_with_json_response(prompt, task_type=TaskType.INTENT_ANALYSIS)
        
        if not response:
            return IntentAnalysisResult(
                conversation_state="new_request",
                action="general", 
                confidence=0.5,
                message="How can I help you today?",
                reasoning="No response from LLM"
            )
        
        print(f"[IntentAnalyzer] LLM decided action: {response.get('action')}, state: {response.get('conversation_state')}")
        
        # Debug execution plan
        execution_plan = response.get("execution_plan")
        if execution_plan:
            strategy = execution_plan.get("strategy", "unknown")
            agent_id = execution_plan.get("selected_agent_id", "none")
            print(f"[IntentAnalyzer] Execution Strategy: {strategy}, Selected Agent: {agent_id}")
        
        return IntentAnalysisResult(
            conversation_state=response.get("conversation_state", "new_request"),
            action=response.get("action", "general"),
            confidence=response.get("confidence", 0.5),
            message=response.get("message", "How can I help you?"),
            reasoning=response.get("reasoning", ""),
            execution_plan=response.get("execution_plan"),
            clarification_options=response.get("clarification_options", [])
        )
