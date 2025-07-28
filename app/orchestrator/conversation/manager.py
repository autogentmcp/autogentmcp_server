"""
Conversation memory and context management
"""

import time
from typing import Dict, List, Any
from ..models import ConversationTurn, ExecutionContext

class ConversationManager:
    """Manages conversation history and context for sessions"""
    
    def __init__(self):
        self._conversation_memory: Dict[str, List[ConversationTurn]] = {}
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for session"""
        turns = self._conversation_memory.get(session_id, [])
        return [
            {
                "timestamp": turn.timestamp,
                "query": turn.query,
                "response": turn.response,
                "execution_results": turn.execution_results,
                "agents_used": turn.agents_used,
                "clarification_options": turn.clarification_options
            }
            for turn in turns
        ]
    
    def save_turn(self, session_id: str, query: str, response: str, 
                  execution_results: List[Dict[str, Any]] = None,
                  clarification_options: List[Dict[str, Any]] = None):
        """Save conversation turn with detailed context"""
        if session_id not in self._conversation_memory:
            self._conversation_memory[session_id] = []
        
        # Extract agent names from successful results
        agents_used = []
        if execution_results:
            agents_used = [r.get("agent_name") for r in execution_results if r.get("success")]
        
        # Create conversation turn
        turn = ConversationTurn(
            timestamp=time.time(),
            query=query,
            response=response,
            execution_results=execution_results or [],
            agents_used=agents_used,
            clarification_options=clarification_options or []
        )
        
        self._conversation_memory[session_id].append(turn)
        
        # Keep only last 10 turns
        if len(self._conversation_memory[session_id]) > 10:
            self._conversation_memory[session_id] = self._conversation_memory[session_id][-10:]
    
    def build_conversation_context(self, context: ExecutionContext) -> str:
        """Build conversation context string with agent usage history"""
        history_text = ""
        if context.conversation_history:
            recent_turns = context.conversation_history[-3:]  # Last 3 turns
            
            if recent_turns:
                history_text = "RECENT CONVERSATION:\\n"
                for i, turn in enumerate(recent_turns, 1):
                    history_text += f"Turn {i} - User: {turn['query'][:100]}...\\n"
                    history_text += f"Turn {i} - Assistant: {turn['response'][:150]}...\\n"
                    
                    # Add agent usage information
                    agents_used = turn.get('agents_used', [])
                    if agents_used:
                        history_text += f"Turn {i} - Agents Used: {', '.join(agents_used)}\\n"
                    
                    # Add execution results summary
                    execution_results = turn.get('execution_results', [])
                    if execution_results:
                        successful_agents = [r for r in execution_results if r.get('success')]
                        total_rows = sum(r.get('row_count', 0) for r in successful_agents)
                        if total_rows > 0:
                            history_text += f"Turn {i} - Data Retrieved: {total_rows} rows\\n"
                    
                    # Add clarification options if any were provided
                    clarification_options = turn.get('clarification_options', [])
                    if clarification_options:
                        agent_names = [opt.get('name', 'Unknown') for opt in clarification_options]
                        history_text += f"Turn {i} - Clarification Offered: {', '.join(agent_names)}\\n"
                    
                    history_text += "\\n"
        
        return history_text
