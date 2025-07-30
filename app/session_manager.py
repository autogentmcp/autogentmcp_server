"""
Session management for maintaining conversation context.
"""
from typing import Dict, List, Optional, Tuple
from app.ollama_client import ollama_client

class SessionManager:
    """Manage session contexts and conversation history."""
    
    def __init__(self, max_history_length: int = 10):
        self.max_history_length = max_history_length
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
    
    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get the conversation history for a session."""
        return self.sessions.get(session_id, [])
    
    def add_to_session(self, session_id: str, user_message: str, assistant_message: str):
        """Add a turn to the session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "user": user_message,
            "assistant": assistant_message
        })
        
        # Keep only recent history
        if len(self.sessions[session_id]) > self.max_history_length:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history_length:]
    
    def get_history_string(self, session_id: str) -> str:
        """Get formatted history string for prompts."""
        history = self.get_session_history(session_id)
        
        if not history:
            return ""
        
        # Check if we need to summarize
        if len(history) > self.max_history_length:
            summary, recent = self._summarize_history(history)
            history = recent
            history_str = f"Summary of previous conversation:\n{summary}\n\n"
        else:
            history_str = ""
        
        history_str += '\n'.join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history])
        return history_str
    
    def _summarize_history(self, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        """Summarize conversation history keeping only recent turns."""
        if len(history) <= self.max_history_length:
            return "", history
        
        # Keep last 3 turns, summarize the rest
        to_summarize = history[:-3]
        recent = history[-3:]
        
        summary_text = '\n'.join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in to_summarize])
        summary_prompt = f"/no_think\nSummarize the following conversation history for context, focusing on key facts, user goals, and important results.\n\n{summary_text}"
        
        print(f"[SessionManager] Summarizing history for session")
        summary = ollama_client.invoke_with_text_response(summary_prompt, allow_diagrams=False)
        
        return summary, recent
    
    def clear_session(self, session_id: str):
        """Clear the history for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"[SessionManager] Cleared session: {session_id}")
    
    def get_all_sessions(self) -> List[str]:
        """Get all session IDs."""
        return list(self.sessions.keys())
    
    def get_session_summary(self, session_id: str) -> Dict[str, any]:
        """Get a summary of a session."""
        history = self.get_session_history(session_id)
        return {
            "session_id": session_id,
            "turn_count": len(history),
            "last_user_message": history[-1]["user"] if history else None,
            "last_assistant_message": history[-1]["assistant"] if history else None
        }

# Global session manager instance
session_manager = SessionManager()
