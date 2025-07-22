"""
Simple Context Manager for maintaining conversation history and summarization.
Helps manage long-running conversations and context overflow.
"""

import json
import time
from typing import Dict, List, Any, Optional
from app.llm_client import LLMClient


class ContextManager:
    """
    Manages conversation context and summarization to prevent token overflow.
    Maintains recent context while summarizing older parts.
    """
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.session_contexts: Dict[str, Dict[str, Any]] = {}
        self.max_context_length = 4000  # Characters
        self.preserve_recent_messages = 3  # Always keep last N messages
        
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for a session, creating if needed."""
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = {
                "messages": [],
                "summary": "",
                "last_updated": time.time(),
                "total_messages": 0
            }
        return self.session_contexts[session_id]
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the session context."""
        context = self.get_session_context(session_id)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        context["messages"].append(message)
        context["last_updated"] = time.time()
        context["total_messages"] += 1
        
        # Check if summarization is needed
        if self._should_summarize(context):
            context = self._summarize_context(session_id, context)
            
        self.session_contexts[session_id] = context
    
    def get_context_for_llm(self, session_id: str, max_length: int = 3000) -> str:
        """Get formatted context for LLM prompt."""
        context = self.get_session_context(session_id)
        
        result_parts = []
        
        # Add summary if exists
        if context["summary"]:
            result_parts.append(f"Previous Context Summary: {context['summary']}")
        
        # Add recent messages
        recent_messages = context["messages"][-self.preserve_recent_messages:]
        for msg in recent_messages:
            timestamp = time.strftime("%H:%M:%S", time.localtime(msg["timestamp"]))
            result_parts.append(f"[{timestamp}] {msg['role']}: {msg['content'][:500]}...")
        
        full_context = "\n".join(result_parts)
        
        # Truncate if too long
        if len(full_context) > max_length:
            full_context = full_context[:max_length] + "... [truncated]"
            
        return full_context
    
    async def summarize_context(self, session_id: str, additional_context: str = "") -> str:
        """Manually trigger context summarization and return summary."""
        context = self.get_session_context(session_id)
        context = self._summarize_context(session_id, context, additional_context)
        self.session_contexts[session_id] = context
        return context["summary"]
    
    def _should_summarize(self, context: Dict[str, Any]) -> bool:
        """Determine if context needs summarization."""
        # Calculate total context length
        total_length = len(context.get("summary", ""))
        for msg in context["messages"]:
            total_length += len(msg["content"])
            
        return total_length > self.max_context_length and len(context["messages"]) > self.preserve_recent_messages
    
    def _summarize_context(self, session_id: str, context: Dict[str, Any], additional_context: str = "") -> Dict[str, Any]:
        """Summarize older context while preserving recent messages."""
        try:
            # Messages to summarize (all except recent ones)
            messages_to_summarize = context["messages"][:-self.preserve_recent_messages]
            recent_messages = context["messages"][-self.preserve_recent_messages:]
            
            if not messages_to_summarize:
                return context  # Nothing to summarize
            
            # Prepare content for summarization
            content_to_summarize = []
            for msg in messages_to_summarize:
                content_to_summarize.append(f"{msg['role']}: {msg['content']}")
            
            if additional_context:
                content_to_summarize.append(f"Additional Context: {additional_context}")
            
            summarization_content = "\n".join(content_to_summarize)
            
            # Create summarization prompt
            prompt = f"""
            Summarize the following conversation/context, preserving key information:
            
            Existing Summary: {context.get('summary', 'None')}
            
            Content to Summarize:
            {summarization_content[:2000]}  # Limit to prevent token overflow
            
            Instructions:
            1. Merge with existing summary if present
            2. Preserve important facts, decisions, and outcomes
            3. Keep key data points and agent results
            4. Be concise but comprehensive
            5. Focus on information relevant to ongoing analysis
            
            Return a clear, structured summary.
            """
            
            # Use sync method for simplicity (can be made async if needed)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            summary = loop.run_until_complete(self.llm_client.query_llm_text(prompt, max_tokens=400))
            loop.close()
            
            # Update context
            context["summary"] = summary
            context["messages"] = recent_messages  # Keep only recent messages
            
            print(f"[ContextManager] Summarized context for session {session_id}: {len(messages_to_summarize)} messages -> summary")
            
        except Exception as e:
            print(f"[ContextManager] Failed to summarize context for session {session_id}: {e}")
            # On error, just keep existing context but trim old messages
            context["messages"] = context["messages"][-self.preserve_recent_messages * 2:]  # Keep more on error
        
        return context
    
    def clear_session_context(self, session_id: str):
        """Clear all context for a session."""
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session context."""
        context = self.get_session_context(session_id)
        return {
            "total_messages": context["total_messages"],
            "current_messages": len(context["messages"]),
            "has_summary": bool(context["summary"]),
            "summary_length": len(context["summary"]),
            "last_updated": context["last_updated"],
            "context_age_minutes": (time.time() - context["last_updated"]) / 60
        }


# Global instance
context_manager = ContextManager()
