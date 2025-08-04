"""
Shared models and data structures for the orchestrator
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import uuid
import time

@dataclass
class ExecutionContext:
    """Context for workflow execution with conversation history"""
    workflow_id: str
    session_id: str
    user_query: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    workflow_streamer: Optional[Any] = None  # Reference to workflow streamer for events

@dataclass
class AgentResult:
    """Standardized result from agent execution"""
    agent_id: str
    agent_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    row_count: int = 0
    query: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    visualization: Optional[Dict[str, Any]] = None

@dataclass
class IntentAnalysisResult:
    """Result from intent analysis"""
    conversation_state: str  # new_request|clarification_response|continuing_conversation
    action: str  # execute|ask_clarification|greeting|capabilities|general
    confidence: float
    message: str
    reasoning: str
    execution_plan: Optional[Dict[str, Any]] = None
    clarification_options: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ConversationTurn:
    """Single conversation turn with rich metadata"""
    timestamp: float
    query: str
    response: str
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    agents_used: List[str] = field(default_factory=list)
    clarification_options: List[Dict[str, Any]] = field(default_factory=list)

def create_workflow_context(user_query: str, session_id: str = None) -> ExecutionContext:
    """Create a new workflow execution context"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    workflow_id = str(uuid.uuid4())
    return ExecutionContext(workflow_id, session_id, user_query)
