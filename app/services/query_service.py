"""
Query service - handles all query-related business logic.
"""
from typing import Dict, Any, Optional
from app.core.orchestrator import core_orchestrator
from app.models.requests import StreamingQueryRequest
from app.models.responses import QueryResponse


class QueryService:
    """Service class for handling query operations."""
    
    def __init__(self):
        self.orchestrator = core_orchestrator
    
    async def execute_query(
        self, 
        query: str, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a simple query.
        
        Args:
            query: The user query
            session_id: Optional session ID
            
        Returns:
            Query execution result
        """
        try:
            result = await self.orchestrator.execute_query(
                query=query,
                session_id=session_id
            )
            
            return {
                "status": "success",
                "session_id": result["session_id"],
                "result": result["result"],
                "endpoint_info": {
                    "type": "simple_orchestration",
                    "features": [
                        "intent_analysis",
                        "single_llm_call", 
                        "agent_execution",
                        "conversation_memory"
                    ]
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Query execution failed"
            }
    
    async def execute_streaming_query(
        self, 
        request: StreamingQueryRequest
    ) -> Dict[str, Any]:
        """
        Execute a query with streaming support.
        This delegates to the streaming controller for implementation.
        """
        # For now, fall back to regular query execution
        # Streaming logic will be in streaming_service.py
        return await self.execute_query(
            query=request.query,
            session_id=request.session_id
        )


# Global service instance
query_service = QueryService()
