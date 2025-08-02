"""
Core orchestrator - simplified single orchestrator for all queries.
"""
from typing import Dict, Any, Optional
from app.orchestrator import simple_orchestrator
from app.core.config import app_config
import uuid


class CoreOrchestrator:
    """
    Simplified core orchestrator that delegates to simple_orchestrator.
    This acts as a clean interface to the orchestration logic.
    """
    
    def __init__(self):
        self.orchestrator = simple_orchestrator
        
    async def execute_query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a query using the simple orchestrator.
        
        Args:
            query: The user query
            session_id: Optional session ID
            **kwargs: Additional parameters
            
        Returns:
            Dict containing the orchestration result
        """
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Execute using simple orchestrator
            result = await self.orchestrator.execute_workflow(
                user_query=query,
                session_id=session_id
            )
            
            return {
                "status": "success",
                "session_id": session_id,
                "result": result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
                "message": "Query execution failed"
            }


# Global orchestrator instance
core_orchestrator = CoreOrchestrator()
