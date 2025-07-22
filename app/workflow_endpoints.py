"""
Enhanced workflow endpoints for the new LangGraph-based dynamic orchestrator
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time

# Import the new LangGraph orchestrator
from app.langgraph_orchestrator import langgraph_orchestrator

router = APIRouter(prefix="/workflow", tags=["dynamic-workflow"])

class WorkflowStartRequest(BaseModel):
    query: str
    session_id: str
    max_steps: Optional[int] = 10

class WorkflowContinueRequest(BaseModel):
    workflow_id: str
    user_response: str

class WorkflowStopRequest(BaseModel):
    workflow_id: str
    session_id: str

@router.post("/start")
async def start_dynamic_workflow(request: WorkflowStartRequest):
    """Start a new dynamic workflow execution"""
    try:
        start_time = time.time()
        
        result = await langgraph_orchestrator.execute_workflow(
            user_query=request.query,
            session_id=request.session_id
        )
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "status": "success",
            "execution_time_ms": round(execution_time, 2),
            "workflow_result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow start failed: {str(e)}")

@router.post("/continue")
async def continue_workflow(request: WorkflowContinueRequest):
    """Continue workflow execution after user input"""
    try:
        result = await langgraph_orchestrator.continue_workflow(
            workflow_id=request.workflow_id,
            user_response=request.user_response
        )
        
        return {
            "status": "success",
            "workflow_result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow continuation failed: {str(e)}")

@router.post("/stop")
async def stop_workflow(request: WorkflowStopRequest):
    """Stop workflow execution"""
    try:
        result = await langgraph_orchestrator.stop_workflow(
            workflow_id=request.workflow_id,
            session_id=request.session_id
        )
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow stop failed: {str(e)}")

@router.get("/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get current workflow status"""
    # TODO: Implement workflow state persistence and retrieval
    return {
        "status": "not_implemented",
        "message": "Workflow status tracking not yet implemented",
        "workflow_id": workflow_id
    }

@router.get("/health")
async def workflow_health_check():
    """Health check for workflow system"""
    return {
        "status": "healthy",
        "orchestrator": "langgraph",
        "version": "2.0",
        "features": [
            "dynamic_planning",
            "multi_agent_execution", 
            "user_interaction",
            "streaming_updates",
            "error_recovery",
            "execution_control"
        ]
    }
