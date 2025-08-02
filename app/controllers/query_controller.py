"""
Query controller - handles all query-related endpoints.
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.query_service import query_service
from app.services.streaming_service import streaming_service
from app.models.requests import StreamingQueryRequest
from app.models.responses import QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def execute_query(request: StreamingQueryRequest):
    """Execute a simple query."""
    result = await query_service.execute_query(
        query=request.query,
        session_id=request.session_id
    )
    return result


@router.post("/enhanced/query", response_model=QueryResponse)
async def enhanced_query(request: StreamingQueryRequest):
    """Enhanced query endpoint (delegates to simple orchestrator)."""
    result = await query_service.execute_query(
        query=request.query,
        session_id=request.session_id
    )
    return result


@router.post("/orchestration/enhanced/stream")
async def orchestration_stream(request: StreamingQueryRequest):
    """Streaming orchestration endpoint."""
    return await streaming_service.create_streaming_response(request)
