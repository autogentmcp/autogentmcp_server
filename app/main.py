from fastapi import FastAPI

from app.langgraph_router import route_query

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query_endpoint(query: dict):
    """Route a user query to the best agent/tool using LangGraph."""
    return route_query(query)
