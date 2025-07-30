"""
Main agent executor that routes to specific agent types
"""

import asyncio
from typing import Dict, List, Any
from .data_agent import DataAgentExecutor
from .application_agent import ApplicationAgentExecutor
from ..models import AgentResult, ExecutionContext
from app.registry import fetch_agents_and_tools_from_registry

class AgentExecutor:
    """Main executor that routes to specific agent types"""
    
    def __init__(self):
        self.data_agent_executor = DataAgentExecutor()
        self.application_agent_executor = ApplicationAgentExecutor()
    
    async def execute_agent(self, agent_id: str, query: str, context: ExecutionContext = None) -> AgentResult:
        """Execute a single agent based on its type"""
        
        print("[AgentExecutor] Executing agent {}".format(agent_id))
        
        try:
            agents = fetch_agents_and_tools_from_registry()
            agent = agents.get(agent_id)
            
            if not agent:
                return AgentResult(
                    agent_id=agent_id,
                    agent_name="Unknown",
                    success=False,
                    error="Agent not found"
                )
            
            agent_type = agent.get("agent_type")
            agent_name = agent.get("name", agent_id)
            
            if agent_type == "data_agent":
                return await self.data_agent_executor.execute_data_agent(agent_id, query)
            elif agent_type == "application":
                return await self.application_agent_executor.execute_application_agent(agent_id, query)
            else:
                return AgentResult(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    success=False,
                    error="Unsupported agent type: {}".format(agent_type)
                )
        
        except Exception as e:
            return AgentResult(
                agent_id=agent_id,
                agent_name="Unknown",
                success=False,
                error="Agent execution error: {}".format(str(e))
            )
    
    async def execute_single(self, context: ExecutionContext, step: Dict[str, Any]) -> List[AgentResult]:
        """Execute single agent"""
        result = await self.execute_agent(step["agent_id"], step["query"], context)
        return [result]
    
    async def execute_sequential(self, context: ExecutionContext, steps: List[Dict[str, Any]]) -> List[AgentResult]:
        """Execute agents sequentially"""
        results = []
        agent_results = {}
        
        for step in steps:
            # Check if this step depends on another
            if "depends_on" in step:
                # Enhance query with previous result
                prev_result = agent_results.get(step["depends_on"])
                if prev_result and prev_result.success:
                    step["query"] = await self._enhance_query_with_context(
                        step["query"], prev_result
                    )
            
            result = await self.execute_agent(step["agent_id"], step["query"], context)
            results.append(result)
            agent_results[step["agent_id"]] = result
            
        return results
    
    async def execute_parallel(self, context: ExecutionContext, steps: List[Dict[str, Any]]) -> List[AgentResult]:
        """Execute agents in parallel"""
        tasks = []
        for step in steps:
            task = self.execute_agent(step["agent_id"], step["query"], context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(AgentResult(
                    agent_id=steps[i]["agent_id"],
                    agent_name="Unknown",
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)
                
        return final_results
    
    async def _enhance_query_with_context(self, query: str, prev_result: AgentResult) -> str:
        """Enhance query with context from previous result"""
        # This could be enhanced with LLM-based context integration
        # For now, just return the original query
        return query
