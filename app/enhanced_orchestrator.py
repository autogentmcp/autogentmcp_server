"""
Enhanced Multi-Step Workflow Orchestrator
Implements dynamic agent selection and iterative execution following the workflow:
A[User Input] -> B[LLM: Recommend Agent] -> C[Select Agent] -> D[LLM: Prepare Request] -> 
E[Execute Agent] -> F[LLM: Next Step?] -> [Yes: back to D] -> [No: Return Final Result]
"""

import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
from dataclasses import dataclass

from app.workflow_streamer import workflow_streamer, EventType, StreamEvent
from app.ollama_client import OllamaClient
from app.registry import fetch_agents_and_tools_from_registry
from app.database_query_executor import DatabaseQueryExecutor
from app.context_manager import ContextManager


@dataclass
class AgentExecutionResult:
    """Result from executing an agent."""
    success: bool
    data: Any
    agent_id: str
    agent_name: str
    execution_time: float
    query_executed: Optional[str] = None
    error: Optional[str] = None


@dataclass 
class WorkflowContext:
    """Context maintained throughout the workflow."""
    session_id: str
    workflow_id: str
    original_query: str
    conversation_history: List[Dict[str, Any]]
    agent_results: List[AgentExecutionResult]
    current_step: int
    max_steps: int = 10
    start_time: float = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
    
    def add_result(self, result: AgentExecutionResult):
        """Add an agent execution result to the context."""
        self.agent_results.append(result)
        self.current_step += 1
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of execution."""
        return {
            "total_steps": len(self.agent_results),
            "total_time": time.time() - self.start_time,
            "agents_used": [r.agent_name for r in self.agent_results],
            "successful_executions": sum(1 for r in self.agent_results if r.success),
            "failed_executions": sum(1 for r in self.agent_results if not r.success)
        }


class RegistryManager:
    """Simple wrapper for registry functions."""
    
    def list_available_agents(self):
        """Get list of available agents from registry."""
        agents_dict = fetch_agents_and_tools_from_registry()
        return list(agents_dict.values())
    
    def get_agent_enhanced_details(self, agent_id: str):
        """Get enhanced details for a specific agent."""
        agents_dict = fetch_agents_and_tools_from_registry() 
        agent = agents_dict.get(agent_id)
        if agent:
            return agent
        return {}


class EnhancedOrchestrator:
    """
    Enhanced orchestrator that supports multi-step iterative workflows.
    Maintains context and allows LLM to decide on next steps.
    """
    
    def __init__(self):
        self.llm_client = OllamaClient()
        self.registry_manager = RegistryManager()
        self.db_executor = DatabaseQueryExecutor()
        self.context_manager = ContextManager()
        
    async def execute_workflow(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """
        Execute the enhanced multi-step workflow.
        
        Args:
            user_query: The user's query
            session_id: Session identifier
            
        Returns:
            Final workflow result
        """
        print(f"[EnhancedOrchestrator] Starting workflow for query: '{user_query}' session: {session_id}")
        workflow_id = str(uuid.uuid4())
        
        # Initialize workflow context
        context = WorkflowContext(
            session_id=session_id,
            workflow_id=workflow_id,
            original_query=user_query,
            conversation_history=[],
            agent_results=[],
            current_step=0
        )
        
        # Emit workflow started
        workflow_streamer.emit_workflow_started(
            workflow_id=workflow_id,
            session_id=session_id,
            title="Enhanced Multi-Step Analysis",
            description="Dynamic agent orchestration with iterative execution",
            steps=context.max_steps
        )

        # Allow event to be processed
        await asyncio.sleep(0.1)
        
        try:
            # Main workflow loop
            print(f"[EnhancedOrchestrator] Starting main workflow loop, max_steps: {context.max_steps}")
            while context.current_step < context.max_steps:
                print(f"[EnhancedOrchestrator] Step {context.current_step + 1}/{context.max_steps}")
                
                # Step 1: LLM recommends agent based on current context
                agent_recommendation = await self._get_agent_recommendation(context)
                print(f"[EnhancedOrchestrator] Agent recommendation: {agent_recommendation}")
                
                # Allow streaming events to be processed
                await asyncio.sleep(0.1)
                
                if agent_recommendation.get("action") == "complete":
                    # LLM decided workflow is complete
                    print(f"[EnhancedOrchestrator] LLM decided workflow is complete")
                    break
                elif agent_recommendation.get("action") == "require_user_choice":
                    # LLM wants user to choose between multiple agents
                    await self._request_user_agent_choice(context, agent_recommendation)
                    break  # Pause workflow for user input
                elif agent_recommendation.get("action") == "require_user_input":
                    # LLM needs additional information from user
                    await self._request_user_input(context, agent_recommendation)
                    break  # Pause workflow for user input
                
                # Step 2: Select and validate the recommended agent
                selected_agent = await self._select_agent(context, agent_recommendation)
                
                # Allow streaming events to be processed
                await asyncio.sleep(0.1)
                
                if not selected_agent:
                    # No suitable agent found
                    workflow_streamer.emit_error(
                        workflow_id, session_id, f"step_{context.current_step}",
                        "No suitable agent found for the current task"
                    )
                    break
                
                # Step 3: LLM prepares the specific request for the agent
                agent_request = await self._prepare_agent_request(context, selected_agent)
                
                # Handle complexity analysis results
                if agent_request.get("action") == "suggest_narrowing":
                    # Query is too complex, suggest user to narrow scope
                    await self._request_query_narrowing(context, agent_request)
                    break  # Pause workflow for user refinement
                
                if not agent_request or not agent_request.get("sql_query"):
                    # No valid request could be prepared
                    workflow_streamer.emit_error(
                        workflow_id, session_id, f"step_{context.current_step}",
                        "Could not prepare a valid request for the selected agent"
                    )
                    break
                
                # Step 4: Execute the agent (may be chunked execution)
                if agent_request.get("is_chunked"):
                    execution_result = await self._execute_chunked_agent(context, selected_agent, agent_request)
                else:
                    execution_result = await self._execute_agent(context, selected_agent, agent_request)
                
                # Allow streaming events to be processed
                await asyncio.sleep(0.1)
                
                # Add result to context
                context.add_result(execution_result)
                
                # Allow streaming events for step completion to be processed
                await asyncio.sleep(0.1)
                
                # Step 5: LLM evaluates results and decides next step
                next_step_decision = await self._evaluate_and_decide_next_step(context)
                
                # Allow streaming events to be processed
                await asyncio.sleep(0.1)
                
                # Hard loop detection - force completion if too many similar queries
                recent_queries = [r.query_executed for r in context.agent_results[-5:] if r.query_executed]
                query_counts = {}
                for query in recent_queries:
                    normalized_query = ' '.join(query.lower().split())
                    query_counts[normalized_query] = query_counts.get(normalized_query, 0) + 1
                
                max_repeated_count = max(query_counts.values()) if query_counts else 0
                if max_repeated_count >= 3:  # Hard stop after 3 identical queries
                    workflow_streamer.emit_debug_info(
                        context.workflow_id, context.session_id, f"step_{context.current_step}",
                        f"HARD LOOP DETECTED: Query repeated {max_repeated_count} times. Forcing completion.",
                        {"repeated_queries": query_counts}
                    )
                    next_step_decision = {
                        "action": "complete",
                        "reasoning": f"Workflow forced to complete due to loop detection - query repeated {max_repeated_count} times",
                        "completeness_score": 0.7,
                        "confidence": 0.9,
                        "loop_detected": True
                    }
                
                # Update conversation history with safe serialization
                try:
                    # Create a safe copy of execution result without datetime objects
                    safe_execution_result = {
                        "success": execution_result.success,
                        "agent_id": execution_result.agent_id,
                        "agent_name": execution_result.agent_name,
                        "execution_time": execution_result.execution_time,
                        "query_executed": execution_result.query_executed,
                        "error": execution_result.error,
                        "row_count": execution_result.data.get("row_count", 0) if execution_result.success else 0
                    }
                    
                    context.conversation_history.append({
                        "step": context.current_step,
                        "agent": selected_agent["name"],
                        "result": safe_execution_result,
                        "decision": next_step_decision
                    })
                except Exception as e:
                    print(f"[EnhancedOrchestrator] Warning: Could not add to conversation history: {str(e)}")
                    # Add minimal history entry
                    context.conversation_history.append({
                        "step": context.current_step,
                        "agent": selected_agent["name"],
                        "success": execution_result.success,
                        "decision_action": next_step_decision.get("action", "unknown")
                    })
                
                # Check if LLM wants to continue or complete
                if next_step_decision.get("action") == "complete":
                    break
                elif next_step_decision.get("action") == "require_user_input":
                    # Request user input and wait
                    await self._request_user_input(context, next_step_decision)
                    break  # For now, break on user input (can be extended)
            
            # Generate final result
            final_result = await self._generate_final_result(context)
            
            # Emit workflow completed
            workflow_streamer.emit_workflow_completed(
                workflow_id=workflow_id,
                session_id=session_id,
                final_answer=final_result["final_answer"],
                execution_time=time.time() - context.start_time
            )
            
            return final_result
            
        except Exception as e:
            print(f"[EnhancedOrchestrator] Workflow error: {str(e)}")
            workflow_streamer.emit_error(
                workflow_id, session_id, f"step_{context.current_step}",
                f"Workflow error: {str(e)}"
            )
            raise
    
    async def _get_agent_recommendation(self, context: WorkflowContext) -> Dict[str, Any]:
        """Step 1: LLM recommends which agent to use next."""
        print(f"[EnhancedOrchestrator] Getting agent recommendation for step {context.current_step}")
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, f"step_{context.current_step}_recommend",
            "agent_recommendation", "🤖 LLM analyzing requirements and recommending agent..."
        )
        
        # Allow event to be processed
        await asyncio.sleep(0.1)
        
        # Get available agents
        available_agents = self.registry_manager.list_available_agents()
        print(f"[EnhancedOrchestrator] Found {len(available_agents)} available agents")
        
        # Prepare context for LLM
        context_summary = await self._prepare_context_summary(context)
        print(f"[EnhancedOrchestrator] Context summary: {context_summary}")
        
        # Build agent information for LLM with proper capabilities extraction
        available_agents_for_llm = []
        for a in available_agents:
            agent_info = {
                "id": a.get("appKey") or a["id"],
                "database_id": a["id"],
                "name": a["name"], 
                "description": a.get("description", ""),
                "agent_type": a.get("agent_type", "data_agent")
            }
            
            # Extract capabilities properly based on agent type
            if a.get("agent_type") == "application":
                # For applications, extract endpoint details as capabilities
                endpoints = a.get("endpoints", [])
                capabilities = []
                for endpoint in endpoints:
                    cap = {
                        "name": endpoint.get("name", ""),
                        "method": endpoint.get("method", "GET"),
                        "path": endpoint.get("path", ""),
                        "description": endpoint.get("description", "")
                    }
                    capabilities.append(cap)
                agent_info["capabilities"] = capabilities
            else:
                # For data agents, use database type and description
                agent_info["capabilities"] = {
                    "database_type": a.get("database_type", ""),
                    "description": a.get("description", "")
                }
            
            available_agents_for_llm.append(agent_info)

        prompt = f"""
        You are an expert AI orchestrator. Analyze the current situation and recommend the next agent to execute.
        
        Original User Query: {context.original_query}
        Current Step: {context.current_step + 1}/{context.max_steps}
        
        Context Summary: {context_summary}
        
        Available Agents: {json.dumps(available_agents_for_llm, indent=2)}
        
        Previous Results: {json.dumps([{"agent": r.agent_name, "success": r.success, "data_summary": str(r.data)[:200]} for r in context.agent_results], indent=2)}
        
        Instructions:
        1. If the user query is fully answered with existing results, return {{"action": "complete"}}
        2. If more data/analysis is needed, recommend the best agent for the next step
        3. For APPLICATIONS (agent_type: "application"), use their primary identifier (id field, which is the appKey)
        4. For DATA AGENTS (agent_type: "data_agent"), use their database_id field
        5. Consider what information is still missing or needs clarification
        6. Choose agents that complement previous results, not duplicate them
        7. For cross-database analysis or multi-agent queries, prefer the enhanced LLM orchestrator with high confidence
        8. Only request user choice if the query is ambiguous about the specific data source or business context
        9. If you need additional context or information from the user, return {{"action": "require_user_input"}}
        10. If the same agent has been used repeatedly without clear progress, try a different approach
        
        IMPORTANT: When recommending an agent, use the "id" field from the Available Agents list above as the recommended_agent_id.
        
        For inventory analysis, cross-database comparisons, or multi-agent scenarios, prefer using the enhanced LLM orchestrator automatically.
        
        IMPORTANT: When recommending an agent, use the "id" field from the Available Agents list above as the recommended_agent_id.
        
        Respond with JSON:
        {{
            "action": "recommend_agent" | "complete" | "require_user_choice" | "require_user_input",
            "recommended_agent_id": "agent_id_if_recommending",
            "reasoning": "why this agent is needed or why user choice is required",
            "expected_outcome": "what this agent should accomplish",
            "confidence": 0.95,
            "user_choice_options": [
                {{"id": "agent1", "name": "Agent Name", "reason": "why this agent", "expected_outcome": "what it would do"}},
                {{"id": "agent2", "name": "Agent Name", "reason": "why this agent", "expected_outcome": "what it would do"}}
            ],
            "user_input_request": "what specific information to ask the user"
        }}
        """
        
        try:
            print(f"[EnhancedOrchestrator] Calling LLM for agent recommendation...")
            recommendation = self.ollama_client.invoke_with_json_response(prompt, timeout=600)
            print(f"[EnhancedOrchestrator] LLM recommendation result: {recommendation}")
            if not recommendation:
                raise Exception("No valid JSON response received from LLM")
            
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"step_{context.current_step}_recommend",
                "agent_recommendation", 2.0
            )
            
            return recommendation
            
        except Exception as e:
            print(f"[EnhancedOrchestrator] Error getting agent recommendation: {str(e)}")
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_recommend",
                f"Failed to get agent recommendation: {str(e)}"
            )
            return {"action": "complete"}  # Fail gracefully
    
    async def _select_agent(self, context: WorkflowContext, recommendation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Step 2: Select and validate the recommended agent."""
        if recommendation.get("action") != "recommend_agent":
            return None
            
        agent_id = recommendation.get("recommended_agent_id")
        if not agent_id:
            return None
        
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, f"step_{context.current_step}_select",
            "agent_selection", f"🎯 Selecting agent: {agent_id}..."
        )
        
        # Get agent details from registry
        agents = self.registry_manager.list_available_agents()
        
        # For applications, prioritize appKey as the unique identifier
        # For data agents, use id as primary identifier
        selected_agent = None
        
        # First try to find by appKey (primary identifier for applications)
        selected_agent = next((a for a in agents if a.get("appKey") == agent_id), None)
        
        # If not found, try by name (secondary identifier for applications) 
        if not selected_agent:
            selected_agent = next((a for a in agents if a.get("name") == agent_id), None)
        
        # If still not found, try by database ID (primary identifier for data agents)
        if not selected_agent:
            selected_agent = next((a for a in agents if a["id"] == agent_id), None)
        
        if not selected_agent:
            print(f"[EnhancedOrchestrator] Warning: Agent '{agent_id}' not found. Available agents:")
            for agent in agents[:5]:  # Log first 5 for debugging
                print(f"  - ID: {agent['id']}, Name: {agent.get('name', 'N/A')}, AppKey: {agent.get('appKey', 'N/A')}, Type: {agent.get('agent_type', 'N/A')}")
        else:
            print(f"[EnhancedOrchestrator] Selected agent: {selected_agent.get('name')} (Type: {selected_agent.get('agent_type', 'N/A')}, AppKey: {selected_agent.get('appKey', 'N/A')})")
        
        if selected_agent:
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"step_{context.current_step}_select",
                "agent_selection", 1.0
            )
            
            # Emit agent selection event
            workflow_streamer.emit_routing_decision(
                context.workflow_id, context.session_id, "agent_selection",
                selected_agent["name"], recommendation.get("confidence", 0.8) * 100,
                recommendation.get("reasoning", "LLM recommendation")
            )
        
        return selected_agent
    
    async def _prepare_agent_request(self, context: WorkflowContext, agent: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: LLM prepares specific request for the agent."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, f"step_{context.current_step}_prepare",
            "request_preparation", f"📝 Preparing request for {agent['name']}..."
        )
        
        # Check agent type to determine preparation method
        agent_type = agent.get("agent_type", "data_agent")
        
        if agent_type == "application":
            # Handle application API request preparation
            return await self._prepare_application_request(context, agent)
        else:
            # Handle data agent SQL request preparation (existing logic)
            return await self._prepare_data_agent_request(context, agent)
    
    async def _prepare_application_request(self, context: WorkflowContext, agent: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare API request for application agents."""
        # Get application details from registry
        from app.registry import get_enhanced_agent_details_for_llm
        app_details = get_enhanced_agent_details_for_llm(agent["id"])
        
        if not app_details:
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_prepare",
                f"Application details not found for {agent['id']}"
            )
            return {}
        
        context_summary = await self._prepare_context_summary(context)
        
        # Build endpoints information for LLM
        endpoints_info = []
        for endpoint in app_details.get('endpoints', []):
            endpoints_info.append({
                "name": endpoint.get('name'),
                "method": endpoint.get('method'),
                "path": endpoint.get('path'),
                "description": endpoint.get('description'),
                "path_params": endpoint.get('pathParams', {}),
                "query_params": endpoint.get('queryParams', {}),
                "request_body": endpoint.get('requestBody', {})
            })
        
        prompt = f"""
        You are an API integration expert. Generate a structured API request for the user's query.

        USER QUERY: {context.original_query}

        APPLICATION DETAILS:
        - Name: {app_details.get('name')}
        - Description: {app_details.get('description')}
        - Base Domain: {app_details.get('environment', {}).get('baseDomain', '')}

        AVAILABLE ENDPOINTS:
        {json.dumps(endpoints_info, indent=2)}

        CONTEXT FROM PREVIOUS STEPS:
        {context_summary}

        Previous Agent Results:
        {json.dumps([{"agent": r.agent_name, "query": r.query_executed, "success": r.success, "error": r.error} for r in context.agent_results], indent=2)}

        Instructions:
        1. Select the most appropriate endpoint for the user's request
        2. Extract required path parameters, query parameters, and request body from the user query
        3. Build the complete API URL by combining base domain and endpoint path
        4. Ensure all required parameters are provided

        Respond with JSON:
        {{
            "selected_endpoint": "endpoint_name",
            "http_method": "GET|POST|PUT|DELETE",
            "api_url": "complete_url_with_path_params_filled",
            "path_params": {{"param_name": "extracted_value"}},
            "query_params": {{"param_name": "extracted_value"}},
            "request_body": {{"key": "value"}} or null,
            "headers": {{"Content-Type": "application/json"}},
            "reasoning": "why this endpoint and parameters were selected",
            "confidence": 0.95
        }}
        """
        
        try:
            request_details = self.ollama_client.invoke_with_json_response(prompt, timeout=600)
            if not request_details:
                raise Exception("No valid JSON response received from LLM")
            
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"step_{context.current_step}_prepare",
                "request_preparation", 2.0
            )
            
            return request_details
            
        except Exception as e:
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_prepare",
                f"Failed to prepare application request: {str(e)}"
            )
            return {}

    async def _prepare_data_agent_request(self, context: WorkflowContext, agent: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare SQL request for data agents."""
        # Get enhanced agent details for context - use the same function as langgraph orchestrator
        from app.registry import get_enhanced_agent_details_for_llm
        agent_details = get_enhanced_agent_details_for_llm(agent["id"])
        
        if not agent_details:
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_prepare",
                f"Enhanced agent details not found for {agent['id']}"
            )
            return {}
        
        # First, check if we need to break down the request
        request_analysis = await self._analyze_request_complexity(context, agent_details)
        
        if request_analysis.get("action") == "suggest_narrowing":
            # Request is too complex, suggest user to narrow scope
            return {
                "action": "suggest_narrowing",
                "reasoning": request_analysis.get("reasoning"),
                "suggestions": request_analysis.get("suggestions"),
                "complexity_score": request_analysis.get("complexity_score")
            }
        elif request_analysis.get("action") == "break_down":
            # Request can be broken down into smaller queries
            return await self._prepare_chunked_request(context, agent_details, request_analysis)
        
        # Normal single query processing
        
        context_summary = await self._prepare_context_summary(context)
        
        # Extract schema information for the prompt
        database_type = agent_details.get('database_type', 'unknown')
        agent_name = agent_details.get('name', 'Unknown')
        description = agent_details.get('description', '')
        schema_info = agent_details.get('schema', '')
        sample_queries = agent_details.get('sample_queries', [])
        
        # Generate column reference from LLM client like in langgraph
        from app.ollama_client import ollama_client
        column_reference = ""
        tables_data = agent_details.get('tables', [])
        if tables_data:
            column_reference = ollama_client._extract_column_reference_from_structured_data(tables_data)
        else:
            column_reference = "⚠️  Use ONLY exact column names from the schema below"
        
        prompt = f"""
        You are an expert SQL developer. Generate a structured response for the user's SQL query request.

        ⚠️ CRITICAL: You MUST use ONLY the exact column names from the schema below. DO NOT assume or invent any column names.

        USER QUERY: {context.original_query}

        DATABASE DETAILS:
        - Database Type: {database_type}
        - Agent Name: {agent_name}
        - Description: {description}
        - Total Tables: {len(agent_details.get('tables', []))}
        - Total Relations: {len(agent_details.get('table_relations', []))}

        EXACT COLUMN VALIDATION:
        {column_reference}

        DATABASE SCHEMA:
        {schema_info}

        SAMPLE QUERIES FOR REFERENCE:
        {chr(10).join(sample_queries[:3])}

        CONTEXT FROM PREVIOUS STEPS:
        {context_summary}

        Previous Agent Results:
        {json.dumps([{"agent": r.agent_name, "query": r.query_executed, "success": r.success, "error": r.error} for r in context.agent_results], indent=2)}

        Instructions:
        1. Generate a specific SQL query that uses ONLY the exact table and column names shown above
        2. Focus on data that hasn't been retrieved yet or needs different analysis  
        3. Consider the agent's available tables and relationships
        4. Make the query targeted and efficient
        5. Validate all column names against the schema before finalizing
        
        Respond with JSON:
        {{
            "column_validation": "Step 1: Tables needed: [...]. Step 2: Exact columns from schema: [...]. Step 3: Verified all columns exist in schema.",
            "sql_query": "SELECT ... your query here using EXACT column names",
            "query_purpose": "what this query will accomplish", 
            "expected_data": "description of expected results",
            "reasoning": "why this specific query addresses the user's needs",
            "confidence": 0.95
        }}
        """
        
        try:
            request_details = self.ollama_client.invoke_with_json_response(prompt, timeout=600)
            if not request_details:
                raise Exception("No valid JSON response received from LLM")
            
            workflow_streamer.emit_sql_generated(
                context.workflow_id, context.session_id,
                request_details["sql_query"], agent_details.get("database_type", "unknown"),
                request_details.get("query_purpose")
            )
            
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"step_{context.current_step}_prepare",
                "request_preparation", 2.0
            )
            
            return request_details
            
        except Exception as e:
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_prepare",
                f"Failed to prepare agent request: {str(e)}"
            )
            return {}

    async def _request_query_narrowing(self, context: WorkflowContext, request: Dict[str, Any]):
        """Request user to narrow the query scope due to complexity."""
        narrowing_request = {
            "type": "query_narrowing",
            "complexity_score": request.get("complexity_score", 0.0),
            "reasoning": request.get("reasoning", "Query is too complex for efficient execution"),
            "suggestions": request.get("suggestions", []),
            "performance_concerns": request.get("performance_concerns", []),
            "workflow_id": context.workflow_id,
            "step": context.current_step
        }
        
        workflow_streamer.emit_user_input_required(
            context.workflow_id, context.session_id, f"step_{context.current_step}_narrowing",
            narrowing_request
        )

    async def _execute_chunked_agent(self, context: WorkflowContext, agent: Dict[str, Any], request: Dict[str, Any]) -> AgentExecutionResult:
        """Execute an agent with chunked queries for complex requests."""
        start_time = time.time()
        
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, f"step_{context.current_step}_execute_chunked",
            "chunked_execution", f"⚡ Executing {agent['name']} in chunks..."
        )
        
        chunk_info = request.get("chunk_info", {})
        total_chunks = chunk_info.get("total_chunks", 1)
        
        try:
            # Execute the first chunk
            sql_query = request.get("sql_query")
            if not sql_query:
                raise ValueError("No SQL query provided in chunked request")
            
            # Get agent vault key and connection type
            vault_key = agent.get("vault_key")
            connection_type = agent.get("database_type") or agent.get("connectionType", "postgresql")
            
            if not vault_key:
                raise ValueError(f"No vault key found for agent {agent['id']}")
            
            # Execute the first chunk query
            chunk_result = self.db_executor.execute_query(
                vault_key=vault_key,
                connection_type=connection_type,
                sql_query=sql_query,
                limit=100
            )
            
            # Combine results from all chunks (for now, just return the first chunk)
            # In a full implementation, you would iterate through all chunks
            combined_data = chunk_result.get("data", [])
            combined_row_count = chunk_result.get("row_count", 0)
            
            execution_time = time.time() - start_time
            
            # Create result object
            execution_result = AgentExecutionResult(
                success=chunk_result.get("status") == "success",
                data={
                    "data": combined_data,
                    "row_count": combined_row_count,
                    "chunked": True,
                    "chunks_executed": 1,
                    "total_chunks": total_chunks,
                    "chunk_info": chunk_info
                },
                agent_id=agent["id"],
                agent_name=agent["name"],
                execution_time=execution_time,
                query_executed=f"Chunked Query 1/{total_chunks}: {sql_query}",
                error=chunk_result.get("message") if chunk_result.get("status") != "success" else None
            )
            
            if execution_result.success:
                workflow_streamer.emit_step_completed(
                    context.workflow_id, context.session_id, f"step_{context.current_step}_execute_chunked",
                    "chunked_execution", execution_time
                )
                
                # Emit chunked data received event
                workflow_streamer.emit_event(StreamEvent(
                    event_type=EventType.DATA_QUERY,
                    workflow_id=context.workflow_id,
                    timestamp=datetime.utcnow(),
                    session_id=context.session_id,
                    step_id=f"step_{context.current_step}_execute_chunked",
                    data={
                        "agent_name": agent["name"],
                        "query": sql_query,
                        "row_count": combined_row_count,
                        "execution_time": execution_time,
                        "chunks_executed": 1,
                        "total_chunks": total_chunks,
                        "message": f"📊 Retrieved {combined_row_count} rows from {agent['name']} (Chunk 1/{total_chunks})"
                    }
                ))
            else:
                workflow_streamer.emit_error(
                    context.workflow_id, context.session_id, f"step_{context.current_step}_execute_chunked",
                    f"Chunked agent execution failed: {execution_result.error}"
                )
            
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_execute_chunked",
                f"Chunked agent execution error: {str(e)}"
            )
            
            return AgentExecutionResult(
                success=False,
                data={},
                agent_id=agent["id"],
                agent_name=agent["name"],
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _execute_agent(self, context: WorkflowContext, agent: Dict[str, Any], request: Dict[str, Any]) -> AgentExecutionResult:
        """Step 4: Execute the agent with the prepared request."""
        start_time = time.time()
        
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, f"step_{context.current_step}_execute",
            "agent_execution", f"⚡ Executing {agent['name']}..."
        )
        
        try:
            # Check agent type to determine execution method
            agent_type = agent.get("agent_type", "data_agent")  # Default to data_agent for backward compatibility
            
            if agent_type == "application":
                # Handle application API calls
                return await self._execute_application_api(context, agent, request, start_time)
            else:
                # Handle data agent SQL queries (existing logic)
                return await self._execute_data_agent_sql(context, agent, request, start_time)
                
        except Exception as e:
            execution_time = time.time() - start_time
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_execute",
                f"Agent execution error: {str(e)}"
            )
            
            return AgentExecutionResult(
                success=False,
                data={},
                agent_id=agent["id"],
                agent_name=agent["name"],
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _execute_data_agent_sql(self, context: WorkflowContext, agent: Dict[str, Any], request: Dict[str, Any], start_time: float) -> AgentExecutionResult:
        """Execute SQL query for data agents."""
        try:
            sql_query = request.get("sql_query")
            if not sql_query:
                raise ValueError("No SQL query provided in request")
            
            # Get agent vault key and connection type
            vault_key = agent.get("vault_key")
            connection_type = agent.get("database_type") or agent.get("connectionType", "postgresql")
            
            if not vault_key:
                raise ValueError(f"No vault key found for agent {agent['id']}")
            
            # Execute the query using the database executor with correct parameters
            result = self.db_executor.execute_query(
                vault_key=vault_key,
                connection_type=connection_type,
                sql_query=sql_query,
                limit=100
            )
            
            execution_time = time.time() - start_time
            
            # Create result object
            execution_result = AgentExecutionResult(
                success=result.get("status") == "success",
                data=result,
                agent_id=agent["id"],
                agent_name=agent["name"],
                execution_time=execution_time,
                query_executed=sql_query,
                error=result.get("message") if result.get("status") != "success" else None
            )
            
            if execution_result.success:
                workflow_streamer.emit_step_completed(
                    context.workflow_id, context.session_id, f"step_{context.current_step}_execute",
                    "agent_execution", execution_time
                )
                
                # Emit data received event
                row_count = result.get("row_count", 0)
                workflow_streamer.emit_event(StreamEvent(
                    event_type=EventType.DATA_QUERY,
                    workflow_id=context.workflow_id,
                    timestamp=datetime.utcnow(),
                    session_id=context.session_id,
                    step_id=f"step_{context.current_step}_execute",
                    data={
                        "agent_name": agent["name"],
                        "query": sql_query,
                        "row_count": row_count,
                        "execution_time": execution_time,
                        "message": f"📊 Retrieved {row_count} rows from {agent['name']}"
                    }
                ))
            else:
                workflow_streamer.emit_error(
                    context.workflow_id, context.session_id, f"step_{context.current_step}_execute",
                    f"Agent execution failed: {execution_result.error}"
                )
            
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_execute",
                f"Data agent execution error: {str(e)}"
            )
            
            return AgentExecutionResult(
                success=False,
                data={},
                agent_id=agent["id"],
                agent_name=agent["name"],
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _execute_application_api(self, context: WorkflowContext, agent: Dict[str, Any], request: Dict[str, Any], start_time: float) -> AgentExecutionResult:
        """Execute API call for applications."""
        import httpx
        
        try:
            # Get API request details from the prepared request
            api_url = request.get("api_url")
            http_method = request.get("http_method", "GET").upper()
            request_headers = request.get("headers", {})
            request_body = request.get("request_body")
            query_params = request.get("query_params", {})
            
            if not api_url:
                raise ValueError("No API URL provided in request")
            
            # Get vault key for authentication headers
            vault_key = agent.get("vault_key")
            if vault_key:
                # Get authentication headers from vault
                auth_headers = await self._get_auth_headers_from_vault(vault_key)
                if auth_headers:
                    request_headers.update(auth_headers)
            
            # Make the API call
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(
                    method=http_method,
                    url=api_url,
                    headers=request_headers,
                    params=query_params,
                    json=request_body if request_body and http_method in ["POST", "PUT", "PATCH"] else None
                )
            
            execution_time = time.time() - start_time
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = {"response": response.text}
            
            # Create result object
            execution_result = AgentExecutionResult(
                success=response.is_success,
                data={
                    "status_code": response.status_code,
                    "response_data": response_data,
                    "request_details": {
                        "url": api_url,
                        "method": http_method,
                        "headers": {k: "***" for k in request_headers.keys()},  # Hide header values
                        "query_params": query_params
                    }
                },
                agent_id=agent["id"],
                agent_name=agent["name"],
                execution_time=execution_time,
                query_executed=f"{http_method} {api_url}",
                error=f"HTTP {response.status_code}: {response.text}" if not response.is_success else None
            )
            
            if execution_result.success:
                workflow_streamer.emit_step_completed(
                    context.workflow_id, context.session_id, f"step_{context.current_step}_execute",
                    "agent_execution", execution_time
                )
                
                # Emit API call event
                workflow_streamer.emit_event(StreamEvent(
                    event_type=EventType.DATA_QUERY,  # Using same event type for consistency
                    workflow_id=context.workflow_id,
                    timestamp=datetime.utcnow(),
                    session_id=context.session_id,
                    step_id=f"step_{context.current_step}_execute",
                    data={
                        "agent_name": agent["name"],
                        "query": f"{http_method} {api_url}",
                        "status_code": response.status_code,
                        "execution_time": execution_time,
                        "message": f"🌐 API call to {agent['name']} completed with status {response.status_code}"
                    }
                ))
            else:
                workflow_streamer.emit_error(
                    context.workflow_id, context.session_id, f"step_{context.current_step}_execute",
                    f"API call failed: {execution_result.error}"
                )
            
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_execute",
                f"Application API execution error: {str(e)}"
            )
            
            return AgentExecutionResult(
                success=False,
                data={},
                agent_id=agent["id"],
                agent_name=agent["name"],
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _get_auth_headers_from_vault(self, vault_key: str) -> Dict[str, str]:
        """Get authentication headers from vault using the vault key."""
        try:
            from app.vault_manager import VaultManager
            vault_manager = VaultManager()
            
            # Get credentials from vault
            credentials = vault_manager.get_secret(vault_key)
            if not credentials:
                return {}
            
            # Build authentication headers based on the credentials
            headers = {}
            
            # Common header mappings
            if "api_key" in credentials:
                headers["Authorization"] = f"Bearer {credentials['api_key']}"
            elif "authorization" in credentials:
                headers["Authorization"] = credentials["authorization"]
            elif "bearer_token" in credentials:
                headers["Authorization"] = f"Bearer {credentials['bearer_token']}"
            elif "access_token" in credentials:
                headers["Authorization"] = f"Bearer {credentials['access_token']}"
            
            # Add other headers that might be stored
            for key, value in credentials.items():
                if key.startswith("header_"):
                    header_name = key.replace("header_", "").replace("_", "-").title()
                    headers[header_name] = value
                elif key in ["x-api-key", "x-auth-token", "content-type"]:
                    headers[key] = value
            
            return headers
            
        except Exception as e:
            print(f"[Orchestrator] Error getting auth headers from vault: {e}")
            return {}
    
    async def _evaluate_and_decide_next_step(self, context: WorkflowContext) -> Dict[str, Any]:
        """Step 5: LLM evaluates results and decides on next step."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, f"step_{context.current_step}_evaluate",
            "evaluation", "🤔 Evaluating results and planning next step..."
        )
        
        # Prepare comprehensive context for evaluation
        recent_results = context.agent_results[-3:]  # Last 3 results
        context_summary = await self._prepare_context_summary(context)
        
        # Check for technical failures in the most recent result
        recent_failure = None
        if recent_results and not recent_results[-1].success:
            recent_failure = recent_results[-1]
        
        # Check for loops - if we've executed similar queries multiple times
        recent_queries = [r.query_executed for r in recent_results if r.query_executed]
        query_counts = {}
        for query in recent_queries:
            normalized_query = ' '.join(query.lower().split())
            query_counts[normalized_query] = query_counts.get(normalized_query, 0) + 1
        
        has_loops = any(count >= 2 for count in query_counts.values())
        max_repeated_count = max(query_counts.values()) if query_counts else 0
        
        # Check if this is a technical failure that shouldn't trigger agent switching
        is_technical_failure = (recent_failure and 
                               recent_failure.error and 
                               any(tech_error in recent_failure.error.lower() for tech_error in 
                                   ['syntax error', 'limit', 'sql server', 'database error', 'connection', 'timeout']))
        
        # Check if user query specifically requested data from a particular agent/database
        user_wants_specific_source = any(keyword in context.original_query.lower() for keyword in 
                                       ['mssql', 'sql server', 'autogent', 'retail', 'specific database'])
        
        prompt = f"""
        You are evaluating the progress of a multi-step data analysis workflow.
        
        Original User Query: {context.original_query}
        Current Step: {context.current_step}/{context.max_steps}
        
        Context Summary: {context_summary}
        
        Recent Results:
        {json.dumps([{
            "agent": r.agent_name,
            "success": r.success,
            "query": r.query_executed,
            "row_count": r.data.get("row_count", 0) if r.success else 0,
            "execution_time": r.execution_time,
            "error": r.error
        } for r in recent_results], indent=2)}
        
        All Results Summary: {json.dumps(context.get_execution_summary(), indent=2)}
        
        CRITICAL ANALYSIS:
        - Technical failure detected: {"YES" if is_technical_failure else "NO"}
        - User wants specific data source: {"YES" if user_wants_specific_source else "NO"}
        - Recent failure: {json.dumps({"agent": recent_failure.agent_name, "error": recent_failure.error} if recent_failure else None)}
        
        LOOP DETECTION ALERT: 
        - Similar queries repeated: {"YES" if has_loops else "NO"}
        - Maximum repetition count: {max_repeated_count}
        - Query patterns: {json.dumps(query_counts, indent=2)}
        
        AGENT SWITCHING POLICY:
        1. **NEVER switch agents for technical failures** (SQL syntax, connection issues, etc.)
        2. **NEVER switch agents if user requested specific data source** - they want data from that specific database
        3. **Only switch agents when:**
           - User explicitly requests multi-source analysis
           - Current agent doesn't have relevant data (confirmed by successful query with no results)
           - User query requires complementary data from different sources
        4. **For technical failures:** Either fix the issue and retry the same agent, or complete with error explanation
        
        Instructions:
        1. **CRITICAL**: If this is a technical failure, do NOT continue to another agent - complete the workflow with appropriate error message
        2. **CRITICAL**: If user wants data from a specific source, do NOT switch agents - complete with error explanation if that agent failed
        3. If similar queries have been repeated 2+ times, strongly consider completing the workflow to avoid infinite loops
        4. Evaluate if the original user query has been adequately answered
        5. Consider data quality, completeness, and relevance
        6. **AVOID AGENT SWITCHING** unless user explicitly wants multi-source analysis
        
        Respond with JSON:
        {{
            "action": "continue" | "complete" | "require_user_input",
            "reasoning": "detailed explanation of the decision (must address technical failures and agent switching policy)",
            "completeness_score": 0.85,
            "missing_information": ["list", "of", "gaps"],
            "user_input_request": "what to ask user if action is require_user_input",
            "confidence": 0.90,
            "loop_detected": {str(has_loops).lower()},
            "technical_failure_detected": {str(is_technical_failure).lower()},
            "should_avoid_agent_switching": {str(is_technical_failure or user_wants_specific_source).lower()}
        }}
        """
        
        try:
            decision = self.ollama_client.invoke_with_json_response(prompt, timeout=600)
            if not decision:
                raise Exception("No valid JSON response received from LLM")
            
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"step_{context.current_step}_evaluate",
                "evaluation", 1.5
            )
            
            # Emit decision event
            action = decision.get("action", "complete")
            workflow_streamer.emit_debug_info(
                context.workflow_id, context.session_id, f"step_{context.current_step}_evaluate",
                f"Next step decision: {action}",
                {"decision_details": decision}
            )
            
            return decision
            
        except Exception as e:
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_evaluate",
                f"Failed to evaluate next step: {str(e)}"
            )
            return {"action": "complete"}  # Fail gracefully
    
    async def _request_user_input(self, context: WorkflowContext, decision: Dict[str, Any]):
        """Request user input and pause workflow."""
        input_request = {
            "type": "user_input_required",
            "prompt": decision.get("user_input_request", "Additional information needed"),
            "context": decision.get("missing_information", []),
            "reasoning": decision.get("reasoning", ""),
            "workflow_id": context.workflow_id,
            "step": context.current_step
        }
        
        workflow_streamer.emit_user_input_required(
            context.workflow_id, context.session_id, f"step_{context.current_step}_input",
            input_request
        )
    
    async def _request_user_agent_choice(self, context: WorkflowContext, recommendation: Dict[str, Any]):
        """Request user to choose between multiple agents."""
        choice_options = recommendation.get("user_choice_options", [])
        
        # If no specific options provided, create options from all available agents
        if not choice_options:
            available_agents = self.registry_manager.list_available_agents()
            choice_options = [
                {
                    "id": agent["id"],
                    "name": agent["name"], 
                    "reason": f"Access to {agent.get('description', 'database')}",
                    "expected_outcome": f"Query data from {agent['name']}"
                }
                for agent in available_agents
            ]
        
        # Create user-friendly options list with clear labels
        options_text = ""
        for i, option in enumerate(choice_options, 1):
            options_text += f"\nOption {chr(64 + i)} ({option['name']}): {option.get('reason', option.get('description', 'Database access'))}"
        
        choice_request = {
            "type": "agent_choice_required",
            "prompt": f"I have detected multiple data sources for your query. Can you please tell me which one you are looking for?\n{options_text}\n\nPlease respond with your choice (like 'Option A', 'A', or 'Option A is the right one') or say 'none' if none of these are what you're looking for.",
            "reasoning": recommendation.get("reasoning", "Multiple suitable agents found"),
            "options": choice_options,
            "workflow_id": context.workflow_id,
            "step": context.current_step,
            "allow_multiple": False,  # For now, only single agent selection
            "context_summary": await self._prepare_context_summary(context),
            "cancellation_option": "none"  # Allow user to cancel if no option fits
        }
        
        print(f"[EnhancedOrchestrator] Requesting user agent choice: {len(choice_options)} options")
        
        workflow_streamer.emit_user_input_required(
            context.workflow_id, context.session_id, f"step_{context.current_step}_agent_choice",
            choice_request
        )
    
    async def _generate_final_result(self, context: WorkflowContext) -> Dict[str, Any]:
        """Generate comprehensive final result."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "final_generation",
            "result_generation", "✨ Generating final comprehensive answer..."
        )
        
        # Compile all data and generate final answer
        all_data = []
        for result in context.agent_results:
            if result.success and result.data.get("data"):
                all_data.extend(result.data["data"])
        
        context_summary = await self._prepare_context_summary(context)
        
        # Helper function to handle JSON serialization with datetime objects
        def json_serializer(obj):
            """JSON serializer for objects not serializable by default json code"""
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        # Safe JSON serialization with datetime handling
        try:
            execution_summary_json = json.dumps(context.get_execution_summary(), indent=2, default=json_serializer)
        except Exception as e:
            execution_summary_json = str(context.get_execution_summary())
        
        try:
            all_data_json = json.dumps(all_data[:100], indent=2, default=json_serializer)
        except Exception as e:
            all_data_json = f"[Data contains {len(all_data)} rows - JSON serialization skipped due to datetime objects]"
        
        try:
            agent_results_json = json.dumps([{
                "agent": r.agent_name, 
                "success": r.success,
                "rows": r.data.get("row_count", 0) if r.success else 0,
                "query": r.query_executed
            } for r in context.agent_results], indent=2, default=json_serializer)
        except Exception as e:
            agent_results_json = f"[Agent results summary - {len(context.agent_results)} results processed]"
        
        prompt = f"""
        Generate a comprehensive final answer based on the multi-step analysis results.
        
        Original User Query: {context.original_query}
        
        Execution Summary: {execution_summary_json}
        
        Context: {context_summary}
        
        All Retrieved Data: {all_data_json}
        
        Agent Results: {agent_results_json}
        
        Instructions:
        1. Provide a clear, comprehensive answer to the original question
        2. Highlight key insights from the data
        3. Mention methodology (which agents were used, what data was analyzed)
        4. Include relevant statistics or trends if applicable
        5. Be concise but informative
        
        Format the response as a natural, user-friendly answer.
        """
        
        try:
            final_answer = self.ollama_client.invoke_with_text_response(prompt)
            
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, "final_generation",
                "result_generation", 3.0
            )
            
            return {
                "final_answer": final_answer,
                "execution_summary": context.get_execution_summary(),
                "agents_used": [r.agent_name for r in context.agent_results if r.success],
                "total_data_points": sum(r.data.get("row_count", 0) for r in context.agent_results if r.success),
                "workflow_id": context.workflow_id,
                "session_id": context.session_id
            }
            
        except Exception as e:
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, "final_generation",
                f"Failed to generate final result: {str(e)}"
            )
            
            # Fallback result
            return {
                "final_answer": f"Analysis completed using {len(context.agent_results)} agents. Retrieved {sum(r.data.get('row_count', 0) for r in context.agent_results if r.success)} data points.",
                "execution_summary": context.get_execution_summary(),
                "agents_used": [r.agent_name for r in context.agent_results if r.success],
                "workflow_id": context.workflow_id,
                "session_id": context.session_id
            }
    
    async def _prepare_context_summary(self, context: WorkflowContext) -> str:
        """Prepare a concise context summary for LLM prompts."""
        if not context.agent_results:
            return "No previous results available."
        
        summary_parts = []
        for i, result in enumerate(context.agent_results):
            if result.success:
                row_count = result.data.get("row_count", 0)
                summary_parts.append(f"Step {i+1}: {result.agent_name} - {row_count} rows retrieved")
            else:
                summary_parts.append(f"Step {i+1}: {result.agent_name} - Failed: {result.error}")
        
        # Add conversation context if available
        total_context = f"Execution steps: {', '.join(summary_parts)}"
        
        # Use context manager for conversation summarization if context is getting large
        if len(total_context) > 1000:
            total_context = await self.context_manager.summarize_context(
                context.session_id, total_context
            )
        
        return total_context

    async def _analyze_request_complexity(self, context: WorkflowContext, agent_details: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query complexity and determine if it needs special handling."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, f"step_{context.current_step}_complexity",
            "complexity_analysis", "🧠 Analyzing query complexity..."
        )
        
        # Get database size info
        tables_data = agent_details.get('tables', [])
        total_tables = len(tables_data)
        
        # Calculate estimated table sizes and relationships
        estimated_total_rows = 0
        large_tables = []
        for table in tables_data:
            # More conservative estimation based on table name patterns
            table_name = table.get('name', '').lower()
            if any(keyword in table_name for keyword in ['transaction', 'log', 'event', 'history', 'detail', 'audit']):
                estimated_rows = 500000  # Large transactional tables (500k)
                if estimated_rows > 100000:  # Only flag as large if >100k
                    large_tables.append(table_name)
            elif any(keyword in table_name for keyword in ['customer', 'product', 'order', 'invoice', 'user']):
                estimated_rows = 50000   # Medium master data tables (50k)
            else:
                estimated_rows = 5000    # Small reference tables (5k)
            estimated_total_rows += estimated_rows
        
        context_summary = await self._prepare_context_summary(context)
        
        prompt = f"""
        You are a database performance expert. Analyze the user query for potential complexity and execution impact.
        
        CRITICAL: Only suggest narrowing for queries that would likely take MORE THAN 1 MINUTE to execute or cause significant database load.
        Most normal queries should proceed normally, even if they scan entire tables with moderate data sizes.
        
        Original User Query: {context.original_query}
        Database Agent: {agent_details.get('name', 'Unknown')}
        
        Database Context:
        - Total Tables: {total_tables}
        - Estimated Total Rows: {estimated_total_rows:,}
        - Large Tables (>100k rows): {large_tables}
        - Available Relations: {len(agent_details.get('table_relations', []))}
        
        Previous Context: {context_summary}
        
        ONLY Flag as Complex if Query Has:
        1. Multiple large table joins (>3 tables with >100k rows each) WITHOUT proper indexing
        2. Aggregations across MILLIONS of rows without any filtering
        3. Full text search across very large text fields (>1M records)
        4. Complex analytical queries with multiple window functions over large datasets
        5. Recursive queries or deep CTEs that could cause exponential growth
        6. Cross joins or Cartesian products that could result in billions of combinations
        7. Queries explicitly asking for "ALL historical data" from very large transactional tables
        
        Normal Queries That Should PROCEED:
        - Simple SELECTs with reasonable limits (even without WHERE clauses)
        - Queries on tables under 100k rows regardless of complexity
        - Basic JOINs between 2-3 tables
        - Standard aggregations with reasonable data sizes
        - Most business reporting queries
        - Customer, product, or order lookups
        
        Estimated Execution Time Guidelines:
        - <30 seconds: Always proceed
        - 30-60 seconds: Proceed (acceptable wait time)
        - 1-5 minutes: Consider breaking down
        - >5 minutes: Suggest narrowing
        
        Respond with JSON:
        {{
            "action": "proceed" | "break_down" | "suggest_narrowing",
            "complexity_score": 0.3,
            "reasoning": "detailed analysis of why this query needs special handling or can proceed normally",
            "performance_concerns": ["specific issues that would cause >1min execution"],
            "suggestions": ["concrete suggestions only if suggesting narrowing"],
            "breakdown_strategy": {{
                "approach": "date_chunks" | "entity_groups" | "table_sequence",
                "chunks": [
                    {{"description": "time period 1", "filter": "specific filter condition"}},
                    {{"description": "time period 2", "filter": "specific filter condition"}}
                ]
            }},
            "estimated_execution_time": "<30 seconds" | "30-60 seconds" | "1-5 minutes" | ">5 minutes",
            "database_load_impact": "high" | "medium" | "low"
        }}
        """
        
        try:
            analysis = self.ollama_client.invoke_with_json_response(prompt, timeout=300)
            if not analysis:
                # Default to proceed if analysis fails
                analysis = {"action": "proceed", "complexity_score": 0.5, "reasoning": "Analysis inconclusive, proceeding with caution"}
            
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"step_{context.current_step}_complexity",
                "complexity_analysis", 2.0
            )
            
            # Emit complexity analysis event
            complexity_score = analysis.get("complexity_score", 0.5)
            action = analysis.get("action", "proceed")
            
            workflow_streamer.emit_debug_info(
                context.workflow_id, context.session_id, f"step_{context.current_step}_complexity",
                f"Query complexity: {complexity_score:.2f} - Action: {action}",
                {"complexity_analysis": analysis}
            )
            
            return analysis
            
        except Exception as e:
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_complexity",
                f"Complexity analysis failed: {str(e)}"
            )
            # Default to proceed on error
            return {"action": "proceed", "complexity_score": 0.5, "reasoning": f"Analysis failed: {str(e)}"}

    async def _prepare_chunked_request(self, context: WorkflowContext, agent_details: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a chunked request that will be executed in multiple parts."""
        breakdown_strategy = analysis.get("breakdown_strategy", {})
        chunks = breakdown_strategy.get("chunks", [])
        
        if not chunks:
            # Fall back to single query if no chunks defined
            return await self._prepare_single_query_request(context, agent_details)
        
        # Prepare the first chunk
        first_chunk = chunks[0]
        context_summary = await self._prepare_context_summary(context)
        
        # Store chunking information in context for later use
        if not hasattr(context, 'chunk_info'):
            context.chunk_info = {
                "total_chunks": len(chunks),
                "current_chunk": 0,
                "chunks": chunks,
                "breakdown_strategy": breakdown_strategy
            }
        
        from app.ollama_client import ollama_client
        column_reference = ""
        tables_data = agent_details.get('tables', [])
        if tables_data:
            column_reference = ollama_client._extract_column_reference_from_structured_data(tables_data)
        
        database_type = agent_details.get('database_type', 'unknown')
        agent_name = agent_details.get('name', 'Unknown')
        schema_info = agent_details.get('schema', '')
        
        prompt = f"""
        You are generating the first part of a chunked query strategy to handle a complex user request.
        
        USER QUERY: {context.original_query}
        
        Chunking Strategy: {breakdown_strategy.get('approach', 'sequential')}
        Current Chunk: {first_chunk.get('description', 'First chunk')}
        Chunk Filter: {first_chunk.get('filter', 'No specific filter')}
        Total Chunks Planned: {len(chunks)}
        
        DATABASE DETAILS:
        - Database Type: {database_type}
        - Agent Name: {agent_name}
        
        EXACT COLUMN VALIDATION:
        {column_reference}
        
        DATABASE SCHEMA:
        {schema_info}
        
        CONTEXT FROM PREVIOUS STEPS:
        {context_summary}
        
        Instructions:
        1. Generate a SQL query for ONLY the first chunk of data
        2. Apply the chunk filter to limit the scope
        3. Focus on the most relevant data for this chunk
        4. Use ONLY exact column names from the schema
        5. Keep the query efficient and targeted
        
        Respond with JSON:
        {{
            "column_validation": "Step 1: Tables needed: [...]. Step 2: Exact columns from schema: [...]. Step 3: Verified all columns exist in schema.",
            "sql_query": "SELECT ... with chunk filtering applied",
            "query_purpose": "what this chunked query will accomplish", 
            "expected_data": "description of expected results for this chunk",
            "chunk_info": {{
                "chunk_number": 1,
                "total_chunks": {len(chunks)},
                "description": "{first_chunk.get('description', 'First chunk')}",
                "filter_applied": "{first_chunk.get('filter', 'None')}"
            }},
            "is_chunked": true,
            "confidence": 0.90
        }}
        """
        
        try:
            request_details = self.ollama_client.invoke_with_json_response(prompt, timeout=600)
            if not request_details:
                # Fall back to single query
                return await self._prepare_single_query_request(context, agent_details)
            
            workflow_streamer.emit_sql_generated(
                context.workflow_id, context.session_id,
                request_details["sql_query"], agent_details.get("database_type", "unknown"),
                f"Chunked Query 1/{len(chunks)}: " + request_details.get("query_purpose", "")
            )
            
            return request_details
            
        except Exception as e:
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_prepare_chunk",
                f"Failed to prepare chunked request: {str(e)}"
            )
            # Fall back to single query
            return await self._prepare_single_query_request(context, agent_details)

    async def _prepare_single_query_request(self, context: WorkflowContext, agent_details: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a standard single query request."""
        context_summary = await self._prepare_context_summary(context)
        
        # Extract schema information for the prompt
        database_type = agent_details.get('database_type', 'unknown')
        agent_name = agent_details.get('name', 'Unknown')
        description = agent_details.get('description', '')
        schema_info = agent_details.get('schema', '')
        sample_queries = agent_details.get('sample_queries', [])
        
        # Generate column reference from LLM client like in langgraph
        from app.ollama_client import ollama_client
        column_reference = ""
        tables_data = agent_details.get('tables', [])
        if tables_data:
            column_reference = ollama_client._extract_column_reference_from_structured_data(tables_data)
        else:
            column_reference = "⚠️  Use ONLY exact column names from the schema below"
        
        prompt = f"""
        You are an expert SQL developer. Generate a structured response for the user's SQL query request.

        ⚠️ CRITICAL: You MUST use ONLY the exact column names from the schema below. DO NOT assume or invent any column names.

        USER QUERY: {context.original_query}

        DATABASE DETAILS:
        - Database Type: {database_type}
        - Agent Name: {agent_name}
        - Description: {description}
        - Total Tables: {len(agent_details.get('tables', []))}
        - Total Relations: {len(agent_details.get('table_relations', []))}

        EXACT COLUMN VALIDATION:
        {column_reference}

        DATABASE SCHEMA:
        {schema_info}

        SAMPLE QUERIES FOR REFERENCE:
        {chr(10).join(sample_queries[:3])}

        CONTEXT FROM PREVIOUS STEPS:
        {context_summary}

        Previous Agent Results:
        {json.dumps([{"agent": r.agent_name, "query": r.query_executed, "success": r.success, "error": r.error} for r in context.agent_results], indent=2)}

        Instructions:
        1. Generate a specific SQL query that uses ONLY the exact table and column names shown above
        2. Focus on data that hasn't been retrieved yet or needs different analysis  
        3. Consider the agent's available tables and relationships
        4. Make the query targeted and efficient
        5. Validate all column names against the schema before finalizing
        
        Respond with JSON:
        {{
            "column_validation": "Step 1: Tables needed: [...]. Step 2: Exact columns from schema: [...]. Step 3: Verified all columns exist in schema.",
            "sql_query": "SELECT ... your query here using EXACT column names",
            "query_purpose": "what this query will accomplish", 
            "expected_data": "description of expected results",
            "reasoning": "why this specific query addresses the user's needs",
            "confidence": 0.95
        }}
        """
        
        try:
            request_details = self.ollama_client.invoke_with_json_response(prompt, timeout=600)
            if not request_details:
                raise Exception("No valid JSON response received from LLM")
            
            workflow_streamer.emit_sql_generated(
                context.workflow_id, context.session_id,
                request_details["sql_query"], agent_details.get("database_type", "unknown"),
                request_details.get("query_purpose")
            )
            
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"step_{context.current_step}_prepare",
                "request_preparation", 2.0
            )
            
            return request_details
            
        except Exception as e:
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_prepare",
                f"Failed to prepare agent request: {str(e)}"
            )
            return {}
        
    async def resume_workflow_with_user_choice(self, workflow_id: str, session_id: str, user_choice: Dict[str, Any]) -> Dict[str, Any]:
        """Resume workflow after user makes a choice."""
        print(f"[EnhancedOrchestrator] Resuming workflow {workflow_id} with user choice: {user_choice}")
        
        # For now, we'll implement a simplified version that continues with the chosen agent
        # In a full implementation, you'd want to maintain workflow state across requests
        choice_type = user_choice.get("type")
        
        if choice_type == "agent_choice":
            chosen_agent_id = user_choice.get("agent_id")
            user_message = user_choice.get("message", "")
            
            print(f"[EnhancedOrchestrator] User chose agent: {chosen_agent_id}, message: '{user_message}'")
            
            # Check if user said "none" or equivalent cancellation
            if self._is_cancellation_response(user_message):
                return {
                    "status": "cancelled",
                    "message": "Sorry, we couldn't help you with your request. None of the available data sources match what you're looking for.",
                    "workflow_id": workflow_id,
                    "session_id": session_id
                }
            
            # Parse natural language response to extract option choice
            if not chosen_agent_id and user_message:
                chosen_agent_id = self._parse_option_choice(user_message, user_choice.get("available_options", []))
            
            # Get the chosen agent details
            available_agents = self.registry_manager.list_available_agents()
            chosen_agent = next((a for a in available_agents if a["id"] == chosen_agent_id), None)
            
            if not chosen_agent:
                return {
                    "status": "error",
                    "message": f"I couldn't identify which data source you selected from your response: '{user_message}'. Please try again with a clearer choice like 'Option A' or the name of the data source."
                }
            
            # Create a simplified context for the chosen agent
            # In a real implementation, you'd restore the full workflow context
            simplified_context = WorkflowContext(
                session_id=session_id,
                workflow_id=workflow_id,
                original_query=user_message or "User selected agent continuation",
                conversation_history=[],
                agent_results=[],
                current_step=0
            )
            
            # Execute just this agent with user's message
            if user_message:
                agent_request = await self._prepare_agent_request(simplified_context, chosen_agent)
                execution_result = await self._execute_agent(simplified_context, chosen_agent, agent_request)
                
                # Generate response based on the execution
                final_result = await self._generate_single_agent_result(simplified_context, chosen_agent, execution_result, user_message)
                
                return {
                    "status": "completed",
                    "final_answer": final_result["final_answer"],
                    "execution_summary": {
                        "user_choice": True,
                        "chosen_agent": chosen_agent["name"],
                        "rows_retrieved": execution_result.data.get("row_count", 0) if execution_result.success else 0
                    },
                    "workflow_id": workflow_id,
                    "session_id": session_id
                }
            else:
                return {
                    "status": "waiting_for_query",
                    "message": f"Agent {chosen_agent['name']} selected. Please provide your specific query.",
                    "chosen_agent": chosen_agent["name"]
                }
                
        elif choice_type == "user_input":
            user_input = user_choice.get("input", "")
            print(f"[EnhancedOrchestrator] User provided input: '{user_input}'")
            
            # Check if user wants to cancel
            if self._is_cancellation_response(user_input):
                return {
                    "status": "cancelled",
                    "message": "Sorry, we couldn't help you with your request based on the information provided.",
                    "workflow_id": workflow_id,
                    "session_id": session_id
                }
            
            # Continue workflow with the user input
            # This would require maintaining workflow state - for now return acknowledgment
            return {
                "status": "continued",
                "message": f"Continuing workflow with your input: '{user_input}'",
                "workflow_id": workflow_id
            }
        
        return {
            "status": "error", 
            "message": "Invalid user choice type"
        }
        
    async def _generate_single_agent_result(self, context: WorkflowContext, agent: Dict[str, Any], execution_result, user_query: str) -> Dict[str, Any]:
        """Generate result from a single agent execution after user choice."""
        
        if not execution_result.success:
            return {
                "final_answer": f"I encountered an error while querying {agent['name']}: {execution_result.error}",
                "execution_summary": {"error": True, "agent": agent["name"]},
                "workflow_id": context.workflow_id,
                "session_id": context.session_id
            }
        
        # Generate a response based on the single agent result
        row_count = execution_result.data.get("row_count", 0)
        data_sample = execution_result.data.get("data", [])[:5]  # First 5 rows for summary
        
        prompt = f"""
        Generate a user-friendly response based on a single database query result.
        
        User Query: {user_query}
        Database Agent: {agent['name']}
        SQL Query: {execution_result.query_executed}
        Rows Retrieved: {row_count}
        Sample Data: {json.dumps(data_sample, indent=2, default=str)}
        
        Instructions:
        1. Provide a clear, concise answer to the user's query
        2. Mention the number of results found
        3. Highlight key information from the sample data
        4. Be conversational and helpful
        
        Format as a natural response to the user.
        """
        
        try:
            response = self.ollama_client.invoke_with_text_response(prompt)
            return {
                "final_answer": response,
                "execution_summary": {
                    "agent_used": agent["name"],
                    "rows_retrieved": row_count,
                    "query_executed": execution_result.query_executed
                },
                "workflow_id": context.workflow_id,
                "session_id": context.session_id
            }
        except Exception as e:
            return {
                "final_answer": f"I found {row_count} results from {agent['name']}, but had trouble formatting the response. The data includes: {', '.join([str(item) for item in data_sample[:3]])}..." if row_count > 0 else f"No results found in {agent['name']} for your query.",
                "execution_summary": {
                    "agent_used": agent["name"], 
                    "rows_retrieved": row_count,
                    "formatting_error": str(e)
                },
                "workflow_id": context.workflow_id,
                "session_id": context.session_id
            }


# Global instance
enhanced_orchestrator = EnhancedOrchestrator()
