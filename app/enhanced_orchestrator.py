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
from app.llm_client import LLMClient
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
        self.llm_client = LLMClient()
        self.registry_manager = RegistryManager()
        self.db_executor = DatabaseQueryExecutor()
        self.context_manager = ContextManager()
        # Store workflow contexts in memory (in production, use persistent storage)
        self.workflow_contexts: Dict[str, WorkflowContext] = {}
        
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
        
        # Store workflow context for later retrieval
        self.workflow_contexts[workflow_id] = context
        
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
                    # Return partial result indicating user choice is required
                    return {
                        "status": "user_choice_required",
                        "workflow_id": context.workflow_id,
                        "session_id": context.session_id,
                        "current_step": context.current_step,
                        "message": "Multiple agents available - user choice required to continue",
                        "choice_options": agent_recommendation.get("user_choice_options", [])
                    }
                elif agent_recommendation.get("action") == "require_user_input":
                    # LLM needs additional information from user
                    await self._request_user_input(context, agent_recommendation)
                    # Return partial result indicating user input is required
                    return {
                        "status": "user_input_required", 
                        "workflow_id": context.workflow_id,
                        "session_id": context.session_id,
                        "current_step": context.current_step,
                        "message": "Additional information required to continue",
                        "input_request": agent_recommendation.get("user_input_request", "")
                    }
                
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
                    # Return partial result indicating query narrowing is required
                    return {
                        "status": "query_narrowing_required",
                        "workflow_id": context.workflow_id,
                        "session_id": context.session_id,
                        "current_step": context.current_step,
                        "message": "Query is too complex - please narrow the scope",
                        "narrowing_suggestions": agent_request.get("suggestions", [])
                    }
                
                # Validate request based on agent type
                is_valid_request = False
                if selected_agent.get("agent_type") == "application":
                    # For applications, check for API request details
                    is_valid_request = bool(agent_request and agent_request.get("api_url"))
                else:
                    # For data agents, check for SQL query
                    is_valid_request = bool(agent_request and agent_request.get("sql_query"))
                
                if not is_valid_request:
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
                    # Return partial result indicating user input is required
                    return {
                        "status": "user_input_required",
                        "workflow_id": context.workflow_id, 
                        "session_id": context.session_id,
                        "current_step": context.current_step,
                        "message": "Additional information required to continue analysis",
                        "input_request": next_step_decision.get("user_input_request", "")
                    }
            
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
                "agent_type": a.get("agent_type", "data_agent"),
                "region": self._extract_region_from_agent(a)
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
                agent_info["endpoint_count"] = len(endpoints)
            else:
                # For data agents, get enhanced details for better context
                try:
                    from app.registry import get_enhanced_agent_details_for_llm
                    agent_details = get_enhanced_agent_details_for_llm(a["id"])
                    if agent_details:
                        agent_info["capabilities"] = {
                            "database_type": agent_details.get("database_type", ""),
                            "description": agent_details.get("description", a.get("description", "")),
                            "table_count": len(agent_details.get("tables", [])),
                            "sample_tables": [t.get("name", "") for t in agent_details.get("tables", [])[:3]]
                        }
                    else:
                        agent_info["capabilities"] = {
                            "database_type": a.get("database_type", ""),
                            "description": a.get("description", "")
                        }
                except:
                    # Fallback if enhanced details are not available
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
        2. PREFER making intelligent agent selections over asking for user choice
        3. For ambiguous queries, choose the MOST COMPREHENSIVE or MOST LIKELY agent based on:
           - Data completeness (more tables/data is usually better)
           - Geographic preference (US agents for general queries unless specified otherwise)
           - Agent capabilities and data richness
        4. For APPLICATIONS (agent_type: "application"), use their primary identifier (id field, which is the appKey)
        5. For DATA AGENTS (agent_type: "data_agent"), use their database_id field
        6. Only require user choice if agents are FUNDAMENTALLY different domains (e.g., retail vs financial vs HR)
        7. For similar agents in different regions, default to the most comprehensive one
        8. If you need additional context or information from the user, return {{"action": "require_user_input"}}
        
        Examples of automatic selection (PREFERRED):
        - Query: "retail sales data" → Select the agent with most tables/comprehensive data
        - Query: "customer data" → Select the most complete customer database
        - Query: "store information" → Select the retail agent with store data
        
        Only require user choice for truly different domains:
        - Query: "sales data" where options are: Retail DB, Financial DB, HR DB
        
        IMPORTANT: When recommending an agent, use the "id" field from the Available Agents list above as the recommended_agent_id.
        
        Respond with JSON:
        {{
            "action": "recommend_agent" | "complete" | "require_user_choice" | "require_user_input",
            "recommended_agent_id": "agent_id_if_recommending",
            "reasoning": "detailed explanation of why this specific agent was selected OR why user choice is required for fundamentally different domains",
            "expected_outcome": "what this agent should accomplish if recommending single agent",
            "confidence": 0.95,
            "user_choice_options": [
                {{"id": "agent1", "name": "Agent Name", "region": "geographical or business area", "description": "detailed description of what this agent provides", "reason": "why this agent matches the query", "expected_outcome": "what data/results it would provide"}},
                {{"id": "agent2", "name": "Agent Name", "region": "geographical or business area", "description": "detailed description of what this agent provides", "reason": "why this agent matches the query", "expected_outcome": "what data/results it would provide"}}
            ],
            "user_input_request": "what specific information to ask the user",
            "ambiguity_detected": true/false,
            "matching_agents_count": 0,
            "selection_strategy": "automatic_best_match | user_choice_required | insufficient_info"
        }}
        """
        
        try:
            print(f"[EnhancedOrchestrator] Calling LLM for agent recommendation...")
            recommendation = self.llm_client.invoke_with_json_response(prompt, timeout=600)
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
        
        # Get agent details from registry first to get the friendly name
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
        
        # Use the agent name for display, fallback to agent_id if not found
        display_name = selected_agent.get("name", agent_id) if selected_agent else agent_id
        
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, f"step_{context.current_step}_select",
            "agent_selection", f"🎯 Selecting agent: {display_name}..."
        )
        
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
        # Get application details from registry using appKey (the agent id that LLM recommends)
        from app.registry import get_enhanced_agent_details_for_llm
        agent_lookup_id = agent.get("appKey") or agent["id"]  # Use appKey if available, fallback to id
        app_details = get_enhanced_agent_details_for_llm(agent_lookup_id)
        
        if not app_details:
            workflow_streamer.emit_error(
                context.workflow_id, context.session_id, f"step_{context.current_step}_prepare",
                f"Application details not found for {agent_lookup_id} (appKey: {agent.get('appKey')}, id: {agent.get('id')})"
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
            request_details = self.llm_client.invoke_with_json_response(prompt, timeout=600)
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
        from app.llm_client import llm_client
        column_reference = ""
        tables_data = agent_details.get('tables', [])
        if tables_data:
            column_reference = llm_client._extract_column_reference_from_structured_data(tables_data)
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
            request_details = self.llm_client.invoke_with_json_response(prompt, timeout=600)
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
        
        # Check for loops - if we've executed similar queries multiple times
        recent_queries = [r.query_executed for r in recent_results if r.query_executed]
        query_counts = {}
        for query in recent_queries:
            normalized_query = ' '.join(query.lower().split())
            query_counts[normalized_query] = query_counts.get(normalized_query, 0) + 1
        
        has_loops = any(count >= 2 for count in query_counts.values())
        max_repeated_count = max(query_counts.values()) if query_counts else 0
        
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
        
        LOOP DETECTION ALERT: 
        - Similar queries repeated: {"YES" if has_loops else "NO"}
        - Maximum repetition count: {max_repeated_count}
        - Query patterns: {json.dumps(query_counts, indent=2)}
        
        Instructions:
        1. **CRITICAL**: If similar queries have been repeated 2+ times, strongly consider completing the workflow to avoid infinite loops
        2. Evaluate if the original user query has been adequately answered
        3. Consider data quality, completeness, and relevance
        4. Decide if more data gathering or analysis is needed
        5. Consider if user input is required for clarification
        6. **AVOID INFINITE LOOPS** - aim for completion when reasonable, especially if loops detected
        
        Respond with JSON:
        {{
            "action": "continue" | "complete" | "require_user_input",
            "reasoning": "detailed explanation of the decision (must address loop detection if applicable)",
            "completeness_score": 0.85,
            "missing_information": ["list", "of", "gaps"],
            "user_input_request": "what to ask user if action is require_user_input",
            "confidence": 0.90,
            "loop_detected": {str(has_loops).lower()}
        }}
        """
        
        try:
            decision = self.llm_client.invoke_with_json_response(prompt, timeout=600)
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
        """Request user to choose between multiple agents with enhanced descriptions."""
        choice_options = recommendation.get("user_choice_options", [])
        
        # If no specific options provided, create enhanced options from all available agents
        if not choice_options:
            available_agents = self.registry_manager.list_available_agents()
            
            # Get detailed descriptions for ambiguous agents
            enhanced_options = []
            for agent in available_agents:
                # Get enhanced agent details for better description
                from app.registry import get_enhanced_agent_details_for_llm
                agent_details = get_enhanced_agent_details_for_llm(agent["id"])
                
                enhanced_option = {
                    "id": agent.get("appKey") or agent["id"],
                    "name": agent["name"],
                    "region": self._extract_region_from_agent(agent),
                    "description": agent_details.get("description", agent.get("description", "")),
                    "agent_type": agent.get("agent_type", "data_agent"),
                    "reason": f"Access to {agent.get('description', 'database')}",
                    "expected_outcome": f"Query data from {agent['name']}"
                }
                
                # Add specific details based on agent type
                if agent.get("agent_type") == "application":
                    enhanced_option["capabilities"] = [
                        f"{ep.get('method', 'GET')} {ep.get('path', '')}: {ep.get('description', '')}"
                        for ep in agent.get("endpoints", [])[:3]  # Limit to first 3 endpoints
                    ]
                    enhanced_option["expected_outcome"] = f"API calls to {agent['name']} application"
                else:
                    # For data agents, include database info
                    enhanced_option["database_type"] = agent.get("database_type", "unknown")
                    enhanced_option["table_count"] = len(agent_details.get("tables", [])) if agent_details else 0
                    enhanced_option["expected_outcome"] = f"SQL queries on {agent['name']} database ({enhanced_option['table_count']} tables)"
                
                enhanced_options.append(enhanced_option)
            
            choice_options = enhanced_options
        
        # Enhance existing options if they don't have full details
        enhanced_choice_options = []
        for option in choice_options:
            if not option.get("description") or len(option.get("description", "")) < 20:
                # Get more detailed description from registry
                try:
                    from app.registry import get_enhanced_agent_details_for_llm
                    agent_details = get_enhanced_agent_details_for_llm(option["id"])
                    if agent_details:
                        option["description"] = agent_details.get("description", option.get("description", ""))
                        option["region"] = self._extract_region_from_name(option["name"])
                        
                        # Add database context for data agents
                        if not option.get("agent_type") or option.get("agent_type") == "data_agent":
                            tables = agent_details.get("tables", [])
                            if tables:
                                option["table_info"] = f"{len(tables)} tables available"
                                option["sample_tables"] = [t.get("name", "") for t in tables[:3]]
                except:
                    pass  # Continue with existing option if enhancement fails
            
            enhanced_choice_options.append(option)
        
        # Create a summary of options for the prompt
        options_summary = []
        for i, option in enumerate(enhanced_choice_options, 1):
            options_summary.append(f"{i}. {option.get('name', 'Unknown')} (ID: {option.get('id', 'N/A')}) - {option.get('region', 'Unknown')} - {option.get('description', 'No description')[:100]}...")
        
        prompt_text = f"Multiple agents could handle your request. Please choose the most appropriate one:\n\n" + "\n".join(options_summary)
        
        choice_request = {
            "type": "agent_choice_required",
            "prompt": prompt_text,
            "reasoning": recommendation.get("reasoning", "Multiple suitable agents found - need clarification to proceed with the right data source"),
            "options": enhanced_choice_options,
            "workflow_id": context.workflow_id,
            "step": context.current_step,
            "allow_multiple": False,
            "context_summary": await self._prepare_context_summary(context),
            "ambiguity_info": {
                "query": context.original_query,
                "matching_count": len(enhanced_choice_options),
                "ambiguity_reason": recommendation.get("reasoning", ""),
                "recommendation_confidence": recommendation.get("confidence", 0.0)
            }
        }
        
        print(f"[EnhancedOrchestrator] Requesting user agent choice: {len(enhanced_choice_options)} enhanced options")
        for i, option in enumerate(enhanced_choice_options):
            print(f"[EnhancedOrchestrator] Option {i+1}: ID={option.get('id', 'N/A')}, Name={option.get('name', 'N/A')}, Region={option.get('region', 'N/A')}")
        
        print(f"[EnhancedOrchestrator] Choice request prompt: {choice_request['prompt']}")
        
        workflow_streamer.emit_user_input_required(
            context.workflow_id, context.session_id, f"step_{context.current_step}_agent_choice",
            choice_request
        )
    
    def _extract_region_from_agent(self, agent: Dict[str, Any]) -> str:
        """Extract geographical or business region from agent information."""
        name = agent.get("name", "").lower()
        description = agent.get("description", "").lower()
        combined_text = f"{name} {description}"
        
        # Common region patterns
        region_patterns = {
            "us": ["us", "united states", "america", "usa"],
            "canada": ["canada", "canadian", "ca"],
            "latam": ["latam", "latin america", "south america", "mexico", "brazil"],
            "europe": ["europe", "eu", "emea", "uk", "germany", "france"],
            "asia": ["asia", "apac", "japan", "china", "india"],
            "global": ["global", "worldwide", "international"]
        }
        
        for region, patterns in region_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                return region.upper()
        
        return "Unknown Region"
    
    def _extract_region_from_name(self, name: str) -> str:
        """Extract region from agent name."""
        name_lower = name.lower()
        
        region_keywords = {
            "US": ["us", "united states", "america", "usa"],
            "Canada": ["canada", "canadian"],
            "LATAM": ["latam", "latin", "mexico", "brazil"],
            "Europe": ["europe", "eu", "uk", "emea"],
            "Asia": ["asia", "apac", "japan", "china"],
            "Global": ["global", "worldwide"]
        }
        
        for region, keywords in region_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                return region
        
        return "Regional"
    
    async def _generate_final_result(self, context: WorkflowContext) -> Dict[str, Any]:
        """Generate comprehensive final result."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "final_generation",
            "result_generation", "✨ Generating final comprehensive answer..."
        )
        
        # Compile all data and generate final answer
        all_data = []
        data_summaries = []
        
        for result in context.agent_results:
            if result.success:
                # Handle different types of agent results
                if "response_data" in result.data:
                    # API response from application agents
                    api_data = result.data.get("response_data", {})
                    all_data.append({
                        "agent_type": "application",
                        "agent_name": result.agent_name,
                        "request": result.query_executed,
                        "response": api_data,
                        "status_code": result.data.get("status_code")
                    })
                    data_summaries.append(f"API response from {result.agent_name}: {len(str(api_data))} chars")
                elif "data" in result.data:
                    # Database response from data agents
                    db_data = result.data.get("data", [])
                    if db_data:
                        all_data.extend(db_data)
                        data_summaries.append(f"Database query from {result.agent_name}: {len(db_data)} rows")
                    else:
                        data_summaries.append(f"Database query from {result.agent_name}: No data returned")
                else:
                    # Generic response handling - check if it's the direct result structure
                    if hasattr(result.data, 'get'):
                        raw_data = result.data.get("data", [])
                        if raw_data:
                            all_data.extend(raw_data)
                            data_summaries.append(f"Data from {result.agent_name}: {len(raw_data)} rows")
                        else:
                            all_data.append({
                                "agent_name": result.agent_name,
                                "result_data": result.data,
                                "row_count": result.data.get("row_count", 0)
                            })
                            data_summaries.append(f"Result from {result.agent_name}: {result.data.get('row_count', 0)} rows")
                    else:
                        all_data.append({
                            "agent_name": result.agent_name,
                            "data": result.data
                        })
                        data_summaries.append(f"Generic data from {result.agent_name}")
        
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
        
        # Comprehensive data analysis - provide meaningful insights not just samples
        try:
            if all_data:
                # Generate comprehensive data analysis and statistics
                data_analysis = await self._analyze_retrieved_data(all_data, context.original_query)
                
                # Also include sample data for context
                sample_data = all_data[:3] if len(all_data) > 3 else all_data
                
                data_info = {
                    "total_records": len(all_data),
                    "data_sources": data_summaries,
                    "data_analysis": data_analysis,
                    "sample_records": sample_data,
                    "data_structure": {
                        "columns": list(all_data[0].keys()) if all_data and isinstance(all_data[0], dict) else [],
                        "record_count": len(all_data)
                    }
                }
                all_data_json = json.dumps(data_info, indent=2, default=json_serializer)
            else:
                all_data_json = json.dumps({
                    "total_records": 0,
                    "data_sources": data_summaries,
                    "message": "No data retrieved from agents"
                }, indent=2)
        except Exception as e:
            # Fallback: provide more data if analysis fails
            try:
                # Provide more substantial sample (50 records instead of 5)
                substantial_sample = all_data[:50] if len(all_data) > 50 else all_data
                data_info = {
                    "total_records": len(all_data),
                    "data_sources": data_summaries,
                    "substantial_sample": substantial_sample,
                    "columns": list(all_data[0].keys()) if all_data and isinstance(all_data[0], dict) else []
                }
                all_data_json = json.dumps(data_info, indent=2, default=json_serializer)
            except:
                all_data_json = f"[Data contains {len(all_data)} records from {len(data_summaries)} sources - {', '.join(data_summaries)}]"
        
        try:
            agent_results_summary = []
            for r in context.agent_results:
                if r.success:
                    if "response_data" in r.data:
                        # API response from application
                        result_summary = {
                            "agent": r.agent_name,
                            "type": "application",
                            "success": r.success,
                            "status_code": r.data.get("status_code"),
                            "request": r.query_executed,
                            "response_size": len(str(r.data.get("response_data", {})))
                        }
                    else:
                        # Database response from data agent
                        result_summary = {
                            "agent": r.agent_name,
                            "type": "data_agent", 
                            "success": r.success,
                            "rows": r.data.get("row_count", 0),
                            "query": r.query_executed
                        }
                else:
                    result_summary = {
                        "agent": r.agent_name,
                        "success": r.success,
                        "error": r.error
                    }
                agent_results_summary.append(result_summary)
            
            agent_results_json = json.dumps(agent_results_summary, indent=2, default=json_serializer)
        except Exception as e:
            agent_results_json = f"[Agent results summary - {len(context.agent_results)} results processed]"
        
        prompt = f"""
        You are providing a final answer to the user's question based on comprehensive data analysis results.
        
        User's Original Question: {context.original_query}
        
        Data Analysis Results: {all_data_json}
        
        Context: {context_summary}
        
        Agent Execution Summary: {agent_results_json}
        
        Instructions:
        - Answer the user's question directly using the comprehensive data analysis provided
        - The data_analysis section contains pre-calculated statistics, summaries, and insights from the FULL dataset
        - Use the numeric_summary for totals, averages, min/max values
        - Use the categorical_breakdown for top performers, regional data, product insights
        - Use the date_ranges for time period analysis
        - The sample_records are just for context - base your analysis on the comprehensive statistics
        - For sales questions: focus on total amounts, top regions/products, date ranges
        - Provide specific numbers and concrete insights from the analysis
        - Write in a conversational, helpful tone that directly addresses what the user wants to know
        - If substantial data was analyzed (check total_records), provide meaningful business insights
        
        CRITICAL: Use the comprehensive data_analysis statistics rather than just the sample records. 
        The analysis contains insights from the complete dataset of {len(all_data) if 'all_data' in locals() else 'X'} records.
        
        Provide a detailed, insightful response with specific findings from the full dataset analysis.
        """
        
        try:
            final_answer = self.llm_client.invoke_with_text_response(prompt)
            
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

    async def _analyze_retrieved_data(self, all_data: list, original_query: str) -> Dict[str, Any]:
        """Analyze the full dataset to provide comprehensive insights for the LLM."""
        try:
            if not all_data or not isinstance(all_data, list):
                return {"error": "No valid data to analyze"}
            
            analysis = {}
            
            # Basic data stats
            analysis["total_records"] = len(all_data)
            
            # If data contains dictionaries (typical database rows)
            if all_data and isinstance(all_data[0], dict):
                analysis["columns"] = list(all_data[0].keys())
                
                # Analyze numeric columns for sales insights
                numeric_insights = {}
                date_insights = {}
                categorical_insights = {}
                
                for record in all_data:
                    for key, value in record.items():
                        key_lower = key.lower()
                        
                        # Analyze numeric fields (sales, amounts, quantities, etc.)
                        if isinstance(value, (int, float)) and value is not None:
                            if key not in numeric_insights:
                                numeric_insights[key] = {"values": [], "sum": 0, "count": 0}
                            numeric_insights[key]["values"].append(value)
                            numeric_insights[key]["sum"] += value
                            numeric_insights[key]["count"] += 1
                        
                        # Analyze date fields
                        elif isinstance(value, str) and any(date_term in key_lower for date_term in ['date', 'time', 'created', 'updated']):
                            if key not in date_insights:
                                date_insights[key] = {"values": set(), "count": 0}
                            date_insights[key]["values"].add(value[:10] if len(value) > 10 else value)  # Extract date part
                            date_insights[key]["count"] += 1
                        
                        # Analyze categorical fields (regions, products, customers, etc.)
                        elif isinstance(value, str) and value:
                            if key not in categorical_insights:
                                categorical_insights[key] = {}
                            categorical_insights[key][value] = categorical_insights[key].get(value, 0) + 1
                
                # Generate summaries
                if numeric_insights:
                    analysis["numeric_summary"] = {}
                    for field, data in numeric_insights.items():
                        values = data["values"]
                        analysis["numeric_summary"][field] = {
                            "total": data["sum"],
                            "count": data["count"],
                            "average": round(data["sum"] / data["count"], 2) if data["count"] > 0 else 0,
                            "min": min(values) if values else 0,
                            "max": max(values) if values else 0
                        }
                
                if date_insights:
                    analysis["date_ranges"] = {}
                    for field, data in date_insights.items():
                        sorted_dates = sorted(list(data["values"]))
                        analysis["date_ranges"][field] = {
                            "earliest": sorted_dates[0] if sorted_dates else None,
                            "latest": sorted_dates[-1] if sorted_dates else None,
                            "unique_dates": len(sorted_dates),
                            "total_records": data["count"]
                        }
                
                if categorical_insights:
                    analysis["categorical_breakdown"] = {}
                    for field, counts in categorical_insights.items():
                        # Only include top categories to avoid overwhelming output
                        top_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
                        analysis["categorical_breakdown"][field] = {
                            "unique_values": len(counts),
                            "top_values": top_items,
                            "total_records": sum(counts.values())
                        }
            
            # Add contextual insights based on query type
            if any(term in original_query.lower() for term in ['sales', 'revenue', 'amount', 'total']):
                analysis["query_context"] = "sales_analysis"
                analysis["recommended_focus"] = ["Look for total sales amounts", "Identify top performing items/regions", "Analyze date ranges and trends"]
            elif any(term in original_query.lower() for term in ['customer', 'client', 'user']):
                analysis["query_context"] = "customer_analysis"
                analysis["recommended_focus"] = ["Customer distribution", "Customer activity patterns", "Regional customer breakdown"]
            elif any(term in original_query.lower() for term in ['product', 'item', 'inventory']):
                analysis["query_context"] = "product_analysis"
                analysis["recommended_focus"] = ["Product performance", "Product categories", "Inventory levels"]
            else:
                analysis["query_context"] = "general_analysis"
                analysis["recommended_focus"] = ["Data distribution patterns", "Key metrics and totals", "Date ranges and trends"]
            
            return analysis
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "fallback_info": f"Dataset contains {len(all_data)} records"
            }

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
            analysis = self.llm_client.invoke_with_json_response(prompt, timeout=300)
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
        
        from app.llm_client import llm_client
        column_reference = ""
        tables_data = agent_details.get('tables', [])
        if tables_data:
            column_reference = llm_client._extract_column_reference_from_structured_data(tables_data)
        
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
            request_details = self.llm_client.invoke_with_json_response(prompt, timeout=600)
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
        from app.llm_client import llm_client
        column_reference = ""
        tables_data = agent_details.get('tables', [])
        if tables_data:
            column_reference = llm_client._extract_column_reference_from_structured_data(tables_data)
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
            request_details = self.llm_client.invoke_with_json_response(prompt, timeout=600)
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
        
        # Retrieve the stored workflow context
        context = self.workflow_contexts.get(workflow_id)
        if not context:
            print(f"[EnhancedOrchestrator] Warning: No stored context found for workflow {workflow_id}")
            return {
                "status": "error",
                "message": f"Workflow {workflow_id} not found or expired. Please start a new query."
            }
        
        choice_type = user_choice.get("type")
        
        if choice_type == "agent_choice":
            chosen_agent_id = user_choice.get("agent_id")
            user_message = user_choice.get("message", "")
            
            print(f"[EnhancedOrchestrator] User chose agent: {chosen_agent_id}, message: '{user_message}'")
            
            # Get the chosen agent details
            available_agents = self.registry_manager.list_available_agents()
            
            # Find agent by ID (appKey or id field)
            chosen_agent = None
            for agent in available_agents:
                if agent.get("appKey") == chosen_agent_id or agent.get("id") == chosen_agent_id:
                    chosen_agent = agent
                    break
            
            if not chosen_agent:
                print(f"[EnhancedOrchestrator] Error: Agent {chosen_agent_id} not found in available agents")
                return {
                    "status": "error",
                    "message": f"Selected agent {chosen_agent_id} not found"
                }
            
            print(f"[EnhancedOrchestrator] Found chosen agent: {chosen_agent.get('name')} (type: {chosen_agent.get('agent_type', 'data_agent')})")
            
            # Continue the workflow with the chosen agent
            try:
                # Step 2: Select the chosen agent (skip LLM recommendation since user already chose)
                workflow_streamer.emit_step_started(
                    context.workflow_id, context.session_id, f"step_{context.current_step}_select",
                    "agent_selection", f"🎯 Using user-selected agent: {chosen_agent['name']}..."
                )
                
                # Step 3: Prepare agent request
                agent_request = await self._prepare_agent_request(context, chosen_agent)
                if not agent_request:
                    return {
                        "status": "error",
                        "message": f"Failed to prepare request for {chosen_agent['name']}"
                    }
                
                # Step 4: Execute the agent
                execution_result = await self._execute_agent(context, chosen_agent, agent_request)
                context.add_result(execution_result)
                
                # Update the stored context
                self.workflow_contexts[workflow_id] = context
                
                # Step 5: Generate final result
                final_result = await self._generate_final_result(context)
                
                # Clean up the workflow context after completion
                if workflow_id in self.workflow_contexts:
                    del self.workflow_contexts[workflow_id]
                
                return {
                    "status": "completed",
                    "final_answer": final_result["final_answer"],
                    "execution_summary": final_result.get("execution_summary", {}),
                    "workflow_id": workflow_id,
                    "session_id": session_id
                }
                
            except Exception as e:
                print(f"[EnhancedOrchestrator] Error continuing workflow: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error continuing workflow: {str(e)}"
                }
                
        elif choice_type == "user_input":
            user_input = user_choice.get("input", "")
            print(f"[EnhancedOrchestrator] User provided input: '{user_input}'")
            
            # Add user input to context and continue workflow
            context.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })
            
            # Update the stored context
            self.workflow_contexts[workflow_id] = context
            
            # Continue the workflow loop from where it left off
            try:
                # Continue the main workflow loop
                while context.current_step < context.max_steps:
                    # Get next agent recommendation with updated context
                    agent_recommendation = await self._get_agent_recommendation(context)
                    
                    if agent_recommendation.get("action") == "complete":
                        break
                    elif agent_recommendation.get("action") == "require_user_choice":
                        await self._request_user_agent_choice(context, agent_recommendation)
                        return {
                            "status": "user_choice_required",
                            "workflow_id": context.workflow_id,
                            "session_id": context.session_id,
                            "choice_options": agent_recommendation.get("user_choice_options", [])
                        }
                    elif agent_recommendation.get("action") == "require_user_input":
                        await self._request_user_input(context, agent_recommendation)
                        return {
                            "status": "user_input_required",
                            "workflow_id": context.workflow_id,
                            "session_id": context.session_id
                        }
                    else:
                        # Continue with agent execution
                        selected_agent = await self._select_agent(context, agent_recommendation)
                        if selected_agent:
                            agent_request = await self._prepare_agent_request(context, selected_agent)
                            execution_result = await self._execute_agent(context, selected_agent, agent_request)
                            context.add_result(execution_result)
                            
                            # Update stored context
                            self.workflow_contexts[workflow_id] = context
                            
                            # Decide next step
                            decision = await self._evaluate_and_decide_next_step(context)
                            if decision.get("action") == "complete":
                                break
                
                # Generate final result
                final_result = await self._generate_final_result(context)
                
                # Clean up workflow context
                if workflow_id in self.workflow_contexts:
                    del self.workflow_contexts[workflow_id]
                
                return {
                    "status": "completed",
                    "final_answer": final_result["final_answer"],
                    "execution_summary": final_result.get("execution_summary", {}),
                    "workflow_id": workflow_id,
                    "session_id": session_id
                }
                
            except Exception as e:
                print(f"[EnhancedOrchestrator] Error continuing workflow with user input: {str(e)}")
                return {
                    "status": "error", 
                    "message": f"Error continuing workflow: {str(e)}"
                }
        
        return {
            "status": "error", 
            "message": "Invalid user choice type"
        }

# Global instance
enhanced_orchestrator = EnhancedOrchestrator()
