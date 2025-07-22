"""
Enhanced workflow management using LangGraph for better state management and flow control.
This is a proposed replacement for the current manual state management approach.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from datetime import datetime
from decimal import Decimal
import uuid
import logging
import time
import json

logger = logging.getLogger(__name__)

class WorkflowState(TypedDict):
    """Dynamic workflow state schema"""
    # Core workflow data
    workflow_id: str
    session_id: str
    user_query: str
    
    # Dynamic execution plan
    execution_plan: Optional[List[Dict[str, Any]]]  # LLM-generated plan
    current_step: int
    plan_reasoning: Optional[str]
    
    # Available agents/tools context
    available_agents: Optional[List[Dict[str, Any]]]  # Data agents + applications
    agent_selection_history: List[Dict[str, Any]]  # Track what agents were used
    
    # Current execution
    current_agent: Optional[str]
    current_agent_name: Optional[str] 
    current_action: Optional[str]  # "select_agent", "execute_query", "call_api", "ask_user", "finalize"
    current_results: Optional[Dict[str, Any]]
    execution_error: Optional[str]
    
    # LLM reasoning and decisions
    llm_reasoning: Optional[str]
    next_action_reasoning: Optional[str]
    
    # User interaction
    user_input_needed: bool
    user_question: Optional[str] 
    user_response: Optional[str]
    pending_user_context: Optional[str]  # What LLM needs from user
    
    # Accumulated data from all steps
    collected_data: Dict[str, Any]  # All data from various agents
    intermediate_results: List[Dict[str, Any]]  # Step-by-step results
    
    # Final output
    final_answer: Optional[str]
    workflow_status: str  # "planning", "executing", "waiting_input", "completed", "stopped", "error"
    
    # SQL query generation state
    generated_sql_query: Optional[str]  # Generated SQL query
    sql_response_metadata: Optional[Dict[str, Any]]  # LLM response metadata
    target_agent_details: Optional[Dict[str, Any]]  # Full agent details for current operation
    
    # Control flags
    user_stop_requested: bool
    max_steps: int
    
    # Messages for streaming
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Metadata and timing
    created_at: datetime
    updated_at: datetime
    step_count: int
    execution_time_ms: int

def convert_decimals_to_float(obj: Any) -> Any:
    """Convert Decimal objects to floats for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_decimals_to_float(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals_to_float(item) for item in obj]
    else:
        return obj

class WorkflowOrchestrator:
    """LangGraph-based workflow orchestrator for better state management"""
    
    def __init__(self):
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the dynamic workflow graph"""
        
        # Create the graph
        workflow = StateGraph(WorkflowState)
        
        # Add dynamic nodes
        workflow.add_node("initialize", self._initialize_workflow)
        workflow.add_node("create_plan", self._create_dynamic_plan) 
        workflow.add_node("execute_step", self._execute_current_step)
        workflow.add_node("evaluate_progress", self._evaluate_and_decide_next)
        workflow.add_node("handle_user_input", self._handle_user_input)
        workflow.add_node("finalize_workflow", self._finalize_workflow)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Build dynamic flow
        workflow.add_edge("initialize", "create_plan")
        workflow.add_edge("create_plan", "execute_step")
        workflow.add_edge("execute_step", "evaluate_progress")
        
        # Add conditional routing based on LLM decisions
        workflow.add_conditional_edges(
            "evaluate_progress",
            self._route_next_action,
            {
                "continue": "execute_step",  # Continue with next step
                "ask_user": "handle_user_input",  # Need user input
                "finalize": "finalize_workflow",  # Complete workflow
                "stop": END,  # User requested stop
                "error": END  # Error occurred
            }
        )
        
        workflow.add_edge("handle_user_input", "execute_step")  # Resume after user input
        workflow.add_edge("finalize_workflow", END)
        
        # Add recursion limit and checkpointer to prevent infinite loops
        return workflow.compile(
            checkpointer=None,  # No persistence needed for this use case
            interrupt_before=[],  # No interruption points
            debug=False
        )
    
    def _initialize_workflow(self, state: WorkflowState) -> WorkflowState:
        """Initialize dynamic workflow"""
        import time
        
        state["workflow_id"] = str(uuid.uuid4())
        state["created_at"] = datetime.utcnow()
        state["updated_at"] = datetime.utcnow()
        state["step_count"] = 0
        state["current_step"] = 0
        state["user_input_needed"] = False
        state["user_stop_requested"] = False
        state["max_steps"] = 10  # Prevent infinite loops
        state["workflow_status"] = "planning"
        state["collected_data"] = {}
        state["intermediate_results"] = []
        state["agent_selection_history"] = []
        state["execution_time_ms"] = int(time.time() * 1000)
        
        self._emit_streaming_event(state, "workflow_started", 
                                 f"🚀 Starting dynamic workflow for: {state['user_query']}")
        
        return state
    
    def _create_dynamic_plan(self, state: WorkflowState) -> WorkflowState:
        """LLM creates dynamic execution plan"""
        from app.llm_client import llm_client
        from app.registry import fetch_agents_and_tools_from_registry
        
        self._emit_streaming_event(state, "thinking", "🧠 Analyzing query and creating execution plan...")
        
        try:
            # Check for simple greetings first
            query_lower = state['user_query'].lower().strip()
            simple_greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you"]
            
            if query_lower in simple_greetings:
                # Handle simple greeting without complex planning
                greeting_plan = [{
                    "step": 1,
                    "action": "finalize",
                    "description": "Respond to greeting and explain capabilities",
                    "expected_output": "Friendly greeting response with system capabilities"
                }]
                
                state["execution_plan"] = greeting_plan
                state["plan_reasoning"] = "Simple greeting detected - responding with capabilities overview"
                
                self._emit_streaming_event(state, "plan_created", 
                                         "👋 Greeting detected - preparing welcome response")
                
                state["workflow_status"] = "executing"
                return state
            
            # Get available agents and tools
            registry_data = fetch_agents_and_tools_from_registry()
            
            # Simplify agent data for LLM
            available_agents = []
            
            # Process all agents from registry (they're all at the top level)
            for app_id, app in registry_data.items():
                tools = [tool.get("name", "") for tool in app.get("tools", [])]
                
                # Use the actual agent_type from registry data, fallback to application
                app_type = app.get("agent_type", "application")
                
                # For data agents, include database type information
                database_info = ""
                if app_type == "data_agent":
                    database_type = app.get("database_type", "unknown")
                    database_info = f" ({database_type} database)"
                
                available_agents.append({
                    "id": app_id,  # This is the agent key that should be used for targeting
                    "name": app.get("name", "Unknown"),
                    "type": app_type,
                    "database_type": app.get("database_type", None),
                    "description": app.get("description", "") + database_info,
                    "capabilities": tools if tools else ["API calls", "data processing"]
                })
            
            state["available_agents"] = available_agents
            
            # Create planning prompt
            planning_prompt = f"""
            You are a workflow orchestrator. Analyze the user query and create a dynamic execution plan.
            
            USER QUERY: {state['user_query']}
            
            AVAILABLE AGENTS:
            {self._format_agents_for_llm(available_agents)}
            
            Create a step-by-step execution plan. Each step should specify:
            1. Action type: ONLY use: "select_agent", "form_query", "execute_query", "analyze_data", or "finalize"
            2. Target agent ID (use the ID field from the agent list above, not the name)
            3. What data/input is needed
            4. Expected output
            
            CRITICAL REQUIREMENTS:
            - For ANY data-related query (sales, reports, analytics, etc.), you MUST include ALL 4 steps
            - Never skip steps 2 and 3 for data queries - they are essential for data retrieval
            - When specifying "target_agent", always use the agent ID (e.g., "cmdaqhpys0003n5o0583646e7"), not the agent name
            
            MANDATORY SEQUENCE FOR DATA QUERIES:
            1. "select_agent" - Choose the appropriate data agent
            2. "form_query" - Generate SQL using LLM + agent schema details  
            3. "execute_query" - Execute the SQL against the database (THIS IS MANDATORY)
            4. "finalize" - Analyze results and provide insights
            
            Do NOT create shorter plans for data queries. The user needs actual data, not just explanations.
            
            Return JSON:
            {{
                "plan_reasoning": "Why this plan will solve the user's query",
                "execution_plan": [
                    {{
                        "step": 1,
                        "action": "select_agent",
                        "target_agent": "agent_id_here",
                        "description": "Select appropriate data agent",
                        "expected_output": "Agent selected for query"
                    }},
                    {{
                        "step": 2,
                        "action": "form_query",
                        "target_agent": "agent_id_here",
                        "description": "Generate SQL query using agent schema",
                        "expected_output": "SQL query generated"
                    }},
                    {{
                        "step": 3,
                        "action": "execute_query",
                        "target_agent": "agent_id_here",
                        "description": "Execute SQL query against database",
                        "expected_output": "Query results with actual data"
                    }},
                    {{
                        "step": 4,
                        "action": "finalize",
                        "description": "Generate final answer with insights from retrieved data",
                        "expected_output": "Complete response with data analysis"
                    }}
                ],
                "estimated_complexity": "low|medium|high",
                "may_need_user_input": true/false
            }}
            """
            
            # Emit progress event before LLM call to keep streaming alive
            self._emit_streaming_event(state, "llm_planning", "🤖 Waiting for AI to create execution plan...")
            
            plan_result = llm_client.invoke_with_json_response(planning_prompt)
            
            state["execution_plan"] = plan_result.get("execution_plan", [])
            state["plan_reasoning"] = plan_result.get("plan_reasoning", "")
            
            # Debug: Log the actual plan that was generated
            plan_actions = [step.get("action", "unknown") for step in state["execution_plan"]]
            self._emit_streaming_event(state, "plan_debug", 
                                     f"🔍 Generated plan actions: {plan_actions} (Total: {len(state['execution_plan'])} steps)")
            
            # Debug: Log full plan details for troubleshooting
            for i, step in enumerate(state["execution_plan"]):
                self._emit_streaming_event(state, "plan_step_detail", 
                                         f"📋 Step {i+1}: action='{step.get('action')}', target_agent='{step.get('target_agent', 'none')}', description='{step.get('description', 'none')}'")
            
            self._emit_streaming_event(state, "plan_created", 
                                     f"📋 Created plan with {len(state['execution_plan'])} steps: {state['plan_reasoning']}")
            
            # Show detailed plan to user
            plan_summary = "\n".join([
                f"Step {step['step']}: {step['action']} - {step['description']}" 
                for step in state['execution_plan']
            ])
            
            self._emit_streaming_event(state, "plan_details", 
                                     f"📝 Detailed Execution Plan:\n{plan_summary}")
            
            # Validate plan for data queries
            if any(word in state['user_query'].lower() for word in ['sales', 'data', 'report', 'analytics', 'query', 'show', 'find', 'get']):
                required_actions = ["select_agent", "form_query", "execute_query", "finalize"]
                if not all(action in plan_actions for action in required_actions):
                    self._emit_streaming_event(state, "plan_warning", 
                                             f"⚠️ Data query detected but plan missing required steps. Expected: {required_actions}, Got: {plan_actions}")
            
            state["workflow_status"] = "executing"
            
        except Exception as e:
            state["execution_error"] = str(e)
            state["workflow_status"] = "error"
            self._emit_streaming_event(state, "error", f"❌ Planning failed: {str(e)}")
        
        return state
    
    def _execute_current_step(self, state: WorkflowState) -> WorkflowState:
        """Execute the current step in the plan"""
        
        if state["user_stop_requested"]:
            state["workflow_status"] = "stopped"
            return state
            
        if state["current_step"] >= len(state.get("execution_plan", [])):
            state["workflow_status"] = "completed"
            return state
            
        if state["step_count"] >= state["max_steps"]:
            self._emit_streaming_event(state, "warning", 
                                     f"⚠️ Maximum steps ({state['max_steps']}) reached. Stopping to prevent infinite loop.")
            state["workflow_status"] = "completed"
            return state
        
        current_step_data = state["execution_plan"][state["current_step"]]
        action_type = current_step_data.get("action", "unknown")
        
        self._emit_streaming_event(state, "step_started", 
                                 f"▶️ Step {state['current_step'] + 1}: {current_step_data.get('description', 'Unknown step')}")
        
        # Debug: Log step execution details
        self._emit_streaming_event(state, "step_debug", 
                                 f"🔍 Executing step {state['current_step'] + 1}/{len(state['execution_plan'])}: action='{action_type}', target_agent='{current_step_data.get('target_agent', 'none')}'")
        
        state["current_action"] = action_type
        
        try:
            if action_type == "select_agent":
                self._execute_agent_selection(state, current_step_data)
            elif action_type == "form_query":
                self._execute_form_query(state, current_step_data)
            elif action_type == "execute_query":
                self._execute_data_query(state, current_step_data)
            elif action_type == "call_api":
                self._execute_api_call(state, current_step_data)
            elif action_type == "analyze_data":
                self._execute_data_analysis(state, current_step_data)
            elif action_type == "finalize":
                # For finalize actions, mark as ready to finalize
                state["workflow_status"] = "ready_to_finalize"
                self._emit_streaming_event(state, "ready_to_finalize", "✅ Ready to generate final response")
                # Don't return early - let the step be incremented normally
            else:
                # Unknown action - try to infer what was intended
                if "agent" in action_type.lower() or "select" in action_type.lower():
                    self._execute_agent_selection(state, current_step_data)
                elif "form" in action_type.lower() and "query" in action_type.lower():
                    self._execute_form_query(state, current_step_data)
                elif "query" in action_type.lower() or "execute" in action_type.lower() or "sql" in action_type.lower():
                    self._execute_data_query(state, current_step_data)
                else:
                    self._emit_streaming_event(state, "warning", f"⚠️ Unknown action '{action_type}', skipping step")
                
        except Exception as e:
            state["execution_error"] = str(e)
            self._emit_streaming_event(state, "error", f"❌ Step execution failed: {str(e)}")
        
        # Debug: Log step completion
        self._emit_streaming_event(state, "step_completed", 
                                 f"✅ Completed step {state['current_step'] + 1}: {action_type}")
        
        state["step_count"] += 1
        state["current_step"] += 1
        state["updated_at"] = datetime.utcnow()
        
        # Debug: Log next step info
        if state["current_step"] < len(state.get("execution_plan", [])):
            next_step = state["execution_plan"][state["current_step"]]
            self._emit_streaming_event(state, "next_step_info", 
                                     f"➡️ Next step {state['current_step'] + 1}: {next_step.get('action', 'unknown')}")
        else:
            self._emit_streaming_event(state, "plan_complete", 
                                     "🏁 All planned steps completed, moving to evaluation")
        
        return state
    
    def _execute_agent_selection(self, state: WorkflowState, step_data: Dict[str, Any]):
        """Execute agent selection step"""
        from app.unified_router import unified_router
        
        target_agent = step_data.get("target_agent")
        if target_agent:
            # Direct agent selection using agent ID or name
            agent_info = None
            
            # First try to find by ID
            agent_info = next((a for a in state["available_agents"] if a["id"] == target_agent), None)
            
            # If not found by ID, try by name (fallback for older plans)
            if not agent_info:
                agent_info = next((a for a in state["available_agents"] if a["name"] == target_agent), None)
            
            if agent_info:
                state["current_agent"] = agent_info["id"]  # Always use ID internally
                state["current_agent_name"] = agent_info["name"]
                
                self._emit_streaming_event(state, "agent_selected", 
                                         f"🎯 Selected {agent_info['type']}: {agent_info['name']} (ID: {agent_info['id']})")
            else:
                # Agent not found - this is an error
                self._emit_streaming_event(state, "error", 
                                         f"❌ Agent '{target_agent}' not found in available agents")
                state["execution_error"] = f"Agent '{target_agent}' not found"
                return
        else:
            # Use router to select best agent
            result = unified_router.route_query(state["user_query"], state["session_id"], enable_orchestration=False)
            
            state["current_agent"] = result.get("selected_agent")
            state["current_agent_name"] = result.get("agent_name", result.get("selected_agent"))
            
            self._emit_streaming_event(state, "agent_selected", 
                                     f"🎯 Auto-selected: {state['current_agent_name']} (confidence: {result.get('confidence', 0)}%)")
        
        # Track selection
        state["agent_selection_history"].append({
            "step": state["current_step"],
            "agent_id": state["current_agent"],
            "agent_name": state["current_agent_name"],
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def _execute_form_query(self, state: WorkflowState, step_data: Dict[str, Any]):
        """Execute form query step - use LLM + full agent details to generate SQL"""
        from app.registry import get_enhanced_agent_details_for_llm
        from app.llm_client import llm_client
        
        if not state.get("current_agent"):
            self._emit_streaming_event(state, "error", "❌ No agent selected for query formation")
            state["execution_error"] = "No agent selected for query formation"
            return
            
        self._emit_streaming_event(state, "forming_query", 
                                 f"🧠 Generating SQL query using {state['current_agent_name']} schema...")
        
        try:
            # Get enhanced agent details from cache
            self._emit_streaming_event(state, "loading_agent_details", 
                                     f"📋 Loading enhanced details for {state['current_agent_name']}...")
            
            agent_details = get_enhanced_agent_details_for_llm(state["current_agent"])
            
            if not agent_details:
                self._emit_streaming_event(state, "error", 
                                         f"❌ Enhanced agent details not found for {state['current_agent']}")
                state["execution_error"] = f"Enhanced agent details not found for {state['current_agent']}"
                return
            
            self._emit_streaming_event(state, "agent_details_loaded", 
                                     f"✅ Loaded details: {len(agent_details.get('tables', []))} tables, {len(agent_details.get('table_relations', []))} relations")
                
            # Prepare formatted data for the prompt to avoid complex f-string expressions
            self._emit_streaming_event(state, "preparing_prompt", 
                                     "🔧 Preparing comprehensive schema information for LLM...")
            
            database_type = agent_details.get('database_type', 'unknown')
            agent_name = agent_details.get('name', 'Unknown')
            description = agent_details.get('description', '')[:200] + "..." if len(agent_details.get('description', '')) > 200 else agent_details.get('description', '')
            
            # Format tables list
            tables_info = []
            for t in agent_details.get('tables', [])[:10]:  # Limit to 10 tables
                table_name = t.get('name', 'unknown')
                row_count = t.get('row_count', 0)
                table_desc = t.get('description', 'No description')[:80] + "..." if len(t.get('description', '')) > 80 else t.get('description', 'No description')
                tables_info.append(f"• {table_name} ({row_count} rows) - {table_desc}")
            
            # Format relationships
            relations_info = []
            for r in agent_details.get('table_relations', [])[:5]:  # Limit to 5 relations
                rel_type = r.get('relationshipType', 'unknown')
                source_col = r.get('sourceColumn', '')
                target_col = r.get('targetColumn', '')
                rel_desc = r.get('description', '')[:100] + "..." if len(r.get('description', '')) > 100 else r.get('description', '')
                relations_info.append(f"• {rel_type}: {source_col} -> {target_col} | {rel_desc}")
            
            # Format sample queries
            sample_queries_text = "\n".join(agent_details.get('sample_queries', [])[:2])  # Limit to 2 samples
            
            # Format capabilities
            capabilities_text = "\n".join([f"• {cap}" for cap in agent_details.get('capabilities', [])])
            
            # Format tools
            tools_info = []
            for tool in agent_details.get('tools', []):
                tool_name = tool.get('name', '')
                tool_desc = tool.get('description', '')
                if tool_name and tool_desc:
                    tools_info.append(f"• {tool_name}: {tool_desc}")
            
            # Extract structured tables data for column validation
            tables_data = agent_details.get('tables', [])
            
            # Generate column reference using our enhanced method
            from app.llm_client import llm_client
            column_reference = ""
            if tables_data:
                try:
                    column_reference = llm_client._extract_column_reference_from_structured_data(tables_data)
                    print(f"[DEBUG] LangGraphOrchestrator: Generated column reference for {len(tables_data)} tables")
                except Exception as e:
                    print(f"[DEBUG] LangGraphOrchestrator: Error generating column reference: {e}")
                    column_reference = "⚠️  Use ONLY exact column names from the schema below"
            
            # Send enhanced agent details + user query to LLM for SQL generation
            sql_prompt = f"""You are an expert SQL developer. Generate a structured response for the user's SQL query request.

⚠️ CRITICAL: You MUST use ONLY the exact column names from the schema below. DO NOT assume or invent any column names.

USER QUERY: {state['user_query']}

DATABASE DETAILS:
- Database Type: {database_type}
- Agent Name: {agent_name}
- Description: {description}
- Total Tables: {agent_details.get('metadata', {}).get('total_tables', 'unknown')}
- Total Relations: {agent_details.get('metadata', {}).get('total_relations', 'unknown')}

EXACT COLUMN VALIDATION:
{column_reference}

DATABASE SCHEMA:
{agent_details.get('schema', 'No detailed schema available')[:1500] + '...[truncated]' if len(agent_details.get('schema', '')) > 1500 else agent_details.get('schema', 'No detailed schema available')}

AVAILABLE TABLES:
{chr(10).join(tables_info)}

TABLE RELATIONSHIPS:
{chr(10).join(relations_info)}

CAPABILITIES:
{capabilities_text}

SAMPLE QUERIES AND EXAMPLES:
{sample_queries_text}

AVAILABLE TOOLS:
{chr(10).join(tools_info)}

MANDATORY VALIDATION PROCESS:
1. Identify which tables you need for the query
2. For each table, find the exact column names in the schema above  
3. Verify that every column in your SQL query exists in the schema
4. Do NOT use assumed column names like "product_name", "customer_name", etc.
5. Use ONLY the exact column names shown in the schema

Generate a comprehensive SQL response that answers the user's question. Consider:
1. Use appropriate table and column names from the detailed schema above
2. Leverage the table relationships for proper JOINs when needed
3. Apply proper filters and conditions based on business context
4. Use aggregations if needed for the query requirements
5. Follow best practices for {database_type} database
6. Consider row counts for performance optimization
7. Use sample queries as examples of proven patterns
8. Explain your reasoning and approach clearly

Return JSON with the following structure:
{{
    "column_validation": "Step 1: Tables needed: [list]. Step 2: Exact columns from schema: [list each column with table.column format]. Step 3: Verified all columns exist in schema.",
    "sql_query": "SELECT statement using ONLY exact column names from schema above",
    "query_explanation": "Explanation of what the query does and why",
    "reasoning": "Your thought process and approach",
    "confidence": 0.85,
    "assumptions": ["assumption1", "assumption2"],
    "limitations": ["limitation1", "limitation2"],
    "schema_used": ["table1", "table2"],
    "estimated_complexity": "low|medium|high",
    "alternative_approaches": ["approach1", "approach2"],
    "database_context": "{description[:100]}",
    "connection_type": "{database_type}"
}}

CRITICAL VALIDATION CHECKLIST:
- ✅ All column names are from the schema above
- ✅ No assumed or invented column names used
- ✅ Table names match the schema exactly
- ✅ Column names match the schema exactly

Respond with ONLY valid JSON - validate every column against the schema!"""
            
            # Generate structured SQL response using LLM
            self._emit_streaming_event(state, "generating_sql", 
                                     "🧠 LLM analyzing schema and generating structured SQL response...")
            
            # Debug logging for the full prompt
            print(f"[DEBUG] LangGraphOrchestrator: Sending SQL generation prompt to LLM:")
            print("=" * 80)
            print(sql_prompt[:2000] + "..." if len(sql_prompt) > 2000 else sql_prompt)
            print("=" * 80)
            print(f"[DEBUG] LangGraphOrchestrator: Total prompt length: {len(sql_prompt)} chars")
            
            try:
                # Call LLM with timeout to prevent UI timeouts
                sql_response = llm_client.invoke_with_json_response(sql_prompt)
                
                if not sql_response:
                    raise Exception("LLM returned empty response")
                    
            except Exception as llm_error:
                # Fallback if LLM fails
                self._emit_streaming_event(state, "llm_fallback", 
                                         f"⚠️ LLM generation failed: {str(llm_error)}, using fallback")
                
                # Create a basic fallback response
                sql_response = {
                    "sql_query": f"-- Query for: {state['user_query']}\nSELECT * FROM {tables_info[0].split(' ')[1] if tables_info else 'table'} LIMIT 10;",
                    "query_explanation": f"Basic query to explore data for: {state['user_query']}",
                    "reasoning": "Fallback query due to LLM timeout or error",
                    "confidence": 0.3,
                    "assumptions": ["Using fallback due to LLM error"],
                    "limitations": ["This is a basic fallback query"],
                    "schema_used": ["unknown"],
                    "estimated_complexity": "low"
                }
            
            # Extract the SQL query and metadata
            sql_query = sql_response.get("sql_query", "").strip()
            column_validation = sql_response.get("column_validation", "")
            query_explanation = sql_response.get("query_explanation", "")
            confidence = sql_response.get("confidence", 0.5)
            
            # Debug log the column validation
            if column_validation:
                print(f"[DEBUG] LangGraphOrchestrator: LLM Column Validation:")
                print("-" * 60)
                print(column_validation)
                print("-" * 60)
            
            # Debug log the generated SQL
            print(f"[DEBUG] LangGraphOrchestrator: Generated SQL Query:")
            print("-" * 60)
            print(sql_query)
            print("-" * 60)
            
            # Clean up SQL query formatting if needed
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            elif sql_query.startswith("```"):
                sql_query = sql_query.replace("```", "").strip()
            
            # Store the generated SQL query and metadata in state for next step
            state["generated_sql_query"] = sql_query
            state["sql_response_metadata"] = sql_response
            state["target_agent_details"] = agent_details
            
            # Truncate long SQL for display
            sql_display = sql_query
            if len(sql_display) > 150:
                sql_display = sql_display[:150] + "..."
                
            self._emit_streaming_event(state, "sql_generated",
                                     f"📝 Generated SQL ({confidence:.0%} confidence): {sql_display}")
            
            # Emit explanation if available
            if query_explanation:
                self._emit_streaming_event(state, "sql_explanation",
                                         f"💡 Query explanation: {query_explanation}")
            
            # Store results with full metadata
            state["current_results"] = {
                "action": "form_query",
                "agent_id": state["current_agent"],
                "agent_name": state["current_agent_name"],
                "sql_query": sql_query,
                "sql_response": sql_response,
                "agent_details": agent_details,
                "metadata": {
                    "confidence": confidence,
                    "query_explanation": query_explanation,
                    "schema_used": sql_response.get("schema_used", []),
                    "complexity": sql_response.get("estimated_complexity", "unknown")
                }
            }
            
        except Exception as e:
            error_msg = f"Query formation failed: {str(e)}"
            self._emit_streaming_event(state, "error", f"❌ {error_msg}")
            state["execution_error"] = error_msg
            state["current_results"] = {"error": error_msg}
    
    def _execute_data_query(self, state: WorkflowState, step_data: Dict[str, Any]):
        """Execute data query step - run the SQL query generated in form_query step"""
        from app.database_query_executor import DatabaseQueryExecutor
        
        if not state.get("current_agent"):
            self._emit_streaming_event(state, "error", "❌ No agent selected for data query")
            state["execution_error"] = "No agent selected for data query"
            return
        
        # Check if we have a pre-generated SQL query from form_query step
        sql_query = state.get("generated_sql_query")
        agent_details = state.get("target_agent_details")
        
        if not sql_query:
            self._emit_streaming_event(state, "error", "❌ No SQL query found. Run form_query step first.")
            state["execution_error"] = "No SQL query found. Run form_query step first."
            return
            
        self._emit_streaming_event(state, "executing_query", 
                                 f"⚡ Executing generated SQL query with {state['current_agent_name']}...")
        
        # Debug: Log SQL query details
        self._emit_streaming_event(state, "sql_execution_debug", 
                                 f"🔍 SQL Execution - Agent: {state['current_agent']}, Query Length: {len(sql_query)}, Session: {state.get('session_id', 'unknown')}")
        
        try:
            # Execute the SQL query using DatabaseQueryExecutor
            self._emit_streaming_event(state, "database_executor_init", "🔧 Initializing DatabaseQueryExecutor...")
            
            executor = DatabaseQueryExecutor()
            
            self._emit_streaming_event(state, "database_executor_ready", "✅ DatabaseQueryExecutor initialized, preparing query execution...")
            
            # Get vault key and connection type from agent details
            if not agent_details:
                raise Exception("Agent details not available for database connection")
            
            vault_key = agent_details.get("vault_key")
            connection_type = agent_details.get("database_type") or agent_details.get("connectionType", "postgresql")
            
            if not vault_key:
                raise Exception(f"No vault key found for agent {state['current_agent']}")
            
            self._emit_streaming_event(state, "database_connection_info", 
                                     f"🔑 Using vault key: {vault_key}, connection type: {connection_type}")
            
            query_result = executor.execute_query(
                vault_key=vault_key,
                connection_type=connection_type,
                sql_query=sql_query,
                limit=100
            )
            
            self._emit_streaming_event(state, "database_query_response", 
                                     f"📡 Database response received - Type: {type(query_result)}, Content: {str(query_result)[:200]}...")
            
            # Process results
            if query_result and query_result.get("data"):
                row_count = len(query_result["data"])
                database_type = agent_details.get('database_type', 'database') if agent_details else 'database'
                self._emit_streaming_event(state, "data_received", 
                                         f"✅ Retrieved {row_count} rows from {database_type}")
                
                # Show sample of data for preview
                if row_count > 0:
                    sample_data = query_result["data"][:3]  # First 3 rows
                    self._emit_streaming_event(state, "data_preview", 
                                             f"📋 Data preview: {len(sample_data)} sample rows retrieved")
                    
                    # Debug: Show actual sample data
                    self._emit_streaming_event(state, "data_sample", 
                                             f"📊 Sample data: {str(sample_data)[:300]}...")
                                             
            elif query_result and query_result.get("error"):
                self._emit_streaming_event(state, "query_error", 
                                         f"⚠️ Query execution error: {query_result['error']}")
                # Don't return here - still store the error result
            elif query_result is None:
                self._emit_streaming_event(state, "query_null_result", 
                                         "⚠️ Query executor returned None - possible connection issue")
            else:
                self._emit_streaming_event(state, "query_complete", "✅ Query executed successfully")
            
            # Store results ALWAYS (even if empty or error)
            state["current_results"] = {
                "action": "execute_query",
                "agent_id": state["current_agent"],
                "agent_name": state["current_agent_name"],
                "query": state["user_query"],
                "sql_query": sql_query,
                "query_result": query_result,
                "metadata": {
                    "row_count": len(query_result.get("data", [])) if query_result else 0,
                    "database_type": agent_details.get("database_type") if agent_details else None,
                    "execution_timestamp": datetime.utcnow().isoformat(),
                    "has_error": bool(query_result and query_result.get("error")) if query_result else True
                }
            }
            
            # Convert Decimal objects to floats for JSON serialization
            state["current_results"] = convert_decimals_to_float(state["current_results"])
            
            # Store in collected data with agent-specific key ALWAYS
            data_key = f"agent_{state['current_agent']}_step_{state['current_step']}"
            state["collected_data"][data_key] = state["current_results"]
            
            # Debug: Confirm data storage
            self._emit_streaming_event(state, "data_stored", 
                                     f"💾 Data stored with key: {data_key}, collected_data_count: {len(state['collected_data'])}")
            
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            self._emit_streaming_event(state, "error", f"❌ {error_msg}")
            state["execution_error"] = error_msg
            
            # Still store error results in collected data
            state["current_results"] = {"error": error_msg, "action": "execute_query"}
            data_key = f"agent_{state['current_agent']}_step_{state['current_step']}"
            state["collected_data"][data_key] = state["current_results"]
            
            self._emit_streaming_event(state, "error_data_stored", 
                                     f"💾 Error data stored with key: {data_key}")
    
    def _execute_api_call(self, state: WorkflowState, step_data: Dict[str, Any]):
        """Execute API call step (placeholder for future implementation)"""
        self._emit_streaming_event(state, "api_call", 
                                 f"🔗 API call: {step_data.get('description', 'Unknown API call')}")
        
        # TODO: Implement actual API calling logic
        state["current_results"] = {"status": "api_call_placeholder", "message": "API calls not yet implemented"}
    
    def _execute_data_analysis(self, state: WorkflowState, step_data: Dict[str, Any]):
        """Execute data analysis step"""
        from app.llm_client import llm_client
        
        self._emit_streaming_event(state, "analyzing", 
                                 f"📈 Analyzing collected data: {step_data.get('description', 'Data analysis')}")
        
        # Analyze collected data with LLM
        analysis_prompt = f"""
        Analyze the following data for the user query: {state['user_query']}
        
        Collected Data: {state['collected_data']}
        
        Provide insights, patterns, and relevant analysis.
        
        Return JSON:
        {{
            "insights": ["insight1", "insight2"],
            "key_findings": "main findings",
            "recommendations": "what should be done next"
        }}
        """
        
        try:
            analysis_result = llm_client.invoke_with_json_response(analysis_prompt)
            state["current_results"] = analysis_result
            
            # Show key findings
            findings = analysis_result.get("key_findings", "Analysis completed")
            self._emit_streaming_event(state, "analysis_complete", f"📊 {findings}")
            
        except Exception as e:
            self._emit_streaming_event(state, "error", f"❌ Analysis failed: {str(e)}")
    
    def _evaluate_and_decide_next(self, state: WorkflowState) -> WorkflowState:
        """LLM evaluates progress and decides next action"""
        from app.llm_client import llm_client
        
        self._emit_streaming_event(state, "thinking", "🤔 Evaluating progress and planning next steps...")
        
        # Debug: Log current evaluation state
        self._emit_streaming_event(state, "evaluation_debug", 
                                 f"🔍 Evaluation: Step {state['current_step']}/{len(state.get('execution_plan', []))}, Status={state.get('workflow_status')}, Error={state.get('execution_error')}")
        
        # Debug: Log detailed state information
        execution_plan = state.get("execution_plan", [])
        current_step = state.get("current_step", 0)
        collected_data_count = len(state.get("collected_data", {}))
        step_count = state.get("step_count", 0)
        
        self._emit_streaming_event(state, "evaluation_state_debug", 
                                 f"🔍 State: current_step={current_step}, step_count={step_count}, plan_length={len(execution_plan)}, collected_data_count={collected_data_count}")
        
        try:
            # Check if we've completed all planned steps first
            at_end_of_plan = current_step >= len(execution_plan)
            
            # Debug: Log evaluation criteria
            self._emit_streaming_event(state, "evaluation_criteria", 
                                     f"🔍 Criteria: at_end_of_plan={at_end_of_plan}, current_step={current_step}/{len(execution_plan)}")
            
            # PRIORITY 1: If we've completed all planned steps, finalize
            if at_end_of_plan:
                # We have reached the end of the plan - time to finalize
                state["workflow_status"] = "ready_to_finalize"
                self._emit_streaming_event(state, "ready_to_finalize", "✅ All planned steps completed - ready to generate final response")
                return state
            
            # PRIORITY 2: Only finalize early if we have ACTUAL DATA from execute_query action
            has_actual_data = False
            if state.get("collected_data"):
                for data_key, data_value in state["collected_data"].items():
                    if isinstance(data_value, dict) and data_value.get("action") == "execute_query":
                        # Only count as meaningful data if it's from actual SQL execution
                        if data_value.get("query_result") and data_value["query_result"].get("data"):
                            has_actual_data = True
                            break
            
            self._emit_streaming_event(state, "data_evaluation", f"🔍 has_actual_data={has_actual_data}")
            
            if has_actual_data:
                # We have actual database query results - can finalize early
                state["workflow_status"] = "ready_to_finalize"
                self._emit_streaming_event(state, "ready_to_finalize", "✅ Database query results available - ready to generate final response")
                return state
            
            # PRIORITY 3: If we haven't completed all planned steps, continue with the plan
            if not at_end_of_plan:
                # Check if there's an execution error that prevents continuation
                if state.get("execution_error"):
                    self._emit_streaming_event(state, "execution_error_detected", 
                                             f"⚠️ Execution error detected: {state['execution_error']}")
                    # Even with errors, try to continue unless it's a critical failure
                    if "not found" in str(state["execution_error"]).lower():
                        state["workflow_status"] = "error"
                        return state
                
                # Continue with next planned step
                if current_step < len(execution_plan):
                    next_step = execution_plan[current_step]
                    self._emit_streaming_event(state, "continuing_plan", 
                                             f"➡️ Continuing with planned step {current_step + 1}: {next_step.get('action', 'unknown')} - {next_step.get('description', 'no description')}")
                else:
                    self._emit_streaming_event(state, "plan_bounds_error", 
                                             f"⚠️ current_step={current_step} >= plan_length={len(execution_plan)} but at_end_of_plan={at_end_of_plan}")
                
                # Debug: Verify we're actually continuing
                self._emit_streaming_event(state, "continue_decision", 
                                         f"🔄 DECISION: Continue execution (not at end of plan)")
                
                state["workflow_status"] = "executing"
                return state
            
            # This should never be reached due to the at_end_of_plan check above
            self._emit_streaming_event(state, "evaluation_fallthrough", 
                                     f"⚠️ Unexpected fallthrough in evaluation logic")
            
            state["workflow_status"] = "ready_to_finalize"
            return state
                
        except Exception as e:
            state["execution_error"] = str(e)
            state["workflow_status"] = "error"
            self._emit_streaming_event(state, "error", f"❌ Evaluation failed: {str(e)}")
        
        return state
    
    def _handle_user_input(self, state: WorkflowState) -> WorkflowState:
        """Handle user input and resume workflow"""
        self._emit_streaming_event(state, "user_input_received",
                                 f"✅ User responded: {state.get('user_response', 'No response')}")
        
        state["user_input_needed"] = False
        state["workflow_status"] = "executing"
        
        # Store user response in context
        state["pending_user_context"] = state.get("user_response", "")
        
        return state
    
    def _finalize_workflow(self, state: WorkflowState) -> WorkflowState:
        """Generate final comprehensive answer"""
        from app.llm_client import llm_client
        import time
        
        self._emit_streaming_event(state, "finalizing", "✨ Generating comprehensive final answer...")
        
        # Add debug logging
        logger.info("Starting workflow finalization")
        logger.info(f"Current state keys: {list(state.keys())}")
        logger.info(f"Collected data keys: {list(state.get('collected_data', {}).keys())}")
        logger.info(f"Collected data content: {state.get('collected_data', {})}")
        
        try:
            # Calculate execution time
            current_time = int(time.time() * 1000)
            state["execution_time_ms"] = current_time - state["execution_time_ms"]
            
            # Check if this is a simple greeting response
            query_lower = state['user_query'].lower().strip()
            simple_greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you"]
            
            if query_lower in simple_greetings:
                # Generate greeting response
                state["final_answer"] = f"""Hello! 👋 

I'm your AI Workflow Assistant powered by LangGraph multi-agent orchestration. I can help you with:

**📊 Data Analysis & Queries**
- SQL database queries and analysis
- Customer analytics and revenue breakdowns  
- Performance comparisons and forecasting
- Financial reporting and insights

**🔄 Multi-Step Workflows**
- Complex business analysis requiring multiple data sources
- Automated report generation
- Cross-functional data integration

**💡 Example Queries:**
- "Show me sales data and create forecasts for next quarter"
- "Compare Q1 vs Q2 performance across all stores"
- "Find top 10 products by profit margin"
- "Generate monthly financial report"

What would you like to work on today? Feel free to ask me anything about your data or business needs!"""
            else:
                # Check if we have SQL query results to format
                sql_results = None
                sql_query = None
                
                # Look for SQL query execution results in collected data
                logger.info("Searching for SQL results in collected data...")
                for data_key, data_value in state.get('collected_data', {}).items():
                    logger.info(f"Checking data_key: {data_key}, data_value type: {type(data_value)}")
                    if isinstance(data_value, dict):
                        logger.info(f"Data value contents: {data_value}")
                        if data_value.get('action') == 'execute_query' and data_value.get('query_result'):
                            sql_results = data_value.get('query_result')
                            sql_query = data_value.get('sql_query')
                            logger.info(f"Found SQL results! Query: {sql_query}, Results type: {type(sql_results)}")
                            break
                
                if sql_results and sql_query:
                    # Use the specialized data answer prompt for SQL results
                    logger.info("Using SQL results for final answer generation")
                    self._emit_streaming_event(state, "formatting_data_results", 
                                             "📊 Formatting SQL query results with LLM analysis...")
                    
                    data_prompt = llm_client.create_data_answer_prompt(
                        query=state['user_query'],
                        sql_query=sql_query,
                        query_result=sql_results
                    )
                    
                    state["final_answer"] = llm_client.invoke_with_text_response(data_prompt)
                    logger.info("Generated final answer using SQL results")
                    
                else:
                    # Create comprehensive final answer prompt for complex queries
                    logger.info("No SQL results found, using general workflow completion")
                    final_prompt = f"""
                    Generate a comprehensive final answer based on the complete workflow execution:
                    
                    Original Query: {state['user_query']}
                    
                    Execution Summary:
                    - Steps Completed: {state['step_count']}
                    - Agents Used: {[h['agent_name'] for h in state.get('agent_selection_history', [])]}
                    - Execution Time: {state['execution_time_ms']/1000:.1f} seconds
                    
                    Collected Data: {state.get('collected_data', {})}
                    
                    Intermediate Results: {state.get('intermediate_results', [])}
                    
                    User Interactions: {state.get('user_response', 'None')}
                    
                    Provide a clear, comprehensive answer that directly addresses the user's original query.
                    Include relevant data insights, recommendations, and next steps if applicable.
                    """
                    
                    state["final_answer"] = llm_client.invoke_with_text_response(final_prompt)
                    logger.info("Generated final answer using general prompt")
            
            state["workflow_status"] = "completed"
            
            # Convert any Decimal objects to floats for JSON serialization
            state["collected_data"] = convert_decimals_to_float(state.get("collected_data", {}))
            if "current_results" in state and state["current_results"]:
                state["current_results"] = convert_decimals_to_float(state["current_results"])
            
            self._emit_streaming_event(state, "workflow_completed", 
                                     f"🎉 Workflow completed in {state['execution_time_ms']/1000:.1f} seconds!")
            
            self._emit_streaming_event(state, "final_result", state["final_answer"])
            
        except Exception as e:
            state["execution_error"] = str(e)
            
            # Check if we have any data despite the error
            has_data = len(state.get('collected_data', {})) > 0
            
            if has_data:
                # Try to provide partial results if we have any data
                state["final_answer"] = f"""**Partial Results Available:**

I was able to retrieve some data for your query "{state.get('user_query', 'data analysis')}", but encountered an error during final processing.

**Available Data:**
{state.get('collected_data', {})}

**Error Details:** {str(e)}

Would you like me to help interpret the available data or retry the analysis?"""
            else:
                # Provide a more user-friendly error message when no data was retrieved
                state["final_answer"] = f"""**Conclusion:**
Unfortunately, the requested {state.get('user_query', 'data analysis')} could not be generated due to a failure in data retrieval. 

**Possible Issues:**
- SQL query execution failed
- Database connection problems  
- Agent configuration issues
- Data source unavailable

**Technical Details:** {str(e)}

**Next Steps:**
1. Check database connectivity
2. Verify data agent configuration
3. Review query syntax and permissions
4. Contact support if the issue persists

Let me know if you need further assistance!"""
            
            self._emit_streaming_event(state, "error", f"❌ Finalization failed: {str(e)}")
            
            # Convert any Decimal objects to floats even in error case
            state["collected_data"] = convert_decimals_to_float(state.get("collected_data", {}))
            if "current_results" in state and state["current_results"]:
                state["current_results"] = convert_decimals_to_float(state["current_results"])
        
        state["updated_at"] = datetime.utcnow()
        return state
    
    def _route_next_action(self, state: WorkflowState) -> str:
        """Route to next node based on workflow status"""
        # Debug: Log routing decision
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", 10)
        current_step = state.get("current_step", 0)
        execution_plan_length = len(state.get("execution_plan", []))
        workflow_status = state.get("workflow_status", "unknown")
        
        self._emit_streaming_event(state, "routing_decision", 
                                 f"🔀 Routing: step_count={step_count}/{max_steps}, current_step={current_step}/{execution_plan_length}, status={workflow_status}")
        
        if state.get("user_stop_requested"):
            return "stop"
        elif state.get("workflow_status") == "error":
            return "error"
        elif state.get("workflow_status") == "waiting_input":
            return "ask_user"
        elif state.get("workflow_status") == "ready_to_finalize":
            return "finalize"
        elif state["current_step"] >= len(state.get("execution_plan", [])):
            return "finalize"
        elif state.get("step_count", 0) >= state.get("max_steps", 10):
            # Prevent infinite loops by enforcing max steps
            self._emit_streaming_event(state, "max_steps_reached", 
                                     f"⚠️ Maximum steps ({state.get('max_steps', 10)}) reached, finalizing workflow")
            return "finalize"
        else:
            return "continue"
    
    def _format_agents_for_llm(self, agents: List[Dict[str, Any]]) -> str:
        """Format agent list for LLM consumption with IDs and types"""
        formatted = []
        for agent in agents:
            # Show both ID and name clearly for LLM to use correct target_agent
            formatted.append(
                f"- ID: {agent['id']} | Name: {agent['name']} ({agent['type']})\n"
                f"  Description: {agent['description']}\n"
                f"  Capabilities: {', '.join(agent.get('capabilities', []))}"
            )
        return "\n".join(formatted)
    
    def _emit_streaming_event(self, state: WorkflowState, event_type: str, message: str):
        """Emit streaming event"""
        print(f"[DEBUG] LangGraphOrchestrator: Emitting event: {event_type} - {message[:50]}...")
        try:
            from app.workflow_streamer import workflow_streamer, EventType
            from app.workflow_streamer import StreamEvent
            from datetime import datetime
            
            # Use specific streaming event methods for known event types
            if event_type == "routing_decision":
                workflow_streamer.emit_routing_decision(
                    workflow_id=state["workflow_id"],
                    session_id=state["session_id"],
                    route_type=state.get("route_type", "unknown"),
                    selected_agent=state.get("selected_agent_name", "unknown"),
                    confidence=state.get("routing_confidence", 0),
                    reasoning=message
                )
            elif event_type == "sql_generated":
                workflow_streamer.emit_sql_generated(
                    workflow_id=state["workflow_id"],
                    session_id=state["session_id"],
                    sql_query=state.get("sql_query", ""),
                    database_type=state.get("route_type", "unknown")
                )
            else:
                # For other orchestrator events, emit them with their specific event types
                # Create a custom event with the exact event_type
                event = StreamEvent(
                    event_type=event_type,  # Use the exact event type (workflow_started, thinking, etc.)
                    workflow_id=state["workflow_id"],
                    timestamp=datetime.utcnow(),
                    session_id=state["session_id"],
                    step_id=f"step_{state['step_count']}",
                    data={
                        "message": message,
                        "step_count": state.get("step_count", 0),
                        "current_step": state.get("current_step", 0),
                        "workflow_status": state.get("workflow_status", "processing")
                    }
                )
                workflow_streamer.emit_event(event)
        except Exception as e:
            print(f"[WorkflowOrchestrator] Streaming event error: {e}")
    
    async def execute_workflow(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """Execute dynamic workflow with proper state management"""
        
        initial_state = WorkflowState(
            workflow_id="",  # Will be set in initialize
            session_id=session_id,
            user_query=user_query,
            messages=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            step_count=0,
            current_step=0,
            user_input_needed=False,
            user_stop_requested=False,
            max_steps=10,
            workflow_status="initializing",
            execution_plan=None,
            plan_reasoning=None,
            available_agents=None,
            agent_selection_history=[],
            current_agent=None,
            current_agent_name=None,
            current_action=None,
            current_results=None,
            execution_error=None,
            llm_reasoning=None,
            next_action_reasoning=None,
            user_question=None,
            user_response=None,
            pending_user_context=None,
            collected_data={},
            intermediate_results=[],
            final_answer=None,
            generated_sql_query=None,
            sql_response_metadata=None,
            target_agent_details=None,
            execution_time_ms=0
        )
        
        try:
            # Execute the dynamic workflow with recursion limit
            final_state = await self.workflow.ainvoke(
                initial_state,
                config={
                    "recursion_limit": 50,  # Increase from default 25 but still prevent infinite loops
                    "configurable": {}
                }
            )
            
            return {
                "status": final_state.get("workflow_status", "unknown"),
                "workflow_id": final_state["workflow_id"],
                "final_answer": final_state.get("final_answer"),
                "user_input_needed": final_state.get("user_input_needed", False),
                "user_question": final_state.get("user_question"),
                "execution_summary": {
                    "steps_executed": final_state["step_count"],
                    "agents_used": [h["agent_name"] for h in final_state.get("agent_selection_history", [])],
                    "execution_time_ms": final_state.get("execution_time_ms", 0),
                    "data_sources": len(final_state.get("collected_data", {})),
                    "plan_steps": len(final_state.get("execution_plan", []))
                },
                "collected_data": final_state.get("collected_data", {}),
                "intermediate_results": final_state.get("intermediate_results", [])
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "final_answer": f"Dynamic workflow execution failed: {str(e)}"
            }
    
    async def continue_workflow(self, workflow_id: str, user_response: str) -> Dict[str, Any]:
        """Continue workflow after user input"""
        # TODO: Implement workflow resumption with user input
        # This would require persisting workflow state and resuming from checkpoint
        return {
            "status": "not_implemented",
            "message": "Workflow resumption not yet implemented"
        }
    
    async def stop_workflow(self, workflow_id: str, session_id: str) -> Dict[str, Any]:
        """Stop workflow execution"""
        try:
            self._emit_streaming_event(
                {"workflow_id": workflow_id, "session_id": session_id}, 
                "workflow_stopped", 
                "🛑 Workflow stopped by user request"
            )
            
            return {
                "status": "stopped",
                "message": "Workflow execution stopped successfully"
            }
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e)
            }

# Global instance
langgraph_orchestrator = WorkflowOrchestrator()
