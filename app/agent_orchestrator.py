"""
Agent Orchestrator for multi-agent workflows and chaining.
Handles complex queries that require multiple agents and user interaction.
"""
import json
import uuid
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio

from app.llm_client import llm_client
from app.session_manager import session_manager
from app.workflow_streamer import workflow_streamer

class WorkflowState(Enum):
    """Workflow execution states."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepType(Enum):
    """Types of workflow steps."""
    DATA_QUERY = "data_query"
    APPLICATION_CALL = "application_call"
    DATA_TRANSFORMATION = "data_transformation"
    USER_INPUT = "user_input"
    CONDITION = "condition"
    AGGREGATION = "aggregation"

@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    step_type: StepType
    description: str
    agent_id: Optional[str] = None
    query: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    depends_on: Optional[List[str]] = None  # List of step_ids this step depends on
    condition: Optional[str] = None  # Condition for conditional execution
    timeout_seconds: int = 300
    retry_count: int = 3
    
    # Execution results
    status: WorkflowState = WorkflowState.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class UserInputRequest:
    """Request for user input during workflow execution."""
    input_id: str
    workflow_id: str
    step_id: str
    prompt: str
    input_type: str  # "text", "choice", "number", "date", etc.
    choices: Optional[List[str]] = None
    validation_rules: Optional[Dict[str, Any]] = None
    timeout_minutes: int = 30
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class Workflow:
    """Complete workflow definition and execution state."""
    workflow_id: str
    title: str
    description: str
    steps: List[WorkflowStep]
    session_id: str
    user_query: str
    
    # Execution state
    status: WorkflowState = WorkflowState.PENDING
    current_step: Optional[str] = None
    results: Dict[str, Any] = None
    final_answer: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # User interaction
    pending_inputs: List[UserInputRequest] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.results is None:
            self.results = {}
        if self.pending_inputs is None:
            self.pending_inputs = []

class AgentOrchestrator:
    """
    Orchestrates multiple agents for complex workflows.
    
    Features:
    - Multi-agent coordination
    - Sequential and parallel execution
    - User input handling
    - State management
    - Error recovery
    - Results aggregation
    """
    
    def __init__(self):
        self.active_workflows: Dict[str, Workflow] = {}
        self.pending_inputs: Dict[str, UserInputRequest] = {}
        self.workflow_history: List[str] = []
        self._unified_router = None  # Lazy loading to avoid circular imports
    
    def _get_unified_router(self):
        """Lazy import of unified router to avoid circular dependency."""
        if self._unified_router is None:
            from app.unified_router import unified_router
            self._unified_router = unified_router
        return self._unified_router
    
    def analyze_and_create_workflow(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Analyze a complex query and determine if it needs multi-agent workflow.
        
        Args:
            query: User query
            session_id: Session identifier
            
        Returns:
            Dictionary with routing decision and workflow if needed
        """
        try:
            # Input validation
            if not query or not query.strip():
                return {
                    "needs_workflow": False,
                    "error": "Empty query provided",
                    "final_answer": "Please provide a valid query to process.",
                    "reasoning": "Query validation failed - empty input"
                }
            
            # Handle conversational queries first - these should never need workflows
            if self._is_conversational_query(query):
                return {
                    "needs_workflow": False,
                    "simple_route": None,  # Let unified router handle it
                    "reasoning": "Conversational query - no workflow needed"
                }
            
            # First, check if this is a simple single-agent query
            try:
                simple_route = self._try_simple_routing(query, session_id)
                
                # Check if simple routing failed completely
                if simple_route.get("route_type") == "none" or "error" in simple_route:
                    return {
                        "needs_workflow": False,
                        "simple_route": simple_route,
                        "final_answer": simple_route.get("final_answer", "I couldn't find an appropriate way to handle your request. Please try rephrasing your question or check if the required services are available."),
                        "reasoning": "Simple routing failed - no suitable agents found"
                    }
                
                # If simple routing has high confidence, use it
                if simple_route.get("confidence", 0) > 80:
                    return {
                        "needs_workflow": False,
                        "simple_route": simple_route,
                        "reasoning": "Query can be handled by a single agent with high confidence"
                    }
            except Exception as routing_error:
                print(f"[AgentOrchestrator] Simple routing failed: {routing_error}")
                return {
                    "needs_workflow": False,
                    "error": f"Routing system failed: {str(routing_error)}",
                    "final_answer": "I'm currently unable to process your request due to a system issue. Please try again later.",
                    "reasoning": f"Simple routing error: {str(routing_error)}"
                }

            # Analyze for multi-agent needs
            try:
                workflow_analysis = self._analyze_workflow_requirements(query, session_id)
            except Exception as analysis_error:
                print(f"[AgentOrchestrator] Workflow analysis failed: {analysis_error}")
                # Fall back to simple routing if workflow analysis fails
                return {
                    "needs_workflow": False,
                    "simple_route": simple_route,
                    "final_answer": simple_route.get("final_answer", "I'll handle this as a simple request since advanced workflow analysis is currently unavailable."),
                    "reasoning": f"Workflow analysis failed, using simple routing: {str(analysis_error)}"
                }
            
            # Check if analysis indicates no multi-agent workflow needed
            if not workflow_analysis.get("needs_multi_agent", False):
                # Fallback to simple routing
                return {
                    "needs_workflow": False,
                    "simple_route": simple_route,
                    "reasoning": "Query doesn't require multi-agent coordination"
                }
            
            # Create workflow
            try:
                workflow = self._create_workflow_from_analysis(query, session_id, workflow_analysis)
                
                return {
                    "needs_workflow": True,
                    "workflow": workflow,
                    "workflow_id": workflow.workflow_id,
                    "reasoning": workflow_analysis.get("reasoning", "Multi-agent workflow required")
                }
            except Exception as workflow_error:
                print(f"[AgentOrchestrator] Workflow creation failed: {workflow_error}")
                # Fall back to simple routing if workflow creation fails
                return {
                    "needs_workflow": False,
                    "simple_route": simple_route,
                    "final_answer": simple_route.get("final_answer", "I'll handle this as a simple request since I couldn't create the required workflow."),
                    "reasoning": f"Workflow creation failed, using simple routing: {str(workflow_error)}"
                }
            
        except Exception as e:
            print(f"[AgentOrchestrator] Error analyzing workflow: {e}")
            # Final fallback - try to get a simple route one more time
            try:
                simple_route = self._get_unified_router().route_query(query, session_id, enable_orchestration=False)
                return {
                    "needs_workflow": False,
                    "simple_route": simple_route,
                    "final_answer": simple_route.get("final_answer", "I encountered an issue but was able to process your request using basic routing."),
                    "reasoning": f"Workflow analysis failed, using emergency fallback routing: {str(e)}"
                }
            except Exception as fallback_error:
                print(f"[AgentOrchestrator] Emergency fallback also failed: {fallback_error}")
                return {
                    "needs_workflow": False,
                    "error": f"Complete system failure: {str(e)} | Fallback error: {str(fallback_error)}",
                    "final_answer": "I'm currently unable to process your request due to system issues. Please try again later or contact support if the problem persists.",
                    "reasoning": "Complete orchestration system failure"
                }
    
    def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute a workflow with proper state management and timeout protection.
        
        Args:
            workflow_id: ID of workflow to execute
            
        Returns:
            Workflow execution results
        """
        if workflow_id not in self.active_workflows:
            return {
                "status": "error",
                "error": f"Workflow {workflow_id} not found",
                "final_answer": "The requested workflow could not be found. It may have been cancelled or expired."
            }
        
        workflow = self.active_workflows[workflow_id]
        
        # Check if workflow is already in a final state
        if workflow.status in [WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.CANCELLED]:
            return {
                "status": workflow.status.value,
                "workflow_id": workflow_id,
                "final_answer": workflow.final_answer or f"Workflow is already {workflow.status.value}",
                "results": workflow.results
            }
        
        try:
            # Start workflow execution
            workflow.status = WorkflowState.RUNNING
            workflow.started_at = datetime.utcnow()
            
            print(f"[AgentOrchestrator] Starting workflow: {workflow.title}")
            
            # Emit workflow started event
            workflow_streamer.emit_workflow_started(
                workflow_id=workflow_id,
                session_id=workflow.session_id,
                title=workflow.title,
                description=workflow.description,
                steps=len(workflow.steps)
            )
            
            # Execute steps with timeout protection
            execution_result = self._execute_workflow_steps(workflow)
            
            # Check for waiting_input status (user interaction required)
            if execution_result.get("status") == "waiting_input":
                workflow.status = WorkflowState.WAITING_INPUT
                return {
                    "status": "waiting_input",
                    "workflow_id": workflow_id,
                    "pending_input": execution_result.get("pending_input"),
                    "final_answer": "This workflow requires your input to continue. Please provide the requested information.",
                    "results": workflow.results
                }
            
            if execution_result.get("status") == "completed":
                workflow.status = WorkflowState.COMPLETED
                workflow.completed_at = datetime.utcnow()
                
                # Calculate execution time
                execution_time = (workflow.completed_at - workflow.started_at).total_seconds()
                
                # Generate final answer
                try:
                    final_answer = self._generate_final_answer(workflow)
                    workflow.final_answer = final_answer
                except Exception as final_answer_error:
                    print(f"[AgentOrchestrator] Error generating final answer: {final_answer_error}")
                    final_answer = f"Workflow completed successfully. {len(workflow.results)} steps were executed, but I encountered an issue summarizing the results. You can review the detailed results below."
                    workflow.final_answer = final_answer
                
                # Emit workflow completed event
                workflow_streamer.emit_workflow_completed(
                    workflow_id=workflow_id,
                    session_id=workflow.session_id,
                    final_answer=final_answer,
                    execution_time=execution_time
                )
                
                # Update session
                try:
                    session_manager.add_to_session(workflow.session_id, workflow.user_query, final_answer)
                except Exception as session_error:
                    print(f"[AgentOrchestrator] Error updating session: {session_error}")
                    # Don't fail the workflow for session update errors
                
                return {
                    "status": "completed",
                    "workflow_id": workflow_id,
                    "final_answer": final_answer,
                    "results": workflow.results,
                    "execution_summary": self._get_execution_summary(workflow)
                }
            
            elif execution_result.get("status") == "waiting_input":
                workflow.status = WorkflowState.WAITING_INPUT
                
                # Emit user input required event
                pending_input = execution_result.get("pending_input")
                if pending_input:
                    workflow_streamer.emit_user_input_required(
                        workflow_id=workflow_id,
                        session_id=workflow.session_id,
                        step_id=pending_input.get("step_id"),
                        input_request=pending_input
                    )
                
                return {
                    "status": "waiting_input",
                    "workflow_id": workflow_id,
                    "pending_input": execution_result.get("pending_input"),
                    "partial_results": workflow.results
                }
            
            else:
                # Handle errors
                workflow.status = WorkflowState.FAILED
                workflow.completed_at = datetime.utcnow()
                error_msg = execution_result.get("error", "Unknown workflow execution error")
                final_answer = execution_result.get("final_answer", f"I wasn't able to complete your request: {error_msg}")
                workflow.final_answer = final_answer
                
                # Emit workflow failed event
                workflow_streamer.emit_error(
                    workflow_id=workflow_id,
                    session_id=workflow.session_id,
                    step_id=workflow.current_step,
                    error_message=error_msg
                )
                
                return {
                    "status": "failed",
                    "workflow_id": workflow_id,
                    "error": error_msg,
                    "final_answer": final_answer,
                    "partial_results": workflow.results
                }
                
        except Exception as e:
            workflow.status = WorkflowState.FAILED
            workflow.completed_at = datetime.utcnow()
            error_msg = f"Workflow execution error: {str(e)}"
            final_answer = f"I encountered an unexpected error while processing your request: {str(e)}. Please try again or contact support if the issue persists."
            workflow.final_answer = final_answer
            
            print(f"[AgentOrchestrator] {error_msg}")
            
            # Emit workflow failed event
            workflow_streamer.emit_error(
                workflow_id=workflow_id,
                session_id=workflow.session_id,
                step_id=workflow.current_step,
                error_message=error_msg,
                error_details=str(e)
            )
            
            return {
                "status": "failed",
                "workflow_id": workflow_id,
                "error": error_msg,
                "final_answer": final_answer,
                "partial_results": workflow.results
            }
    
    def provide_user_input(self, input_id: str, user_response: Any) -> Dict[str, Any]:
        """
        Provide user input to continue workflow execution.
        
        Args:
            input_id: ID of the input request
            user_response: User's response
            
        Returns:
            Updated workflow execution results
        """
        if input_id not in self.pending_inputs:
            return {
                "status": "error",
                "error": f"Input request {input_id} not found or already completed"
            }
        
        input_request = self.pending_inputs[input_id]
        workflow_id = input_request.workflow_id
        
        try:
            # Validate input
            validation_result = self._validate_user_input(input_request, user_response)
            if not validation_result.get("valid", False):
                return {
                    "status": "error",
                    "error": f"Invalid input: {validation_result.get('error', 'Unknown validation error')}"
                }
            
            # Store the input
            workflow = self.active_workflows[workflow_id]
            step = next((s for s in workflow.steps if s.step_id == input_request.step_id), None)
            if step:
                step.result = {"user_input": user_response}
                step.status = WorkflowState.COMPLETED
                step.completed_at = datetime.utcnow()
            
            # Remove from pending
            del self.pending_inputs[input_id]
            workflow.pending_inputs = [inp for inp in workflow.pending_inputs if inp.input_id != input_id]
            
            # Continue workflow execution
            return self.execute_workflow(workflow_id)
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Error processing user input: {str(e)}"
            }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow."""
        if workflow_id not in self.active_workflows:
            return {
                "status": "error",
                "error": f"Workflow {workflow_id} not found"
            }
        
        workflow = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "title": workflow.title,
            "status": workflow.status.value,
            "current_step": workflow.current_step,
            "progress": self._calculate_progress(workflow),
            "pending_inputs": [asdict(inp) for inp in workflow.pending_inputs],
            "results": workflow.results,
            "created_at": workflow.created_at.isoformat() if workflow.created_at else None,
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None
        }
    
    def cancel_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Cancel an active workflow."""
        if workflow_id not in self.active_workflows:
            return {
                "status": "error",
                "error": f"Workflow {workflow_id} not found"
            }
        
        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowState.CANCELLED
        workflow.completed_at = datetime.utcnow()
        
        # Clean up pending inputs
        for inp in workflow.pending_inputs:
            if inp.input_id in self.pending_inputs:
                del self.pending_inputs[inp.input_id]
        
        return {
            "status": "cancelled",
            "workflow_id": workflow_id,
            "message": f"Workflow '{workflow.title}' has been cancelled"
        }
    
    def _try_simple_routing(self, query: str, session_id: str) -> Dict[str, Any]:
        """Try simple single-agent routing first."""
        return self._get_unified_router().route_query(query, session_id, enable_orchestration=False)
    
    def _analyze_workflow_requirements(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Analyze if query requires multi-agent workflow.
        
        Uses LLM to understand query complexity and agent requirements,
        enhanced with pattern detection and templates.
        """
        # First, try to detect common patterns
        try:
            from app.workflow_templates import WorkflowExamples
            detected_pattern = WorkflowExamples.detect_workflow_pattern(query)
            print(f"[AgentOrchestrator] Detected workflow pattern: {detected_pattern}")
        except Exception as e:
            print(f"[AgentOrchestrator] Error detecting pattern: {e}")
            detected_pattern = "general"
        
        analysis_prompt = f"""
Analyze the following user query to determine if it requires multiple agents or a complex workflow.

USER QUERY: "{query}"
DETECTED PATTERN: {detected_pattern}

Consider these factors:
1. Does it need data from multiple different databases/sources?
2. Does it require sequential operations (get data, then transform, then analyze)?
3. Does it need user input or confirmation during execution?
4. Does it require aggregation of results from multiple sources?
5. Does it have conditional logic based on intermediate results?
6. Does it involve comparisons between different data sources?

Scoring Guidelines:
- Single data source, simple query: complexity_score 1-3, needs_multi_agent = false
- Multiple sources or transformations: complexity_score 4-6, needs_multi_agent = true
- Complex conditional logic or user interaction: complexity_score 7-10, needs_multi_agent = true

Respond with JSON:
{{
    "needs_multi_agent": boolean,
    "complexity_score": number (1-10),
    "reasoning": "detailed explanation",
    "detected_pattern": "{detected_pattern}",
    "suggested_steps": [
        {{
            "step_type": "data_query|application_call|user_input|aggregation|data_transformation|condition",
            "description": "what this step does",
            "agent_requirements": "what kind of agent needed",
            "depends_on_previous": boolean,
            "requires_user_input": boolean
        }}
    ],
    "estimated_duration_minutes": number,
    "user_input_needed": boolean,
    "risk_factors": ["list", "of", "potential", "issues"]
}}

Examples of multi-agent queries:
- "Compare sales data from our CRM with inventory levels and suggest reorder quantities" (needs 2+ data sources + analysis)
- "Get customer feedback from multiple sources and create a summary report" (multiple sources + aggregation)
- "Check our financial data and if revenue is down more than 10%, send alerts" (conditional logic)
- "Analyze customer behavior and identify retention opportunities" (complex analysis + segmentation)

Examples of single-agent queries:
- "Show me last month's sales" (single source, simple query)
- "List all customers in California" (single source, simple filter)
- "What's the weather in New York?" (single API call)
"""
        
        try:
            response = llm_client.invoke_with_json_response(analysis_prompt)
            
            if response and isinstance(response, dict):
                # Add detected pattern to response
                response["detected_pattern"] = detected_pattern
                return response
            else:
                # Default to simple routing if analysis fails
                return {
                    "needs_multi_agent": False,
                    "complexity_score": 1,
                    "reasoning": "Could not analyze query complexity",
                    "detected_pattern": detected_pattern
                }
                
        except Exception as e:
            print(f"[AgentOrchestrator] Error in workflow analysis: {e}")
            return {
                "needs_multi_agent": False,
                "complexity_score": 1,
                "reasoning": f"Analysis failed: {str(e)}",
                "detected_pattern": detected_pattern
            }
    
    def _create_workflow_from_analysis(self, query: str, session_id: str, analysis: Dict[str, Any]) -> Workflow:
        """Create a workflow based on analysis results, using templates when possible."""
        
        # Try to use a template based on detected pattern
        detected_pattern = analysis.get("detected_pattern", "general")
        
        try:
            from app.workflow_templates import WorkflowTemplates
            
            # Use template-based creation for known patterns
            if detected_pattern == "data_comparison":
                workflow = WorkflowTemplates.create_data_comparison_workflow(query, session_id)
                print(f"[AgentOrchestrator] Using data comparison template for workflow")
                self.active_workflows[workflow.workflow_id] = workflow
                self.workflow_history.append(workflow.workflow_id)
                return workflow
                
            elif detected_pattern == "financial_analysis":
                workflow = WorkflowTemplates.create_financial_analysis_workflow(query, session_id)
                print(f"[AgentOrchestrator] Using financial analysis template for workflow")
                self.active_workflows[workflow.workflow_id] = workflow
                self.workflow_history.append(workflow.workflow_id)
                return workflow
                
            elif detected_pattern == "customer_insights":
                workflow = WorkflowTemplates.create_customer_insight_workflow(query, session_id)
                print(f"[AgentOrchestrator] Using customer insights template for workflow")
                self.active_workflows[workflow.workflow_id] = workflow
                self.workflow_history.append(workflow.workflow_id)
                return workflow
                
            elif detected_pattern == "conditional_workflows":
                workflow = WorkflowTemplates.create_conditional_workflow(query, session_id)
                print(f"[AgentOrchestrator] Using conditional workflow template")
                self.active_workflows[workflow.workflow_id] = workflow
                self.workflow_history.append(workflow.workflow_id)
                return workflow
                
        except Exception as e:
            print(f"[AgentOrchestrator] Error using templates, falling back to dynamic creation: {e}")
        
        # Fallback to dynamic workflow creation
        workflow_id = str(uuid.uuid4())
        
        # Create steps from analysis
        steps = []
        for i, step_info in enumerate(analysis.get("suggested_steps", [])):
            step_id = f"step_{i+1}"
            step_type_str = step_info.get("step_type", "data_query")
            
            # Map step type string to enum
            try:
                step_type = StepType(step_type_str)
            except ValueError:
                print(f"[AgentOrchestrator] Unknown step type: {step_type_str}, defaulting to data_query")
                step_type = StepType.DATA_QUERY
            
            # Determine dependencies
            depends_on = None
            if i > 0 and step_info.get("depends_on_previous", False):
                depends_on = [f"step_{i}"]
            
            # Create step parameters based on type
            parameters = {}
            if step_type == StepType.USER_INPUT and step_info.get("requires_user_input", False):
                parameters = {
                    "input_type": "text",
                    "timeout_minutes": 10
                }
            
            step = WorkflowStep(
                step_id=step_id,
                step_type=step_type,
                description=step_info.get("description", f"Step {i+1}"),
                query=query if step_type == StepType.DATA_QUERY else None,
                depends_on=depends_on,
                parameters=parameters
            )
            steps.append(step)
        
        # Create workflow
        workflow = Workflow(
            workflow_id=workflow_id,
            title=f"Multi-Agent Query: {query[:50]}...",
            description=analysis.get("reasoning", "Complex multi-agent workflow"),
            steps=steps,
            session_id=session_id,
            user_query=query
        )
        
        # Store workflow
        self.active_workflows[workflow_id] = workflow
        self.workflow_history.append(workflow_id)
        
        return workflow
    
    def _execute_workflow_steps(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute workflow steps with dependency management."""
        completed_steps = set()
        max_iterations = len(workflow.steps) * 2  # Prevent infinite loops
        iteration_count = 0
        
        while len(completed_steps) < len(workflow.steps):
            progress_made = False
            iteration_count += 1
            
            # Safety check to prevent infinite loops
            if iteration_count > max_iterations:
                error_msg = f"Workflow execution exceeded maximum iterations ({max_iterations}). This might indicate circular dependencies or blocked steps."
                print(f"[AgentOrchestrator] {error_msg}")
                
                # Emit error event
                workflow_streamer.emit_error(
                    workflow_id=workflow.workflow_id,
                    session_id=workflow.session_id,
                    step_id=workflow.current_step,
                    error_message=error_msg,
                    error_details="Maximum iteration safety limit reached"
                )
                
                return {
                    "status": "failed",
                    "error": error_msg,
                    "final_answer": "I wasn't able to complete your request due to workflow complexity issues. Please try breaking your request into smaller, simpler questions."
                }
            
            for step in workflow.steps:
                if step.step_id in completed_steps or step.status in [WorkflowState.RUNNING, WorkflowState.COMPLETED]:
                    continue
                
                # Check dependencies
                if step.depends_on:
                    if not all(dep_id in completed_steps for dep_id in step.depends_on):
                        continue  # Dependencies not met
                
                # Execute step
                workflow.current_step = step.step_id
                
                # Emit progress update
                progress = (len(completed_steps) / len(workflow.steps)) * 100
                workflow_streamer.emit_progress_update(
                    workflow_id=workflow.workflow_id,
                    session_id=workflow.session_id,
                    progress_percent=progress,
                    current_step=step.step_id
                )
                
                try:
                    step_result = self._execute_single_step(workflow, step)
                    
                    if step_result.get("status") == "completed":
                        completed_steps.add(step.step_id)
                        progress_made = True
                    elif step_result.get("status") == "waiting_input":
                        return step_result
                    elif step_result.get("status") == "failed":
                        error_msg = f"Step '{step.description}' failed: {step_result.get('error', 'Unknown error')}"
                        return {
                            "status": "failed",
                            "error": error_msg,
                            "failed_step": step.step_id,
                            "final_answer": f"I encountered an error while processing your request: {error_msg}. Please try again or rephrase your question."
                        }
                except Exception as step_error:
                    error_msg = f"Step '{step.description}' encountered an unexpected error: {str(step_error)}"
                    print(f"[AgentOrchestrator] {error_msg}")
                    
                    # Emit error event
                    workflow_streamer.emit_error(
                        workflow_id=workflow.workflow_id,
                        session_id=workflow.session_id,
                        step_id=step.step_id,
                        error_message=error_msg,
                        error_details=str(step_error)
                    )
                    
                    return {
                        "status": "failed",
                        "error": error_msg,
                        "failed_step": step.step_id,
                        "final_answer": f"I encountered an unexpected error while processing your request. Please try again later."
                    }
            
            if not progress_made:
                # Analyze what's blocking progress
                blocked_steps = []
                for step in workflow.steps:
                    if step.step_id not in completed_steps and step.status not in [WorkflowState.COMPLETED]:
                        if step.depends_on:
                            missing_deps = [dep for dep in step.depends_on if dep not in completed_steps]
                            blocked_steps.append(f"Step '{step.description}' waiting for: {', '.join(missing_deps)}")
                        else:
                            blocked_steps.append(f"Step '{step.description}' is blocked for unknown reasons")
                
                error_msg = f"Workflow is stuck - no progress possible. Blocked steps: {'; '.join(blocked_steps)}"
                print(f"[AgentOrchestrator] {error_msg}")
                
                # Emit error event
                workflow_streamer.emit_error(
                    workflow_id=workflow.workflow_id,
                    session_id=workflow.session_id,
                    step_id=workflow.current_step,
                    error_message=error_msg,
                    error_details=f"Completed steps: {completed_steps}, Total steps: {len(workflow.steps)}"
                )
                
                return {
                    "status": "failed",
                    "error": error_msg,
                    "final_answer": "I wasn't able to complete your request due to workflow dependency issues. Please try breaking your request into simpler parts."
                }
        
        return {"status": "completed"}
    
    def _execute_single_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single workflow step."""
        try:
            step.status = WorkflowState.RUNNING
            step.started_at = datetime.utcnow()
            
            print(f"[AgentOrchestrator] Executing step: {step.step_id} ({step.step_type.value})")
            
            # Emit step started event
            workflow_streamer.emit_step_started(
                workflow_id=workflow.workflow_id,
                session_id=workflow.session_id,
                step_id=step.step_id,
                step_type=step.step_type.value,
                description=step.description
            )
            
            if step.step_type == StepType.DATA_QUERY:
                result = self._execute_data_query_step(workflow, step)
            elif step.step_type == StepType.APPLICATION_CALL:
                result = self._execute_application_step(workflow, step)
            elif step.step_type == StepType.USER_INPUT:
                result = self._execute_user_input_step(workflow, step)
            elif step.step_type == StepType.AGGREGATION:
                result = self._execute_aggregation_step(workflow, step)
            elif step.step_type == StepType.DATA_TRANSFORMATION:
                result = self._execute_transformation_step(workflow, step)
            elif step.step_type == StepType.CONDITION:
                result = self._execute_condition_step(workflow, step)
            else:
                step.status = WorkflowState.FAILED
                step.error = f"Unknown step type: {step.step_type}"
                result = {
                    "status": "failed",
                    "error": step.error
                }
            
            # Emit step completed event (if successful)
            if result.get("status") == "completed" and step.started_at:
                execution_time = (datetime.utcnow() - step.started_at).total_seconds()
                workflow_streamer.emit_step_completed(
                    workflow_id=workflow.workflow_id,
                    session_id=workflow.session_id,
                    step_id=step.step_id,
                    step_type=step.step_type.value,
                    execution_time=execution_time
                )
            
            return result
                
        except Exception as e:
            step.status = WorkflowState.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _execute_data_query_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a data query step using existing routing."""
        try:
            # Emit thinking event
            workflow_streamer.emit_llm_thinking(
                workflow_id=workflow.workflow_id,
                session_id=workflow.session_id,
                step_id=step.step_id,
                thinking_about="Analyzing query and selecting appropriate data agent"
            )
            
            # Use the unified router to execute the query
            query_result = self._get_unified_router().route_query(step.query or workflow.user_query, workflow.session_id, enable_orchestration=False)
            
            # Emit data query event if we have SQL query info
            if query_result.get("sql_query"):
                workflow_streamer.emit_data_query(
                    workflow_id=workflow.workflow_id,
                    session_id=workflow.session_id,
                    step_id=step.step_id,
                    query=query_result["sql_query"],
                    database_type=query_result.get("route_type", "unknown")
                )
            
            # Store results
            step.result = query_result
            step.status = WorkflowState.COMPLETED
            step.completed_at = datetime.utcnow()
            
            # Store in workflow results
            workflow.results[step.step_id] = query_result
            
            # Emit debug info about results
            workflow_streamer.emit_debug_info(
                workflow_id=workflow.workflow_id,
                session_id=workflow.session_id,
                step_id=step.step_id,
                debug_message=f"Data query completed successfully",
                debug_data={
                    "route_type": query_result.get("route_type"),
                    "selected_agent": query_result.get("selected_agent"),
                    "confidence": query_result.get("confidence")
                }
            )
            
            return {"status": "completed", "result": query_result}
            
        except Exception as e:
            step.status = WorkflowState.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            
            # Emit error event
            workflow_streamer.emit_error(
                workflow_id=workflow.workflow_id,
                session_id=workflow.session_id,
                step_id=step.step_id,
                error_message=f"Data query step failed: {str(e)}",
                error_details=str(e)
            )
            
            return {"status": "failed", "error": str(e)}
    
    def _execute_application_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Execute an application call step."""
        # Similar to data query but specifically for application endpoints
        return self._execute_data_query_step(workflow, step)
    
    def _execute_user_input_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a user input step."""
        try:
            # Create input request
            input_id = str(uuid.uuid4())
            input_request = UserInputRequest(
                input_id=input_id,
                workflow_id=workflow.workflow_id,
                step_id=step.step_id,
                prompt=step.description,
                input_type=step.parameters.get("input_type", "text") if step.parameters else "text",
                choices=step.parameters.get("choices") if step.parameters else None,
                validation_rules=step.parameters.get("validation") if step.parameters else None
            )
            
            # Store input request
            self.pending_inputs[input_id] = input_request
            workflow.pending_inputs.append(input_request)
            
            step.status = WorkflowState.WAITING_INPUT
            
            return {
                "status": "waiting_input",
                "pending_input": asdict(input_request)
            }
            
        except Exception as e:
            step.status = WorkflowState.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            return {"status": "failed", "error": str(e)}
    
    def _execute_aggregation_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Execute an aggregation step that combines results from previous steps."""
        try:
            # Collect results from dependent steps
            dependent_results = {}
            if step.depends_on:
                for dep_step_id in step.depends_on:
                    if dep_step_id in workflow.results:
                        dependent_results[dep_step_id] = workflow.results[dep_step_id]
            
            # Emit thinking event
            workflow_streamer.emit_llm_thinking(
                workflow_id=workflow.workflow_id,
                session_id=workflow.session_id,
                step_id=step.step_id,
                thinking_about=f"Aggregating results from {len(dependent_results)} previous steps"
            )
            
            # Use LLM to aggregate results
            aggregation_prompt = f"""
Aggregate and summarize the following results from multiple data sources:

ORIGINAL QUERY: {workflow.user_query}
AGGREGATION TASK: {step.description}

RESULTS TO AGGREGATE:
{json.dumps(dependent_results, indent=2, default=str)}

Provide a comprehensive summary that addresses the original query by combining insights from all data sources.
Format the response as clear, actionable information for the user.
"""
            
            aggregated_result = llm_client.invoke_with_text_response(aggregation_prompt)
            
            # Store results
            step.result = {
                "aggregated_data": dependent_results,
                "summary": aggregated_result
            }
            step.status = WorkflowState.COMPLETED
            step.completed_at = datetime.utcnow()
            
            workflow.results[step.step_id] = step.result
            
            # Emit debug info
            workflow_streamer.emit_debug_info(
                workflow_id=workflow.workflow_id,
                session_id=workflow.session_id,
                step_id=step.step_id,
                debug_message=f"Aggregation completed, processed {len(dependent_results)} data sources"
            )
            
            return {"status": "completed", "result": step.result}
            
        except Exception as e:
            step.status = WorkflowState.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            
            # Emit error event
            workflow_streamer.emit_error(
                workflow_id=workflow.workflow_id,
                session_id=workflow.session_id,
                step_id=step.step_id,
                error_message=f"Aggregation step failed: {str(e)}",
                error_details=str(e)
            )
            
            return {"status": "failed", "error": str(e)}
    
    def _execute_transformation_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a data transformation step."""
        try:
            # Get data from previous steps
            source_data = {}
            if step.depends_on:
                for dep_step_id in step.depends_on:
                    if dep_step_id in workflow.results:
                        source_data[dep_step_id] = workflow.results[dep_step_id]
            
            # Use LLM for transformation logic
            transformation_prompt = f"""
Transform the following data according to the specified transformation:

TRANSFORMATION TASK: {step.description}
TRANSFORMATION PARAMETERS: {json.dumps(step.parameters, indent=2) if step.parameters else 'None'}

SOURCE DATA:
{json.dumps(source_data, indent=2, default=str)}

Apply the transformation and return the processed data in a structured format.
"""
            
            transformed_result = llm_client.invoke_with_text_response(transformation_prompt)
            
            # Store results
            step.result = {
                "source_data": source_data,
                "transformed_data": transformed_result
            }
            step.status = WorkflowState.COMPLETED
            step.completed_at = datetime.utcnow()
            
            workflow.results[step.step_id] = step.result
            
            return {"status": "completed", "result": step.result}
            
        except Exception as e:
            step.status = WorkflowState.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            return {"status": "failed", "error": str(e)}
    
    def _execute_condition_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a conditional step that can alter workflow execution."""
        try:
            # Get data for condition evaluation
            condition_data = {}
            if step.depends_on:
                for dep_step_id in step.depends_on:
                    if dep_step_id in workflow.results:
                        condition_data[dep_step_id] = workflow.results[dep_step_id]
            
            # Evaluate condition using LLM
            condition_prompt = f"""
Evaluate the following condition based on the provided data:

CONDITION: {step.condition or step.description}
CONDITION PARAMETERS: {json.dumps(step.parameters, indent=2) if step.parameters else 'None'}

DATA FOR EVALUATION:
{json.dumps(condition_data, indent=2, default=str)}

Return JSON with the evaluation result:
{{
    "condition_met": boolean,
    "reason": "explanation of the evaluation",
    "recommended_action": "what should happen next"
}}
"""
            
            condition_result = llm_client.invoke_with_json_response(condition_prompt)
            
            # Store results
            step.result = condition_result
            step.status = WorkflowState.COMPLETED
            step.completed_at = datetime.utcnow()
            
            workflow.results[step.step_id] = step.result
            
            return {"status": "completed", "result": step.result}
            
        except Exception as e:
            step.status = WorkflowState.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            return {"status": "failed", "error": str(e)}
    
    def _validate_user_input(self, input_request: UserInputRequest, user_response: Any) -> Dict[str, Any]:
        """Validate user input based on type and rules."""
        try:
            if input_request.input_type == "choice" and input_request.choices:
                if user_response not in input_request.choices:
                    return {
                        "valid": False,
                        "error": f"Invalid choice. Must be one of: {', '.join(input_request.choices)}"
                    }
            
            if input_request.validation_rules:
                # Apply custom validation rules
                rules = input_request.validation_rules
                
                if "min_length" in rules and len(str(user_response)) < rules["min_length"]:
                    return {
                        "valid": False,
                        "error": f"Input must be at least {rules['min_length']} characters"
                    }
                
                if "max_length" in rules and len(str(user_response)) > rules["max_length"]:
                    return {
                        "valid": False,
                        "error": f"Input must be no more than {rules['max_length']} characters"
                    }
                
                if "pattern" in rules:
                    import re
                    if not re.match(rules["pattern"], str(user_response)):
                        return {
                            "valid": False,
                            "error": f"Input must match pattern: {rules['pattern']}"
                        }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }
    
    def _generate_final_answer(self, workflow: Workflow) -> str:
        """Generate final answer from workflow results using proper data formatting."""
        try:
            # Check if we have data agent results that need proper formatting
            if workflow.results and any('data' in result for result in workflow.results):
                # Use the data-specific formatting approach
                return self._format_data_agent_results(workflow)
            else:
                # Use generic business analysis approach for non-data queries
                final_prompt = f"""
Generate a comprehensive final answer based on the completed workflow:

ORIGINAL USER QUERY: {workflow.user_query}
WORKFLOW DESCRIPTION: {workflow.description}

WORKFLOW RESULTS:
{json.dumps(workflow.results, indent=2, default=str)}

Provide a clear, comprehensive answer that addresses the original query using all the gathered information.
Format the response to be user-friendly and actionable.
"""
                
                return llm_client.invoke_with_text_response(final_prompt, allow_diagrams=True)
            
        except Exception as e:
            print(f"[AgentOrchestrator] Error generating final answer: {e}")
            return f"Workflow completed successfully, but there was an error generating the final summary. Results: {workflow.results}"
    
    def _format_data_agent_results(self, workflow: Workflow) -> str:
        """Format data agent results using the proper data formatting approach."""
        try:
            # Extract the main data result (use the first data result for primary formatting)
            primary_result = None
            all_results_text = ""
            
            for result in workflow.results:
                if 'data' in result and result.get('data', {}).get('results'):
                    if primary_result is None:
                        primary_result = result
                    
                    # Build formatted text for all results
                    agent_name = result.get('agent_name', 'Unknown Agent')
                    data = result.get('data', {})
                    query = result.get('query', 'Unknown Query')
                    
                    all_results_text += f"=== {agent_name} ===\n"
                    all_results_text += f"Records Retrieved: {len(data.get('results', []))}\n"
                    all_results_text += f"Data: {data}\n"
                    all_results_text += f"Query Executed: {query}\n"
            
            if primary_result:
                # Use the LLM client's data formatting approach
                query = workflow.user_query
                sql_query = primary_result.get('query', 'Data query executed')
                query_result = {'results': [], 'sampling_info': {'strategy_used': 'aggregation', 'row_count': 0}}
                
                # Combine all data results
                for result in workflow.results:
                    if 'data' in result and result.get('data', {}).get('results'):
                        query_result['results'].extend(result['data']['results'])
                        if 'sampling_info' in result['data']:
                            query_result['sampling_info'] = result['data']['sampling_info']
                
                query_result['sampling_info']['row_count'] = len(query_result['results'])
                
                # Use the LLM client's create_data_answer_prompt method
                data_prompt = llm_client.create_data_answer_prompt(query, sql_query, query_result)
                return llm_client.invoke_with_text_response(data_prompt, allow_diagrams=True)
            else:
                # Fallback if no proper data results found
                return f"No specific data results found in the workflow. Results: {workflow.results}"
                
        except Exception as e:
            print(f"[AgentOrchestrator] Error formatting data agent results: {e}")
            # Fallback to generic approach
            final_prompt = f"""
Generate a comprehensive final answer based on the completed workflow:

ORIGINAL USER QUERY: {workflow.user_query}
WORKFLOW DESCRIPTION: {workflow.description}

WORKFLOW RESULTS:
{json.dumps(workflow.results, indent=2, default=str)}

Provide a clear, comprehensive answer that addresses the original query using all the gathered information.
Format the response to be user-friendly and actionable.
"""
            return llm_client.invoke_with_text_response(final_prompt, allow_diagrams=True)
    
    def _calculate_progress(self, workflow: Workflow) -> float:
        """Calculate workflow progress percentage."""
        if not workflow.steps:
            return 0.0
        
        completed = sum(1 for step in workflow.steps if step.status == WorkflowState.COMPLETED)
        return (completed / len(workflow.steps)) * 100
    
    def _get_execution_summary(self, workflow: Workflow) -> Dict[str, Any]:
        """Get execution summary for completed workflow."""
        total_time = None
        if workflow.started_at and workflow.completed_at:
            total_time = (workflow.completed_at - workflow.started_at).total_seconds()
        
        step_summary = []
        for step in workflow.steps:
            step_time = None
            if step.started_at and step.completed_at:
                step_time = (step.completed_at - step.started_at).total_seconds()
            
            step_summary.append({
                "step_id": step.step_id,
                "type": step.step_type.value,
                "status": step.status.value,
                "execution_time_seconds": step_time,
                "description": step.description
            })
        
        return {
            "total_execution_time_seconds": total_time,
            "total_steps": len(workflow.steps),
            "completed_steps": sum(1 for step in workflow.steps if step.status == WorkflowState.COMPLETED),
            "failed_steps": sum(1 for step in workflow.steps if step.status == WorkflowState.FAILED),
            "step_details": step_summary
        }
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get list of all active workflows."""
        return [
            {
                "workflow_id": wf.workflow_id,
                "title": wf.title,
                "status": wf.status.value,
                "progress": self._calculate_progress(wf),
                "created_at": wf.created_at.isoformat() if wf.created_at else None
            }
            for wf in self.active_workflows.values()
        ]
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24):
        """Clean up old completed workflows."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        workflows_to_remove = []
        for workflow_id, workflow in self.active_workflows.items():
            if (workflow.status in [WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.CANCELLED] and
                workflow.completed_at and workflow.completed_at < cutoff_time):
                workflows_to_remove.append(workflow_id)
        
        for workflow_id in workflows_to_remove:
            del self.active_workflows[workflow_id]
            if workflow_id in self.workflow_history:
                self.workflow_history.remove(workflow_id)
        
        print(f"[AgentOrchestrator] Cleaned up {len(workflows_to_remove)} old workflows")

    def _is_conversational_query(self, query: str) -> bool:
        """Check if query is a simple conversational query that doesn't need workflows."""
        query_lower = query.lower().strip()
        
        # Define conversational patterns
        conversational_patterns = [
            # Greetings
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            # Goodbyes  
            "bye", "goodbye", "see you", "farewell", "good night",
            # How are you
            "how are you", "how's it going", "how are things", "what's up",
            # Thanks
            "thank you", "thanks", "thx", "appreciate it",
            # Help requests
            "help", "what can you do", "what are your capabilities", "how do you work"
        ]
        
        # Check for simple conversational patterns
        for pattern in conversational_patterns:
            if pattern in query_lower:
                return True
        
        # Check for very short queries that are likely conversational
        if len(query_lower.split()) <= 3 and any(word in query_lower for word in ["hi", "hello", "hey", "thanks", "help"]):
            return True
            
        return False

# Global instance
agent_orchestrator = AgentOrchestrator()
