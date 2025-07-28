"""
Refactored Simple Orchestrator with modular structure
"""

from typing import Dict, List, Any
from .models import ExecutionContext, create_workflow_context
from .conversation.manager import ConversationManager
from .conversation.intent_analyzer import IntentAnalyzer
from .agents.executor import AgentExecutor
from .handlers.response_handler import ResponseHandler
from app.workflow_streamer import workflow_streamer

class SimpleOrchestrator:
    """Refactored orchestrator with modular, clean structure"""
    
    def __init__(self):
        # Initialize all components
        self.conversation_manager = ConversationManager()
        self.intent_analyzer = IntentAnalyzer()
        self.agent_executor = AgentExecutor()
        self.response_handler = ResponseHandler(self.conversation_manager)
    
    async def execute_workflow(self, user_query: str, session_id: str = None) -> Dict[str, Any]:
        """Main workflow execution with proper error handling"""
        
        # Create execution context
        context = create_workflow_context(user_query, session_id)
        
        # Get conversation history for this session
        context.conversation_history = self.conversation_manager.get_conversation_history(context.session_id)
        
        try:
            # Start workflow
            workflow_streamer.emit_workflow_started(
                context.workflow_id, context.session_id,
                title="Processing your request",
                description="Analyzing and executing your query",
                steps=4
            )
            
            # Step 1: Understand what user wants
            workflow_streamer.emit_step_started(
                context.workflow_id, context.session_id, "analyze",
                "analysis", "🧠 Understanding your request..."
            )
            
            # Build conversation context
            conversation_context = self.conversation_manager.build_conversation_context(context)
            
            # Analyze intent
            intent = await self.intent_analyzer.analyze_intent(context, conversation_context)
            
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, "analyze",
                "analysis", 1.0
            )
            
            # Route to appropriate handler based on intent
            action = intent.action
            
            if action == "greeting":
                return await self.response_handler.handle_greeting(context, intent)
            elif action == "capabilities":
                return await self.response_handler.handle_capabilities(context, intent)
            elif action == "ask_clarification":
                return await self.response_handler.handle_clarification(context, intent)
            elif action == "execute":
                return await self._handle_execution(context, intent)
            else:
                return await self.response_handler.handle_general(context, intent)
                
        except Exception as e:
            workflow_streamer.emit_error(context.workflow_id, context.session_id, "error", str(e))
            return {
                "status": "error",
                "message": f"I encountered an error: {str(e)}"
            }
    
    async def _handle_execution(self, context: ExecutionContext, intent) -> Dict[str, Any]:
        """Handle agent execution with proper validation"""
        
        execution_plan = intent.execution_plan
        if not execution_plan:
            return await self.response_handler.handle_general(context, intent)
        
        strategy = execution_plan.get("strategy", "single")
        selected_agent_id = execution_plan.get("selected_agent_id")
        query_to_execute = execution_plan.get("query_to_execute", context.user_query)
        
        if not selected_agent_id:
            return await self.response_handler.handle_general(context, intent)
        
        # Create steps from the execution plan
        steps = [{
            "agent_id": selected_agent_id,
            "query": query_to_execute
        }]
        
        # Execute based on strategy
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "execute",
            "execution", f"⚡ Executing {strategy} agent plan..."
        )
        
        if strategy == "single":
            results = await self.agent_executor.execute_single(context, steps[0])
        elif strategy == "sequential":
            results = await self.agent_executor.execute_sequential(context, steps)
        elif strategy == "parallel":
            results = await self.agent_executor.execute_parallel(context, steps)
        else:
            results = []
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "execute",
            "execution", 2.0
        )
        
        # Generate final response
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "respond",
            "response", "📝 Generating response..."
        )
        
        response = await self.response_handler.handle_execution_complete(
            context, intent, results, strategy
        )
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "respond",
            "response", 1.0
        )
        
        return response


# Global instance
simple_orchestrator = SimpleOrchestrator()
