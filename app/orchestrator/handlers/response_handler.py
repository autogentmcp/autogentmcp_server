"""
Response handlers for different workflow outcomes
"""

import json
from typing import Dict, List, Any
from ..models import ExecutionContext, IntentAnalysisResult, AgentResult
from ..conversation.manager import ConversationManager
from app.workflow_streamer import workflow_streamer
from app.multimode_llm_client import get_global_llm_client, TaskType
from app.registry import fetch_agents_and_tools_from_registry

class ResponseHandler:
    """Handles different types of workflow responses"""
    
    def __init__(self, conversation_manager: ConversationManager):
        self.conversation_manager = conversation_manager
        self.llm_client = get_global_llm_client()
    
    async def handle_greeting(self, context: ExecutionContext, intent: IntentAnalysisResult) -> Dict[str, Any]:
        """Handle greeting intent"""
        message = intent.message
        
        workflow_streamer.emit_workflow_completed(
            context.workflow_id, context.session_id,
            final_answer=message,
            execution_time=0.5
        )
        
        self.conversation_manager.save_turn(context.session_id, context.user_query, message)
        
        return {
            "greeting": message,
            "status": "ready_to_proceed",
            "type": "greeting"
        }
    
    async def handle_capabilities(self, context: ExecutionContext, intent: IntentAnalysisResult) -> Dict[str, Any]:
        """Handle capabilities request"""
        # Generate capabilities response based on available agents
        agents = fetch_agents_and_tools_from_registry()
        capabilities = []
        
        for agent_id, agent in agents.items():
            if agent.get("agent_type") == "data_agent":
                capabilities.append(f"📊 Query {agent.get('name', agent_id)} database for business insights")
            elif agent.get("agent_type") == "application":
                capabilities.append(f"🔧 {agent.get('name', agent_id)} application operations")
        
        # Limit to top 5 capabilities
        capabilities = capabilities[:5]
        
        capabilities_text = "Here's what I can help you with:\\n" + "\\n".join(capabilities)
        if len(capabilities) >= 5:
            capabilities_text += "\\n\\nAnd more! Just ask me about specific data or tasks you need help with."
        
        workflow_streamer.emit_workflow_completed(
            context.workflow_id, context.session_id,
            final_answer=capabilities_text,
            execution_time=0.5
        )
        
        self.conversation_manager.save_turn(context.session_id, context.user_query, capabilities_text)
        
        return {
            "greeting": capabilities_text,
            "status": "ready_to_proceed",
            "type": "capabilities",
            "available_capabilities": capabilities
        }
    
    async def handle_clarification(self, context: ExecutionContext, intent: IntentAnalysisResult) -> Dict[str, Any]:
        """Handle clarification needed"""
        message = intent.message
        clarification_options = intent.clarification_options
        
        if clarification_options:
            message += "\\n\\nAvailable options:"
            for i, option in enumerate(clarification_options, 1):
                agent_name = option.get("name", "Unknown Agent")
                description = option.get("description", "No description available")
                best_for = option.get("best_for", "")
                
                message += f"\\n{i}. **{agent_name}**"
                message += f"\\n   - {description}"
                if best_for:
                    message += f"\\n   - Best for: {best_for}"
                message += "\\n"
            
            message += "\\nPlease let me know which option you'd prefer, or provide more specific details about what you're looking for."
        
        workflow_streamer.emit_workflow_completed(
            context.workflow_id, context.session_id,
            final_answer=message,
            execution_time=0.5
        )
        
        # Save turn with available options for the next conversation turn
        self.conversation_manager.save_turn(
            context.session_id, 
            context.user_query, 
            message, 
            execution_results=None, 
            clarification_options=clarification_options
        )
        
        return {
            "greeting": message,
            "status": "need_more_info",
            "clarification_needed": [f"Use {opt.get('name')}" for opt in clarification_options],
            "type": "clarification_needed",
            "clarification_options": clarification_options
        }
    
    async def handle_execution_complete(self, context: ExecutionContext, intent: IntentAnalysisResult, 
                                      results: List[AgentResult], strategy: str) -> Dict[str, Any]:
        """Handle completed execution"""
        
        # Generate final response
        final_response = await self._generate_final_response(context, results)
        
        workflow_streamer.emit_workflow_completed(
            context.workflow_id, context.session_id,
            final_answer=final_response,
            execution_time=2.0
        )
        
        # Convert AgentResult objects to dicts for storage
        execution_results = []
        for result in results:
            execution_results.append({
                "agent_id": result.agent_id,
                "agent_name": result.agent_name,
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "row_count": result.row_count,
                "query": result.query,
                "metadata": result.metadata,
                "visualization": result.visualization
            })
        
        self.conversation_manager.save_turn(context.session_id, context.user_query, final_response, execution_results)
        
        # Count successful agents
        successful_agents = [r for r in results if r.success]
        
        # Create execution steps for the response
        execution_steps = []
        for i, result in enumerate(results):
            execution_steps.append({
                "step_id": f"step{i+1}",
                "agent": result.agent_name,
                "action": "query",
                "input": {
                    "query": context.user_query,
                    "agent_id": result.agent_id
                },
                "result": {
                    "success": result.success,
                    "row_count": result.row_count,
                    "error": result.error
                }
            })
        
        return {
            "greeting": final_response,
            "status": "ready_to_proceed",
            "type": "execution_complete",
            "plan": {
                "summary": f"Executed {strategy} strategy with {len(results)} step(s) to answer your query",
                "execution_type": strategy,
                "steps": execution_steps
            },
            "results": execution_results,
            "agents_used": [{"agent_id": r.agent_id, "agent_name": r.agent_name} for r in successful_agents],
            "total_data_points": sum(r.row_count for r in successful_agents if r.row_count)
        }
    
    async def handle_general(self, context: ExecutionContext, intent: IntentAnalysisResult) -> Dict[str, Any]:
        """Handle general responses"""
        message = intent.message
        
        workflow_streamer.emit_workflow_completed(
            context.workflow_id, context.session_id,
            final_answer=message,
            execution_time=0.5
        )
        
        self.conversation_manager.save_turn(context.session_id, context.user_query, message)
        
        return {
            "greeting": message,
            "status": "outside_capability",
            "capability_reason": "This request falls outside my current capabilities. I can help with data queries and agent operations.",
            "type": "general"
        }
    
    async def _generate_final_response(self, context: ExecutionContext, results: List[AgentResult]) -> str:
        """Generate final response from all results"""
        
        # Separate application and data agents
        application_results = [r for r in results if r.success and r.metadata and r.metadata.get("execution_type") == "application"]
        data_results = [r for r in results if r.success and r.metadata and r.metadata.get("execution_type") != "application"]
        
        final_responses = []
        
        # Handle application agent results
        for result in application_results:
            agent_name = result.agent_name
            raw_data = result.data
            
            # Use LLM to generate user-friendly response from raw API data
            prompt = f"""
You are generating a final response for a user query that was executed through an application API.

Agent: {agent_name}
Original User Query: "{context.user_query}"
Raw API Response: {raw_data}

INSTRUCTIONS:
Generate a helpful, conversational response based on the API execution results.

GUIDELINES:
- Be direct and answer the user's specific question
- Include the relevant data from the API response
- Be conversational and helpful
- Don't expose technical API details unless relevant
- If it's weather data, mention the conditions and temperature clearly
- If it's user/order data, summarize the key information
"""
            
            response = self.llm_client.invoke_with_text_response(prompt, task_type=TaskType.FINAL_ANSWER)
            if response:
                final_responses.append(response.strip())
            else:
                # Fallback
                final_responses.append(f"I got this information from {agent_name}: {str(raw_data)[:200]}")
        
        # Handle data agent results (with full data for analysis)
        if data_results:
            results_text = ""
            total_rows = 0
            full_data_for_analysis = []
            
            for result in data_results:
                row_count = result.row_count or 0
                total_rows += row_count
                
                results_text += f"\\n{result.agent_name}: {row_count} records"
                
                # Include actual data for analysis (especially for trends and charts)
                data = result.data
                if isinstance(data, list) and data:
                    # For small datasets (like token trends), include all data
                    if len(data) <= 50:
                        full_data_for_analysis.extend(data)
                        results_text += f"\\nFull dataset for {result.agent_name}:\\n"
                        for i, record in enumerate(data):  # Show up to 20 records in text
                            if isinstance(record, dict):
                                record_str = ", ".join([f"{k}: {v}" for k, v in record.items()])
                                results_text += f"  {i+1}. {record_str}\\n"
                    else:
                        # For larger datasets, include sample + summary stats
                        full_data_for_analysis.extend(data)  # Include first 20 for context
                        first_record = data[0] if data else {}
                        if isinstance(first_record, dict):
                            sample_fields = []
                            for key, value in list(first_record.items())[:5]:
                                sample_fields.append(f"{key}: {str(value)[:50]}")
                            results_text += f"\\nSample fields: {'; '.join(sample_fields)}\\n"
            
            # Create comprehensive prompt with actual data
            data_context = ""
            if full_data_for_analysis:
                data_context = f"\\nACTUAL DATA FOR ANALYSIS:\\n{json.dumps(full_data_for_analysis, indent=2, default=str)}"
            
            prompt = f"""
Generate a helpful response for this user query: "{context.user_query}"

CURRENT RESULTS:
{results_text}
Total records: {total_rows}
{data_context}

RESPONSE GUIDELINES:
1. Directly answer their specific question using the actual data
2. For trend analysis, look at the patterns in the data over time
3. Reference specific data points, dates, and values from the dataset
4. Provide business insights and implications based on the trends
5. If it's a "show me" request, describe what the data reveals
6. Keep it conversational and helpful
7. For token generation trends, analyze daily patterns and total usage
"""
            
            response = self.llm_client.invoke_with_text_response(prompt, task_type=TaskType.FINAL_ANSWER)
            if response:
                final_responses.append(response.strip())
        
        # Combine all responses
        if final_responses:
            return " ".join(final_responses)
        else:
            return "I was unable to process your request."
