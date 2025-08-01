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
            message += "\n\nAvailable options:"
            for i, option in enumerate(clarification_options, 1):
                agent_name = option.get("name", "Unknown Agent")
                description = option.get("description", "No description available")
                best_for = option.get("best_for", "")
                
                message += f"\n{i}. **{agent_name}**"
                message += f"\n   - {description}"
                if best_for:
                    message += f"\n   - Best for: {best_for}"
                message += "\n"
            
            message += "\nPlease let me know which option you'd prefer, or provide more specific details about what you're looking for."
        
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
            "available_agents": clarification_options,  # Structured data for better UI rendering
            "user_interface": {
                "type": "agent_selection",
                "options": [
                    {
                        "id": i,
                        "name": opt.get("name", "Unknown Agent"),
                        "description": opt.get("description", "No description available"),
                        "best_for": opt.get("best_for", ""),
                        "capabilities": opt.get("capabilities", []),
                        "recommendation_confidence": opt.get("confidence", 0.5)
                    }
                    for i, opt in enumerate(clarification_options, 1)
                ]
            },
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
            "total_data_points": sum(r.row_count for r in successful_agents if r.row_count),
            "visualization_ready": any(
                r.get("visualization", {}).get("output_format", []) != ["table"] 
                or "chart" in str(r.get("query", "")).lower() 
                or "trend" in str(r.get("query", "")).lower() 
                for r in execution_results 
                if r.get("success")
            ),
            "data_summary": {
                "total_agents": len(results),
                "successful_agents": len(successful_agents),
                "total_records": sum(r.row_count for r in successful_agents if r.row_count),
                "has_visualizations": any(r.get("visualization") for r in execution_results)
            }
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
        
        # Handle application agent results (API calls)
        for result in application_results:
            agent_name = result.agent_name
            raw_data = result.data
            
            # Enhanced LLM prompt for better application agent responses
            prompt = f"""
You are generating a final response for a user query that was executed through an application API.

Agent Used: {agent_name}
Original User Query: "{context.user_query}"
Raw API Response: {raw_data}

RESPONSE REQUIREMENTS:
1. **Direct Answer**: Address the user's specific question directly
2. **Clear Information**: Extract and present the most relevant data from the API response
3. **User-Friendly Format**: Present information in a conversational, easy-to-understand way
4. **Context Awareness**: Consider what the user was asking for and prioritize that information
5. **Actionable Insights**: If applicable, provide context about what the information means

FORMATTING GUIDELINES:
- For weather data: Include current conditions, temperature, and any relevant alerts
- For user/account data: Summarize key information without exposing sensitive details
- For status/operational data: Explain what the status means and any required actions
- For search results: Present the most relevant findings clearly
- Keep technical jargon to a minimum unless specifically requested

Generate a helpful, informative response:
"""
            
            response = self.llm_client.invoke_with_text_response(prompt, task_type=TaskType.FINAL_ANSWER)
            if response:
                final_responses.append(response.strip())
            else:
                # Enhanced fallback with better formatting
                if isinstance(raw_data, dict):
                    key_info = []
                    for key, value in list(raw_data.items())[:5]:  # Show top 5 key-value pairs
                        key_info.append(f"{key}: {value}")
                    fallback_msg = f"Here's what I found from {agent_name}: {'; '.join(key_info)}"
                else:
                    fallback_msg = f"I got this information from {agent_name}: {str(raw_data)[:200]}"
                final_responses.append(fallback_msg)
        
        # Handle data agent results (with enhanced data analysis)
        if data_results:
            results_text = ""
            total_rows = 0
            full_data_for_analysis = []
            visualization_context = ""
            
            for result in data_results:
                row_count = result.row_count or 0
                total_rows += row_count
                
                results_text += f"\\n{result.agent_name}: {row_count} records"
                
                # Include visualization context for better insights
                viz_spec = result.visualization or {}
                chart_types = viz_spec.get('output_format', [])
                if chart_types and len(chart_types) > 1:  # More than just 'table'
                    chart_info = [fmt for fmt in chart_types if fmt != 'table']
                    if chart_info:
                        visualization_context += f"\\nVisualization recommended: {', '.join(chart_info)}"
                
                # Include actual data for analysis (especially for trends and charts)
                data = result.data
                if isinstance(data, list) and data:
                    # For small datasets, include all data for comprehensive analysis
                    if len(data) <= 100:
                        full_data_for_analysis.extend(data)
                        results_text += f"\\nComplete dataset for {result.agent_name}:\\n"
                        # Show key sample records
                        for i, record in enumerate(data[:10]):  # Show first 10 records in summary
                            if isinstance(record, dict):
                                record_str = ", ".join([f"{k}: {v}" for k, v in record.items()])
                                results_text += f"  {i+1}. {record_str}\\n"
                        if len(data) > 10:
                            results_text += f"  ... and {len(data) - 10} more records\\n"
                    else:
                        # For larger datasets, include sample + summary stats
                        full_data_for_analysis.extend(data[:50])  # Include first 50 for analysis
                        first_record = data[0] if data else {}
                        last_record = data[-1] if len(data) > 1 else {}
                        
                        if isinstance(first_record, dict):
                            sample_fields = []
                            for key, value in list(first_record.items())[:5]:
                                sample_fields.append(f"{key}: {str(value)[:50]}")
                            results_text += f"\\nSample fields (first record): {'; '.join(sample_fields)}\\n"
                            
                            if last_record and isinstance(last_record, dict):
                                last_fields = []
                                for key, value in list(last_record.items())[:5]:
                                    last_fields.append(f"{key}: {str(value)[:50]}")
                                results_text += f"Sample fields (last record): {'; '.join(last_fields)}\\n"
            
            # Create comprehensive prompt with actual data and visualization context
            data_context = ""
            if full_data_for_analysis:
                # Include actual data but limit JSON size
                if len(full_data_for_analysis) <= 20:
                    data_context = f"\\nCOMPLETE DATA FOR ANALYSIS:\\n{json.dumps(full_data_for_analysis, indent=2, default=str)}"
                else:
                    # Include sample for large datasets
                    sample_data = full_data_for_analysis[:20]
                    data_context = f"\\nSAMPLE DATA FOR ANALYSIS (first 20 records):\\n{json.dumps(sample_data, indent=2, default=str)}"
            
            prompt = f"""
Generate a comprehensive, insightful response for this user query: "{context.user_query}"

QUERY RESULTS SUMMARY:
{results_text}
Total records analyzed: {total_rows}
{visualization_context}
{data_context}

RESPONSE REQUIREMENTS:
1. **Direct Answer**: Address their specific question using the actual data
2. **Data Insights**: For trend analysis, identify patterns, growth/decline, peaks, and notable changes
3. **Specific Values**: Reference actual data points, dates, amounts, and percentages from the dataset
4. **Business Context**: Provide actionable insights and implications based on the trends
5. **Visualization Readiness**: If charts are recommended, describe what the user will see in the visualization
6. **Conversational Tone**: Keep it helpful and engaging, not just a data dump
7. **Key Findings**: Highlight the most important discoveries from the data analysis

ANALYSIS GUIDELINES:
- For sales trends: Focus on growth patterns, seasonal effects, peak periods
- For time series: Identify trends, outliers, and significant changes over time
- For comparisons: Highlight top performers, significant differences
- For metrics: Provide context about what the numbers mean for business decisions

Generate a response that gives the user valuable insights they can act upon:
"""
            
            response = self.llm_client.invoke_with_text_response(prompt, task_type=TaskType.FINAL_ANSWER)
            if response:
                final_responses.append(response.strip())
        
        # Combine all responses
        if final_responses:
            return " ".join(final_responses)
        else:
            return "I was unable to process your request."
