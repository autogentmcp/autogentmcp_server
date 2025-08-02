"""
Response handlers for different workflow outcomes
"""

import json
from typing import Dict, List, Any
from ..models import ExecutionContext, IntentAnalysisResult, AgentResult
from ..conversation.manager import ConversationManager
from app.workflows.workflow_streamer import workflow_streamer
from app.llm.multimode import MultiModeLLMClient
from app.registry.client import fetch_agents_and_tools_from_registry

class ResponseHandler:
    """Handles different types of workflow responses"""
    
    def __init__(self, conversation_manager: ConversationManager):
        self.conversation_manager = conversation_manager
        self.llm_client = MultiModeLLMClient()
    
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
            "chart_data_specifications": [
                {
                    "agent_name": r.get("agent_name"),
                    "chart_spec": r.get("visualization", {}).get("chart_spec", {}),
                    "output_formats": r.get("visualization", {}).get("output_format", ["table"]),
                    "data_preview": r.get("data", [])[:5] if r.get("data") else [],  # First 5 records for chart preview
                    "total_records": r.get("row_count", 0)
                }
                for r in execution_results 
                if r.get("success") and r.get("visualization", {}).get("chart_spec")
            ],
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
        data_results = [r for r in results if r.success and r.metadata and r.metadata.get("execution_type") == "data"]
        
        print(f"[ResponseHandler] Processing {len(application_results)} application results and {len(data_results)} data results")
        
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
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm_client.generate_response(messages, task_type="general_chat")
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
            print(f"[ResponseHandler] Processing {len(data_results)} data results for final response")
            results_text = ""
            total_rows = 0
            full_data_for_analysis = []
            visualization_context = ""
            
            for result in data_results:
                row_count = result.row_count or 0
                total_rows += row_count
                
                results_text += f"\\n{result.agent_name}: {row_count} records"
                
                # Include enhanced visualization context for better insights
                viz_spec = result.visualization or {}
                chart_types = viz_spec.get('output_format', [])
                chart_spec = viz_spec.get('chart_spec', {})
                
                if chart_types and len(chart_types) > 1:  # More than just 'table'
                    chart_info = [fmt for fmt in chart_types if fmt != 'table']
                    if chart_info:
                        visualization_context += f"\\nVisualization recommended: {', '.join(chart_info)}"
                        
                        # Add detailed chart specifications
                        if chart_spec:
                            chart_type = chart_spec.get('type', 'chart')
                            x_axis = chart_spec.get('x', 'category')
                            y_axis = chart_spec.get('y', 'value')
                            title = chart_spec.get('title', 'Data Visualization')
                            
                            visualization_context += f"\\nChart Details:"
                            visualization_context += f"\\n  • Type: {chart_type.replace('_', ' ').title()}"
                            visualization_context += f"\\n  • X-axis: {x_axis}"
                            visualization_context += f"\\n  • Y-axis: {y_axis}"
                            visualization_context += f"\\n  • Title: {title}"
                            
                            # Add color scheme if available
                            if chart_spec.get('color_scheme'):
                                visualization_context += f"\\n  • Colors: {chart_spec.get('color_scheme')}"
                
                # Include actual data for analysis (especially for trends and charts)
                data = result.data
                if isinstance(data, list) and data:
                    # For small datasets, include all data for comprehensive analysis
                    if len(data) <= 100:
                        full_data_for_analysis.extend(data)
                        results_text += f"\\nComplete dataset for {result.agent_name}:\\n"
                        
                        # Enhanced data presentation for small datasets
                        if len(data) <= 12:  # For very small datasets like yours, show more detail
                            results_text += "DETAILED RECORDS:\\n"
                            for i, record in enumerate(data, 1):
                                if isinstance(record, dict):
                                    # Show more fields for small datasets
                                    record_fields = []
                                    for key, value in record.items():
                                        if len(record_fields) < 6:  # Show up to 6 fields
                                            formatted_value = str(value)[:40] + ('...' if len(str(value)) > 40 else '')
                                            record_fields.append(f"{key}: {formatted_value}")
                                    results_text += f"  Record {i}: {' | '.join(record_fields)}\\n"
                        else:
                            # Show key sample records for medium datasets
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
Generate a comprehensive, highly detailed response for this user query: "{context.user_query}"

QUERY RESULTS SUMMARY:
{results_text}
Total records analyzed: {total_rows}
{visualization_context}
{data_context}

RESPONSE REQUIREMENTS:
1. **Direct Answer**: Address their specific question using the actual data values, trends, and patterns
2. **Detailed Data Insights**: For trend analysis, identify specific patterns, growth/decline percentages, peaks, valleys, and notable changes
3. **Specific Values & Metrics**: Reference actual data points, dates, amounts, percentages, and calculations from the dataset
4. **Business Context & Implications**: Provide actionable insights and strategic recommendations based on the trends
5. **Chart Data Generation**: If charts are recommended, provide:
   - Detailed description of what the chart will show
   - Key data points that will be highlighted
   - Trends and patterns the visualization will reveal
   - Specific chart configuration details (axes, scales, colors)
6. **Conversational & Comprehensive**: Keep it engaging while being thoroughly analytical
7. **Key Findings & Recommendations**: Highlight the most important discoveries and next steps

ENHANCED ANALYSIS GUIDELINES:
- **For Small Datasets (1-15 records)**: Provide detailed analysis of each record, identify patterns, outliers, and specific insights
- **For Sales/Revenue Trends**: Calculate growth rates, identify seasonal patterns, highlight peak/low periods with specific values
- **For Time Series Data**: Identify trend direction, calculate rates of change, spot outliers and inflection points
- **For Comparisons**: Rank top performers, calculate percentage differences, identify significant gaps
- **For Performance Metrics**: Provide context about targets, benchmarks, and what the numbers mean for business decisions
- **For Categorical Data**: Show distributions, identify dominant categories, highlight interesting patterns
- **For Single Records**: Explain the significance of individual data points and their business context

CHART DATA ENHANCEMENT:
- If visualization is recommended, describe exactly what users will see in the chart
- Mention specific data points that will be plotted
- Explain how the visualization helps understand the data better
- Suggest additional chart types that could provide complementary insights

RESPONSE STRUCTURE:
1. **Executive Summary**: One clear sentence answering the main question with key findings
2. **Key Metrics**: 3-5 most important numbers with context and business significance
3. **Detailed Analysis**: Deep dive into trends, patterns, and insights from the actual data
4. **Individual Record Insights**: For small datasets, highlight notable individual records and their significance
5. **Visual Insights**: If charts are available, explain exactly what they reveal and why it matters
6. **Business Implications**: What this means for decision-making and strategic planning
7. **Recommended Actions**: Specific next steps or follow-up questions to drive action

SPECIAL HANDLING FOR DATASET SIZES:
- **1-5 records**: Analyze each record individually with detailed business context
- **6-15 records**: Highlight key patterns, outliers, and top/bottom performers with specific examples
- **16+ records**: Focus on aggregate patterns, trends, and statistical insights

Generate a response that gives the user valuable, actionable insights they can immediately act upon:
"""
            
            print(f"[ResponseHandler] Making final LLM call for data analysis with {total_rows} total records")
            print(f"[ResponseHandler] Prompt preview: {prompt[:200]}...")
            
            try:
                messages = [{"role": "user", "content": prompt}]
                response = await self.llm_client.generate_response(messages, task_type="general_chat")
                
                print(f"[ResponseHandler] LLM response received: {bool(response)}")
                if response:
                    print(f"[ResponseHandler] Response preview: {response[:200]}...")
                    final_responses.append(response.strip())
                else:
                    print(f"[ResponseHandler] LLM response was None, using enhanced fallback analysis")
                    fallback_response = await self._generate_enhanced_fallback_analysis(context, data_results, total_rows, full_data_for_analysis, visualization_context)
                    final_responses.append(fallback_response)
            except Exception as llm_error:
                print(f"[ResponseHandler] ERROR during LLM call: {llm_error}")
                print(f"[ResponseHandler] Generating enhanced fallback analysis instead")
                fallback_response = await self._generate_enhanced_fallback_analysis(context, data_results, total_rows, full_data_for_analysis, visualization_context)
                final_responses.append(fallback_response)
        
        # Combine all responses
        print(f"[ResponseHandler] Combining {len(final_responses)} final responses")
        if final_responses:
            combined_response = " ".join(final_responses)
            print(f"[ResponseHandler] Final combined response length: {len(combined_response)} characters")
            return combined_response
        else:
            print(f"[ResponseHandler] No final responses generated, returning default message")
            return "I was unable to process your request."
    
    async def _generate_enhanced_fallback_analysis(self, context: ExecutionContext, data_results: List[AgentResult], 
                                                 total_rows: int, full_data_for_analysis: List[Dict[str, Any]], 
                                                 visualization_context: str) -> str:
        """Generate detailed analysis even when main LLM call fails"""
        
        print(f"[ResponseHandler] Generating enhanced fallback analysis for {total_rows} records")
        
        # Start with executive summary
        response_parts = []
        response_parts.append(f"**Data Analysis Summary**")
        response_parts.append(f"I successfully retrieved and analyzed {total_rows} records from your query: \"{context.user_query}\"")
        
        # Analyze each data result
        for i, result in enumerate(data_results):
            agent_name = result.agent_name
            row_count = result.row_count or 0
            data = result.data
            
            response_parts.append(f"\n**📊 {agent_name} Analysis:**")
            response_parts.append(f"• Records Retrieved: {row_count}")
            
            if isinstance(data, list) and data and len(data) > 0:
                # Analyze data structure
                first_record = data[0] if data else {}
                if isinstance(first_record, dict):
                    columns = list(first_record.keys())
                    response_parts.append(f"• Data Fields: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
                    
                    # Provide sample data insights
                    if row_count <= 5:
                        response_parts.append(f"\n**📋 Complete Dataset:**")
                        for idx, record in enumerate(data, 1):
                            if isinstance(record, dict):
                                record_summary = []
                                for key, value in list(record.items())[:4]:  # Show first 4 fields
                                    record_summary.append(f"{key}: {str(value)[:30]}{'...' if len(str(value)) > 30 else ''}")
                                response_parts.append(f"  {idx}. {' | '.join(record_summary)}")
                    else:
                        response_parts.append(f"\n**📋 Sample Records:**")
                        for idx, record in enumerate(data[:3], 1):  # Show first 3 records
                            if isinstance(record, dict):
                                record_summary = []
                                for key, value in list(record.items())[:4]:
                                    record_summary.append(f"{key}: {str(value)[:30]}{'...' if len(str(value)) > 30 else ''}")
                                response_parts.append(f"  {idx}. {' | '.join(record_summary)}")
                        if row_count > 3:
                            response_parts.append(f"  ... and {row_count - 3} more records")
                    
                    # Add data insights
                    insights = self._extract_data_insights(data, context.user_query)
                    if insights:
                        response_parts.append(f"\n**🔍 Key Insights:**")
                        response_parts.extend([f"• {insight}" for insight in insights])
            
            # Add visualization information
            viz_spec = result.visualization or {}
            chart_types = viz_spec.get('output_format', [])
            chart_spec = viz_spec.get('chart_spec', {})
            
            if chart_types and len(chart_types) > 1:
                chart_info = [fmt for fmt in chart_types if fmt != 'table']
                if chart_info:
                    response_parts.append(f"\n**📈 Visualization Ready:**")
                    response_parts.append(f"• Chart Type: {', '.join(chart_info).replace('_', ' ').title()}")
                    
                    if chart_spec:
                        chart_type = chart_spec.get('type', 'chart')
                        x_axis = chart_spec.get('x', 'category')
                        y_axis = chart_spec.get('y', 'value')
                        title = chart_spec.get('title', 'Data Visualization')
                        
                        response_parts.append(f"• Chart: {chart_type.replace('_', ' ').title()}")
                        response_parts.append(f"• X-axis: {x_axis} | Y-axis: {y_axis}")
                        response_parts.append(f"• Title: {title}")
                        
                        # Describe what the chart will show
                        chart_description = self._describe_chart_content(data, chart_spec, context.user_query)
                        if chart_description:
                            response_parts.append(f"• Chart shows: {chart_description}")
        
        # Add business context and recommendations
        business_context = self._generate_business_context(context.user_query, total_rows, data_results)
        if business_context:
            response_parts.append(f"\n**💼 Business Implications:**")
            response_parts.extend([f"• {point}" for point in business_context])
        
        # Add next steps
        next_steps = self._suggest_next_steps(context.user_query, data_results)
        if next_steps:
            response_parts.append(f"\n**🎯 Recommended Next Steps:**")
            response_parts.extend([f"• {step}" for step in next_steps])
        
        return "\n".join(response_parts)
    
    def _extract_data_insights(self, data: List[Dict[str, Any]], query: str) -> List[str]:
        """Extract deep business insights and identify concerns from the data"""
        insights = []
        
        if not data or not isinstance(data, list) or len(data) == 0:
            return insights
        
        try:
            first_record = data[0] if data else {}
            if not isinstance(first_record, dict):
                return insights
            
            columns = list(first_record.keys())
            
            # Advanced time series analysis for sales data
            if len(data) > 2 and any(col.lower() in ['month', 'date', 'time'] for col in columns):
                insights.extend(self._analyze_time_series_trends(data))
            
            # Analyze data patterns and variations
            if len(data) > 1:
                # Look for numeric columns for advanced statistical insights
                numeric_columns = []
                for col in columns:
                    try:
                        values = [float(record.get(col, 0)) for record in data if record.get(col) is not None and str(record.get(col)).replace('.', '').replace('-', '').isdigit()]
                        if values and len(values) > 1:
                            numeric_columns.append((col, values))
                    except (ValueError, TypeError):
                        continue
                
                # Generate advanced insights for numeric columns
                for col_name, values in numeric_columns[:3]:  # Analyze up to 3 columns
                    if values:
                        insights.extend(self._analyze_numeric_column(col_name, values, data))
            
            # Look for categorical patterns and distributions
            categorical_columns = [col for col in columns if col.lower() in ['category', 'type', 'department', 'region', 'status', 'product', 'customer']]
            for col in categorical_columns[:2]:  # Analyze first 2 categorical columns
                insights.extend(self._analyze_categorical_column(col, data))
            
            # Business-specific insights based on query context
            insights.extend(self._generate_business_specific_insights(data, query))
            
        except Exception as e:
            print(f"[ResponseHandler] Error extracting insights: {e}")
        
        return insights
    
    def _analyze_time_series_trends(self, data: List[Dict[str, Any]]) -> List[str]:
        """Analyze time series data for trends, seasonality, and concerns"""
        insights = []
        
        try:
            # Find date and value columns
            date_col = None
            value_col = None
            
            for col in data[0].keys():
                if col.lower() in ['month', 'date', 'time', 'period']:
                    date_col = col
                elif any(word in col.lower() for word in ['sales', 'revenue', 'amount', 'value', 'total']):
                    value_col = col
            
            if not date_col or not value_col:
                return insights
            
            # Extract values and analyze trends
            values = []
            for record in data:
                try:
                    val = float(record.get(value_col, 0))
                    values.append(val)
                except:
                    continue
            
            if len(values) < 3:
                return insights
            
            # Calculate trend metrics
            total_change = values[-1] - values[0]
            total_change_pct = (total_change / values[0]) * 100 if values[0] != 0 else 0
            
            # Month-over-month analysis
            mom_changes = []
            for i in range(1, len(values)):
                if values[i-1] != 0:
                    mom_change = ((values[i] - values[i-1]) / values[i-1]) * 100
                    mom_changes.append(mom_change)
            
            avg_mom_change = sum(mom_changes) / len(mom_changes) if mom_changes else 0
            
            # Identify trend direction and concerns
            if total_change_pct > 15:
                insights.append(f"🟢 **Strong Growth**: {value_col.replace('_', ' ').title()} increased {total_change_pct:.1f}% over the period (${total_change:,.0f})")
            elif total_change_pct > 5:
                insights.append(f"🟡 **Moderate Growth**: {value_col.replace('_', ' ').title()} grew {total_change_pct:.1f}% (${total_change:,.0f})")
            elif total_change_pct < -10:
                insights.append(f"🔴 **DECLINING TREND**: {value_col.replace('_', ' ').title()} dropped {abs(total_change_pct):.1f}% (${abs(total_change):,.0f}) - NEEDS ATTENTION")
            elif total_change_pct < -5:
                insights.append(f"🟠 **Warning**: {value_col.replace('_', ' ').title()} declined {abs(total_change_pct):.1f}% - Monitor closely")
            else:
                insights.append(f"⚪ **Flat Performance**: {value_col.replace('_', ' ').title()} relatively stable ({total_change_pct:.1f}% change)")
            
            # Volatility analysis
            if len(mom_changes) > 2:
                volatility = sum(abs(change) for change in mom_changes) / len(mom_changes)
                if volatility > 15:
                    insights.append(f"⚠️ **HIGH VOLATILITY**: Average month-to-month variation of {volatility:.1f}% indicates unstable performance")
                elif volatility > 8:
                    insights.append(f"🟡 **Moderate Volatility**: {volatility:.1f}% average monthly variation - some instability")
                
                # Identify best and worst performing months
                if len(values) >= 6:
                    max_val = max(values)
                    min_val = min(values)
                    max_idx = values.index(max_val)
                    min_idx = values.index(min_val)
                    
                    insights.append(f"📈 **Peak Performance**: Month {max_idx + 1} achieved highest value (${max_val:,.0f})")
                    insights.append(f"📉 **Lowest Point**: Month {min_idx + 1} recorded minimum value (${min_val:,.0f})")
                    
                    # Recent performance vs peak
                    recent_vs_peak = ((values[-1] - max_val) / max_val) * 100
                    if recent_vs_peak < -20:
                        insights.append(f"🚨 **CRITICAL**: Current performance is {abs(recent_vs_peak):.1f}% below peak - urgent review needed")
                    elif recent_vs_peak < -10:
                        insights.append(f"⚠️ **Underperforming**: Current value {abs(recent_vs_peak):.1f}% below peak performance")
            
            # Seasonal patterns (if 12 months of data)
            if len(values) == 12:
                q1 = sum(values[0:3]) / 3
                q2 = sum(values[3:6]) / 3
                q3 = sum(values[6:9]) / 3
                q4 = sum(values[9:12]) / 3
                
                quarters = [("Q1", q1), ("Q2", q2), ("Q3", q3), ("Q4", q4)]
                best_q = max(quarters, key=lambda x: x[1])
                worst_q = min(quarters, key=lambda x: x[1])
                
                insights.append(f"🏆 **Best Quarter**: {best_q[0]} averaged ${best_q[1]:,.0f}")
                insights.append(f"📊 **Seasonal Pattern**: {worst_q[0]} was weakest quarter (${worst_q[1]:,.0f}) - {((best_q[1] - worst_q[1])/worst_q[1]*100):.1f}% difference")
            
        except Exception as e:
            print(f"[ResponseHandler] Error in time series analysis: {e}")
        
        return insights
    
    def _analyze_numeric_column(self, col_name: str, values: List[float], data: List[Dict[str, Any]]) -> List[str]:
        """Analyze numeric column for statistical insights and outliers"""
        insights = []
        
        try:
            if not values or len(values) < 2:
                return insights
            
            total = sum(values)
            avg = total / len(values)
            min_val = min(values)
            max_val = max(values)
            
            # Standard deviation and coefficient of variation
            variance = sum((x - avg) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            cv = (std_dev / avg) * 100 if avg != 0 else 0
            
            # Basic statistics with business context
            insights.append(f"📊 **{col_name.replace('_', ' ').title()} Range**: ${min_val:,.0f} - ${max_val:,.0f} (Spread: ${max_val - min_val:,.0f})")
            
            # Outlier detection
            if max_val > avg + (2 * std_dev):
                outlier_factor = max_val / avg
                insights.append(f"🎯 **Outlier Detected**: Peak value is {outlier_factor:.1f}x above average - investigate this exceptional performance")
            
            if min_val < avg - (2 * std_dev):
                insights.append(f"⚠️ **Low Outlier**: Minimum value significantly below average - potential problem area")
            
            # Consistency analysis
            if cv > 30:
                insights.append(f"🔄 **High Variability**: Coefficient of variation is {cv:.1f}% - inconsistent performance needs attention")
            elif cv < 10:
                insights.append(f"✅ **Consistent Performance**: Low variability ({cv:.1f}%) indicates stable operations")
            
            # Concentration analysis
            top_20_percent = int(len(values) * 0.2) or 1
            sorted_values = sorted(values, reverse=True)
            top_contribution = sum(sorted_values[:top_20_percent])
            concentration = (top_contribution / total) * 100
            
            if concentration > 60:
                insights.append(f"⚡ **High Concentration**: Top {top_20_percent} periods contribute {concentration:.1f}% of total - dependency risk")
            
        except Exception as e:
            print(f"[ResponseHandler] Error in numeric analysis: {e}")
        
        return insights
    
    def _analyze_categorical_column(self, col_name: str, data: List[Dict[str, Any]]) -> List[str]:
        """Analyze categorical data for distribution and patterns"""
        insights = []
        
        try:
            categories = {}
            for record in data:
                value = record.get(col_name)
                if value:
                    categories[value] = categories.get(value, 0) + 1
            
            if not categories:
                return insights
            
            total_records = len(data)
            sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            
            # Dominant category analysis
            top_category = sorted_categories[0]
            top_percentage = (top_category[1] / total_records) * 100
            
            if top_percentage > 70:
                insights.append(f"🎯 **Heavy Concentration**: {top_category[0]} dominates with {top_percentage:.1f}% ({top_category[1]} records)")
            elif top_percentage > 50:
                insights.append(f"📊 **Market Leader**: {top_category[0]} holds {top_percentage:.1f}% share ({top_category[1]} records)")
            
            # Distribution analysis
            if len(categories) > 3:
                top_3_share = sum(cat[1] for cat in sorted_categories[:3])
                top_3_percentage = (top_3_share / total_records) * 100
                insights.append(f"🏆 **Top 3 {col_name.replace('_', ' ').title()}**: Control {top_3_percentage:.1f}% of records")
            
        except Exception as e:
            print(f"[ResponseHandler] Error in categorical analysis: {e}")
        
        return insights
    
    def _generate_business_specific_insights(self, data: List[Dict[str, Any]], query: str) -> List[str]:
        """Generate insights specific to business context and query"""
        insights = []
        query_lower = query.lower()
        
        try:
            # E-commerce/Retail specific insights
            if any(word in query_lower for word in ['ecommerce', 'retail', 'sales', 'revenue']):
                # Look for sales-specific patterns
                if len(data) >= 6:
                    insights.append("💡 **Actionable Intelligence**: With 6+ months of data, you can identify seasonal trends and plan inventory/marketing accordingly")
                
                # Revenue threshold analysis
                sales_values = []
                for record in data:
                    for key, value in record.items():
                        if any(word in key.lower() for word in ['sales', 'revenue', 'amount']) and isinstance(value, (int, float)):
                            sales_values.append(value)
                            break
                
                if sales_values:
                    avg_sales = sum(sales_values) / len(sales_values)
                    if avg_sales > 50000000:  # 50M threshold
                        insights.append("🚀 **Scale Indicator**: Operating at >$50M monthly scale - enterprise-level performance")
                    
                    # Growth sustainability check
                    if len(sales_values) >= 3:
                        recent_trend = sales_values[-1] - sales_values[-3]
                        if recent_trend < 0:
                            insights.append("🚨 **Immediate Concern**: Recent 2-month decline needs urgent investigation and corrective action")
                        elif recent_trend > avg_sales * 0.1:
                            insights.append("⚡ **Growth Momentum**: Strong recent growth - capitalize with increased investment")
            
            # Customer/User analysis
            elif any(word in query_lower for word in ['customer', 'user', 'client']):
                insights.append("👥 **Customer Intelligence**: Analyze customer segments and retention patterns for targeted strategies")
            
            # Performance/KPI analysis
            elif any(word in query_lower for word in ['performance', 'kpi', 'metrics']):
                insights.append("📈 **Performance Optimization**: Benchmark against industry standards and set realistic targets")
            
            # General business insights
            insights.append("💼 **Strategic Recommendation**: Use this data to build predictive models and early warning systems")
            
        except Exception as e:
            print(f"[ResponseHandler] Error in business insights: {e}")
        
        return insights
    
    def _describe_chart_content(self, data: List[Dict[str, Any]], chart_spec: Dict[str, Any], query: str) -> str:
        """Describe what the chart will display"""
        if not data or not chart_spec:
            return ""
        
        try:
            chart_type = chart_spec.get('type', 'chart')
            x_column = chart_spec.get('x', '')
            y_column = chart_spec.get('y', '')
            
            if not x_column or not y_column:
                return f"Data visualization of {len(data)} records"
            
            # Analyze the data to describe chart content
            if chart_type == 'line_chart':
                return f"Trend line showing {y_column.replace('_', ' ')} over {x_column.replace('_', ' ')} for {len(data)} data points"
            elif chart_type == 'bar_chart':
                return f"Bar comparison of {y_column.replace('_', ' ')} across {len(data)} different {x_column.replace('_', ' ')} categories"
            elif chart_type == 'pie_chart':
                return f"Distribution breakdown showing percentage share of {y_column.replace('_', ' ')} by {x_column.replace('_', ' ')}"
            elif chart_type == 'metric':
                # Get the total value if possible
                try:
                    total_value = sum(float(record.get(y_column, 0)) for record in data)
                    return f"Total {y_column.replace('_', ' ')}: ${total_value:,.0f}"
                except:
                    return f"Key metric display for {y_column.replace('_', ' ')}"
            else:
                return f"{chart_type.replace('_', ' ').title()} visualization of {len(data)} data points"
                
        except Exception as e:
            print(f"[ResponseHandler] Error describing chart: {e}")
            return f"Visualization of {len(data)} data records"
    
    def _generate_business_context(self, query: str, total_rows: int, data_results: List[AgentResult]) -> List[str]:
        """Generate strategic business context and critical implications"""
        context_points = []
        
        query_lower = query.lower()
        
        # Analyze data patterns for business context
        has_time_series = any(
            result.data and len(result.data) > 3 and 
            any(col.lower() in ['month', 'date', 'time'] for col in (result.data[0].keys() if result.data else []))
            for result in data_results
        )
        
        # Sales/Revenue context with urgency indicators
        if any(word in query_lower for word in ['sales', 'revenue', 'profit', 'income']):
            context_points.append(f"💰 **Revenue Intelligence**: {total_rows} data points provide sufficient granularity for strategic decision-making")
            
            if has_time_series and total_rows >= 12:
                context_points.append("📊 **Forecasting Ready**: Full year+ of data enables reliable quarterly and annual projections")
                context_points.append("⚠️ **Monitoring Critical**: Any month-over-month decline >10% requires immediate investigation")
            elif total_rows >= 6:
                context_points.append("🎯 **Trend Identification**: 6+ months enables seasonal pattern recognition and tactical adjustments")
            
            context_points.append("🚨 **Action Threshold**: Revenue volatility >15% indicates operational instability requiring intervention")
            
        # Customer context with retention insights
        elif any(word in query_lower for word in ['customer', 'client', 'user']):
            context_points.append(f"👥 **Customer Intelligence**: {total_rows} records enable segmentation and retention strategy optimization")
            context_points.append("💡 **Retention Priority**: Focus on top 20% customers who likely drive 60-80% of revenue")
            
        # Performance context with benchmarking
        elif any(word in query_lower for word in ['performance', 'metrics', 'kpi']):
            context_points.append(f"📈 **Performance Baseline**: {total_rows} metrics establish benchmarks for goal setting and improvement tracking")
            context_points.append("🎯 **Target Setting**: Use 75th percentile as stretch goals, 90th percentile as exceptional targets")
            
        # Trend context with predictive insights
        elif any(word in query_lower for word in ['trend', 'growth', 'change', 'over time']):
            context_points.append(f"📊 **Trend Analysis**: {total_rows} time periods enable robust pattern identification and forecasting")
            if total_rows >= 12:
                context_points.append("🔮 **Predictive Capability**: 12+ periods support machine learning models for 3-6 month forecasts")
            
        # E-commerce specific context
        elif any(word in query_lower for word in ['ecommerce', 'retail', 'online']):
            context_points.append("🛒 **E-commerce Intelligence**: Digital retail patterns enable rapid optimization and A/B testing")
            context_points.append("⚡ **Quick Wins**: Focus on high-impact, low-effort improvements in top-performing categories")
            
        # General strategic context
        else:
            context_points.append(f"📊 **Data-Driven Foundation**: {total_rows} records provide statistical significance for confident decision-making")
        
        # Add urgency and priority context
        for result in data_results:
            if result.data and len(result.data) > 0:
                # Check for declining trends that need attention
                numeric_cols = []
                for record in result.data[:3]:  # Check first 3 records
                    if isinstance(record, dict):
                        for key, value in record.items():
                            if isinstance(value, (int, float)) and value > 1000:  # Likely monetary
                                numeric_cols.append(key)
                                break
                
                if numeric_cols and len(result.data) >= 3:
                    context_points.append("⏰ **Time-Sensitive**: Quarterly reviews recommended to catch negative trends early")
                    break
        
        # Add visualization impact
        has_charts = any(
            result.visualization and len(result.visualization.get('output_format', [])) > 1 
            for result in data_results
        )
        
        if has_charts:
            context_points.append("📊 **Executive Communication**: Visual representations essential for board/stakeholder presentations")
            context_points.append("🎯 **Decision Acceleration**: Charts reduce analysis time and improve decision confidence")
        
        return context_points
    
    def _suggest_next_steps(self, query: str, data_results: List[AgentResult]) -> List[str]:
        """Suggest specific, actionable next steps with priority levels"""
        next_steps = []
        
        query_lower = query.lower()
        
        # Immediate actions based on query type
        if any(word in query_lower for word in ['sales', 'revenue', 'profit']):
            next_steps.append("🚨 **URGENT**: Investigate any months showing >10% decline - identify root causes within 48 hours")
            next_steps.append("📊 **This Week**: Segment sales by product/region to identify specific problem areas")
            next_steps.append("🎯 **30-Day Plan**: Develop action plan for underperforming segments with specific targets")
            next_steps.append("📈 **Quarterly Goal**: Establish early warning dashboard for real-time trend monitoring")
            
        elif any(word in query_lower for word in ['trend', 'growth', 'change']):
            next_steps.append("🔍 **Immediate**: Analyze contributing factors to trend changes (marketing, seasonality, competition)")
            next_steps.append("📊 **This Month**: Build predictive model for 3-6 month forecasting")
            next_steps.append("⚠️ **Ongoing**: Set up automated alerts for trend reversals")
            
        elif any(word in query_lower for word in ['customer', 'client', 'user']):
            next_steps.append("👥 **Priority 1**: Identify and secure top 20% revenue-generating customers")
            next_steps.append("📞 **This Week**: Conduct retention interviews with at-risk high-value customers")
            next_steps.append("🎯 **Campaign**: Launch targeted retention campaign for declining segments")
            
        elif any(word in query_lower for word in ['top', 'best', 'highest']):
            next_steps.append("🏆 **Success Analysis**: Document and replicate strategies from top performers")
            next_steps.append("📋 **Best Practices**: Create playbook based on high-performing patterns")
            next_steps.append("🚀 **Scale Strategy**: Apply successful tactics to underperforming areas")
            
        else:
            next_steps.append("📊 **Data Quality**: Validate data accuracy and completeness before major decisions")
            next_steps.append("📈 **Baseline Establishment**: Use current metrics as performance benchmarks")
        
        # Technical and operational next steps
        has_time_series = any(
            result.data and len(result.data) > 3 
            for result in data_results
        )
        
        if has_time_series:
            next_steps.append("🤖 **Automation**: Set up monthly automated reports for trend tracking")
            next_steps.append("📱 **Mobile Dashboard**: Create executive mobile dashboard for real-time access")
        
        # Chart and visualization specific actions
        has_charts = any(
            result.visualization and len(result.visualization.get('output_format', [])) > 1 
            for result in data_results
        )
        
        if has_charts:
            next_steps.append("📊 **Executive Brief**: Prepare chart-based presentation for leadership team")
            next_steps.append("🎯 **Team Alignment**: Share visualizations with department heads for coordinated action")
        
        # Risk management
        total_records = sum(result.row_count or 0 for result in data_results)
        if total_records < 6:
            next_steps.append("⚠️ **Data Limitation**: Collect more historical data for reliable trend analysis")
        elif total_records >= 12:
            next_steps.append("🔮 **Advanced Analytics**: Consider seasonal decomposition and forecasting models")
        
        # Business continuity
        next_steps.append("💾 **Data Backup**: Ensure regular backups and establish data governance protocols")
        next_steps.append("🔄 **Review Cycle**: Schedule monthly data reviews and quarterly strategy adjustments")
        
        return next_steps
