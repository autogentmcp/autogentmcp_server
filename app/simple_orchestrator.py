"""
Simple LLM Orchestrator
One workflow that:
1. Understands user input (greeting/capabilities/clarification/execution)
2. Plans execution (single/sequential/parallel agents)
3. Executes agents
4. Generates final response
All with streaming status updates
"""

import json
import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from app.workflows.workflow_streamer import workflow_streamer
from app.llm import MultiModeLLMClient
from app.registry.client import fetch_agents_and_tools_from_registry, get_enhanced_agent_details_for_llm
from app.database.database_query_executor import DatabaseQueryExecutor
from app.orchestrator.agents.data_agent import DataAgentExecutor
from app.orchestrator.models import ExecutionContext

@dataclass
class ExecutionContext:
    """Simple context for workflow execution"""
    workflow_id: str
    session_id: str
    user_query: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    
class SimpleOrchestrator:
    """Simple orchestrator with one clean workflow"""
    
    def __init__(self):
        self.llm_client = MultiModeLLMClient()
        self.db_executor = DatabaseQueryExecutor()
        self.data_agent_executor = DataAgentExecutor()
        self._conversation_memory = {}  # Enhanced conversation storage
        self._conversation_summaries = {}  # Rolling summaries of older conversations
        
    async def execute_workflow(self, user_query: str, session_id: str = None) -> Dict[str, Any]:
        """One simple workflow that handles everything"""
        if not session_id:
            session_id = str(uuid.uuid4())
            
        workflow_id = str(uuid.uuid4())
        context = ExecutionContext(workflow_id, session_id, user_query)
        
        # Get conversation history
        context.conversation_history = self._get_conversation_history(session_id)
        
        try:
            # Start workflow
            workflow_streamer.emit_workflow_started(
                workflow_id, session_id,
                title="Processing your request",
                description="Analyzing and executing your query",
                steps=4
            )
            
            # Step 1: Understand what user wants
            workflow_streamer.emit_step_started(
                workflow_id, session_id, "analyze",
                "analysis", "🧠 Understanding your request..."
            )
            
            intent = await self._analyze_intent(context)
            
            workflow_streamer.emit_step_completed(
                workflow_id, session_id, "analyze",
                "analysis", 1.0
            )
            
            # Handle different intents
            if intent["type"] == "greeting":
                return await self._handle_greeting(context, intent)
                
            elif intent["type"] == "capabilities":
                return await self._handle_capabilities(context, intent)
                
            elif intent["type"] == "clarification_needed":
                return await self._handle_clarification(context, intent)
                
            elif intent["type"] == "execute":
                return await self._handle_execution(context, intent)
                
            else:
                return await self._handle_general(context, intent)
                
        except Exception as e:
            workflow_streamer.emit_error(workflow_id, session_id, "error", str(e))
            return {
                "status": "error",
                "message": f"I encountered an error: {str(e)}"
            }
    
    async def _analyze_intent(self, context: ExecutionContext) -> Dict[str, Any]:
        """Single LLM call to understand intent and plan execution"""
        agents = fetch_agents_and_tools_from_registry()
        
        # Build enhanced agent summary with clear capability distinctions
        agent_list = []
        for agent_id, agent in agents.items():
            if agent.get("agent_type") in ["data_agent", "application"]:
                agent_info = {
                    "id": agent_id,
                    "name": agent.get("name"),
                    "type": agent.get("agent_type"),
                    "description": agent.get("description", "")
                }
                
                # Add database schema and capabilities for data agents
                if agent.get("agent_type") == "data_agent" and agent.get("vault_key"):
                    schema_summary = self._build_schema_summary(agent)
                    agent_info["database_schema"] = schema_summary
                    agent_info["database_type"] = agent.get("database_type", "unknown")
                    agent_info["connection_type"] = agent.get("connection_type", "unknown")
                    agent_info["capabilities"] = "DATABASE_QUERIES"
                    agent_info["data_types"] = self._extract_data_types_from_schema(agent)
                
                # Add API capabilities for application agents
                elif agent.get("agent_type") == "application":
                    agent_info["capabilities"] = "API_OPERATIONS"
                    agent_info["endpoints"] = self._get_application_endpoints(agent)
                    
                agent_list.append(agent_info)
        
        # Build enhanced conversation context
        history_text = ""
        if context.conversation_history:
            recent_turns = context.conversation_history[-5:]  # Last 5 full turns
            
            if recent_turns:
                history_text = "RECENT CONVERSATION CONTEXT:\n"
                for i, turn in enumerate(recent_turns, 1):
                    history_text += f"\n--- Turn {i} ---\n"
                    history_text += f"User: {turn['query']}\n"
                    history_text += f"Assistant: {turn['response'][:200]}...\n"
                    
                    # Add execution details if available
                    if turn.get('execution_results'):
                        agents_used = [r.get('agent_name', 'Unknown') for r in turn['execution_results'] if r.get('success')]
                        if agents_used:
                            history_text += f"Data Sources: {', '.join(agents_used)}\n"
                            total_records = sum(r.get('row_count', 0) for r in turn['execution_results'] if r.get('success'))
                            if total_records > 0:
                                history_text += f"Records Retrieved: {total_records}\n"
            
            # Add conversation summary if exists
            conversation_summary = self._get_conversation_summary(context.session_id)
            if conversation_summary:
                history_text += f"\nPREVIOUS CONVERSATION SUMMARY:\n{conversation_summary}\n"
        
        prompt = f"""
You are a helpful AI assistant with memory of our conversation. Analyze the user's current request in context.

User Query: "{context.user_query}"

{history_text}

Available Agents:
{json.dumps(agent_list, indent=2)}

CONTEXT AWARENESS:
- Use conversation history to understand follow-up questions and references
- If user refers to "that data", "those results", or "the previous query", relate it to recent conversation
- For follow-up questions, consider building on previous execution results
- Maintain conversation flow and remember what data was previously retrieved

AGENT SELECTION RULES:
- **DATA AGENTS** (capabilities: "DATABASE_QUERIES"): Use for data retrieval, statistics, reports, analytics
  - Have database_schema showing available tables and columns
  - Can execute SQL queries against specific databases
  - Example: "show employee statistics", "get sales data", "analyze inventory"
  
- **APPLICATION AGENTS** (capabilities: "API_OPERATIONS"): Use for actions, updates, external API calls
  - Have endpoints for specific operations
  - Can perform actions like creating, updating, calling external services
  - Example: "update order status", "send notification", "get weather data"

IMPORTANT: For data agents, use the database_schema information to understand what tables and columns are available. Only query columns that actually exist in the schema.

QUERY GUIDELINES:
- Consider the database_type and connection_type for SQL syntax
- SQL Server: Use TOP instead of LIMIT, square brackets for reserved words
- MySQL: Use LIMIT, backticks for reserved words  
- PostgreSQL: Use LIMIT, double quotes for case-sensitive names
- Always limit results to manageable sizes (10-50 rows typically)
- Focus on efficient queries that answer the question with minimal data

Determine the intent and plan:
1. If greeting (hi, hello) → type: "greeting"
2. If asking about capabilities → type: "capabilities"  
3. If vague/unclear → type: "clarification_needed"
4. If clear data/task request → type: "execute"
5. Otherwise → type: "general"

For execution type, plan the agent flow:
- Single agent for simple queries
- Sequential for dependent operations (e.g., follow-up analysis)
- Parallel for independent multi-source queries

When generating queries for data agents, ONLY use columns that exist in the database_schema. If the user asks for something that doesn't exist in the schema, suggest alternatives or explain what's available.

Respond with JSON:
{{
    "type": "greeting|capabilities|clarification_needed|execute|general",
    "message": "your response to the user",
    "execution_plan": {{
        "strategy": "single|sequential|parallel",
        "steps": [
            {{"agent_id": "agent1", "query": "specific query for agent"}},
            {{"agent_id": "agent2", "query": "query", "depends_on": "agent1"}}
        ]
    }},
    "clarification_needed": ["what to ask if unclear"],
    "references_previous_data": false,
    "conversation_context": "brief summary of how this relates to previous conversation"
}}
"""
        
        response = self.llm_client.invoke_with_json_response(prompt)
        return response or {"type": "general", "message": "How can I help you today?"}
    
    async def _handle_greeting(self, context: ExecutionContext, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle greeting intent"""
        workflow_streamer.emit_workflow_completed(
            context.workflow_id, context.session_id,
            final_answer=intent["message"],
            execution_time=0.5
        )
        
        self._save_turn(context.session_id, context.user_query, intent["message"])
        
        return {
            "status": "success",
            "message": intent["message"],
            "final_answer": intent["message"],  # Add final_answer for streaming compatibility
            "type": "greeting"
        }
    
    async def _handle_capabilities(self, context: ExecutionContext, intent: Dict[str, Any]) -> Dict[str, Any]:
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
        
        capabilities_text = "Here's what I can help you with:\n" + "\n".join(capabilities)
        if len(capabilities) >= 5:
            capabilities_text += "\n\nAnd more! Just ask me about specific data or tasks you need help with."
        
        workflow_streamer.emit_workflow_completed(
            context.workflow_id, context.session_id,
            final_answer=capabilities_text,
            execution_time=0.5
        )
        
        self._save_turn(context.session_id, context.user_query, capabilities_text)
        
        return {
            "status": "success",
            "message": capabilities_text,
            "final_answer": capabilities_text,  # Add final_answer for streaming compatibility
            "type": "capabilities"
        }
    
    async def _handle_clarification(self, context: ExecutionContext, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clarification needed"""
        workflow_streamer.emit_workflow_completed(
            context.workflow_id, context.session_id,
            final_answer=intent["message"],
            execution_time=0.5
        )
        
        self._save_turn(context.session_id, context.user_query, intent["message"])
        
        return {
            "status": "waiting_for_input",
            "message": intent["message"],
            "final_answer": intent["message"],  # Add final_answer for streaming compatibility
            "type": "clarification_needed",
            "clarification_needed": intent.get("clarification_needed", [])
        }
    
    async def _handle_execution(self, context: ExecutionContext, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle execution of agents"""
        plan = intent.get("execution_plan", {})
        strategy = plan.get("strategy", "single")
        steps = plan.get("steps", [])
        
        if not steps:
            return await self._handle_general(context, intent)
        
        # Execute based on strategy
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "execute",
            "execution", f"⚡ Executing {strategy} agent plan..."
        )
        
        if strategy == "single":
            results = await self._execute_single(context, steps[0])
        elif strategy == "sequential":
            results = await self._execute_sequential(context, steps)
        elif strategy == "parallel":
            results = await self._execute_parallel(context, steps)
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
        
        final_response = await self._generate_final_response(context, results)
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "respond",
            "response", 1.0
        )
        
        workflow_streamer.emit_workflow_completed(
            context.workflow_id, context.session_id,
            final_answer=final_response,
            execution_time=2.0
        )
        
        self._save_turn(context.session_id, context.user_query, final_response, results)
        
        # Count successful agents
        successful_agents = [r for r in results if r.get("success")]
        
        return {
            "status": "success",
            "message": final_response,
            "final_answer": final_response,  # Add final_answer for streaming compatibility
            "type": "execution_complete",
            "results": results,
            "agents_used": [{"agent_id": r["agent_id"], "agent_name": r.get("agent_name")} for r in successful_agents],
            "total_data_points": sum(r.get("row_count", 0) for r in successful_agents)
        }
    
    async def _handle_general(self, context: ExecutionContext, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general responses"""
        workflow_streamer.emit_workflow_completed(
            context.workflow_id, context.session_id,
            final_answer=intent["message"],
            execution_time=0.5
        )
        
        self._save_turn(context.session_id, context.user_query, intent["message"])
        
        return {
            "status": "success",
            "message": intent["message"],
            "final_answer": intent["message"],  # Add final_answer for streaming compatibility
            "type": "general"
        }
    
    async def _execute_single(self, context: ExecutionContext, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute single agent"""
        result = await self._execute_agent(step["agent_id"], step["query"], context.workflow_id, context.session_id)
        return [result]
    
    async def _execute_sequential(self, context: ExecutionContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute agents sequentially"""
        results = []
        agent_results = {}
        
        for step in steps:
            # Check if this step depends on another
            if "depends_on" in step:
                # Enhance query with previous result
                prev_result = agent_results.get(step["depends_on"])
                if prev_result and prev_result.get("success"):
                    step["query"] = await self._enhance_query_with_context(
                        step["query"], prev_result
                    )
            
            result = await self._execute_agent(step["agent_id"], step["query"], context.workflow_id, context.session_id)
            results.append(result)
            agent_results[step["agent_id"]] = result
            
        return results
    
    async def _execute_parallel(self, context: ExecutionContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute agents in parallel"""
        tasks = []
        for step in steps:
            task = self._execute_agent(step["agent_id"], step["query"], context.workflow_id, context.session_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    "agent_id": steps[i]["agent_id"],
                    "success": False,
                    "error": str(result)
                })
            else:
                final_results.append(result)
                
        return final_results
    
    async def _execute_agent(self, agent_id: str, query: str, workflow_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """Execute a single agent"""
        try:
            agents = fetch_agents_and_tools_from_registry()
            agent = agents.get(agent_id)
            
            if not agent:
                return {"agent_id": agent_id, "success": False, "error": "Agent not found"}
            
            agent_type = agent.get("agent_type")
            
            if agent_type == "data_agent":
                # Create execution context for data agent
                context = ExecutionContext(
                    workflow_id=workflow_id or "simple_workflow",
                    session_id=session_id or "simple_session",
                    workflow_streamer=workflow_streamer
                )
                
                # Use the sophisticated DataAgentExecutor
                result = await self.data_agent_executor.execute_data_agent(agent_id, query, context)
                
                # Convert AgentResult to simple orchestrator format
                if result.success:
                    return {
                        "agent_id": agent_id,
                        "agent_name": result.agent_name,
                        "success": True,
                        "data": result.data,
                        "row_count": result.row_count,
                        "query": result.query,
                        "metadata": result.metadata,
                        "visualization": result.visualization
                    }
                else:
                    return {
                        "agent_id": agent_id,
                        "agent_name": result.agent_name,
                        "success": False,
                        "error": result.error,
                        "metadata": result.metadata,
                        "visualization": result.visualization
                    }
            
            elif agent_type == "application":
                # Simplified application execution
                return {
                    "agent_id": agent_id,
                    "agent_name": agent.get("name"),
                    "success": True,
                    "data": {"message": f"Executed {query} on {agent.get('name')}"}
                }
            
        except Exception as e:
            return {
                "agent_id": agent_id,
                "success": False,
                "error": str(e)
            }
    
    async def _enhance_query_with_context(self, query: str, prev_result: Dict[str, Any]) -> str:
        """Enhance query with context from previous result"""
        prompt = f"""
Enhance this query with context from previous result:

Original Query: "{query}"
Previous Result Summary: {prev_result.get('row_count', 0)} rows from {prev_result.get('agent_name')}

Return enhanced query that uses the context. Be specific.
"""
        
        response = self.llm_client.invoke(prompt)
        return response.content.strip()
    
    async def _generate_final_response(self, context: ExecutionContext, results: List[Dict[str, Any]]) -> str:
        """Generate final response from all results with conversation context"""
        # Prepare results summary
        results_text = ""
        total_rows = 0
        successful_results = []
        
        for result in results:
            if result.get("success"):
                successful_results.append(result)
                agent_name = result.get("agent_name", "Unknown")
                row_count = result.get("row_count", 0)
                total_rows += row_count
                
                results_text += f"\n{agent_name}: {row_count} records"
                
                # Add sample data with more context
                data = result.get("data", [])
                if isinstance(data, list) and data:
                    # Show first record with field names
                    first_record = data[0] if data else {}
                    if isinstance(first_record, dict):
                        sample_fields = []
                        for key, value in list(first_record.items())[:5]:  # Show first 5 fields
                            sample_fields.append(f"{key}: {str(value)[:50]}")
                        results_text += f"\nSample fields: {'; '.join(sample_fields)}\n"
        
        # Get conversation context for better responses
        recent_context = ""
        if context.conversation_history:
            last_turn = context.conversation_history[-1] if context.conversation_history else None
            if last_turn and last_turn.get('agents_used'):
                recent_context = f"\nPrevious query used: {', '.join(last_turn['agents_used'])}"
        
        prompt = f"""
Generate a helpful response for this user query: "{context.user_query}"

CONVERSATION CONTEXT:
- This is part of an ongoing conversation
- User may be asking follow-up questions or drilling down into data
- Consider previous interactions to maintain conversation flow{recent_context}

CURRENT RESULTS:
{results_text}
Total records: {total_rows}

RESPONSE GUIDELINES:
1. Directly answer their specific question
2. Reference the actual data retrieved (numbers, names, values)
3. Provide business insights and implications
4. If this seems like a follow-up question, acknowledge the connection
5. Suggest logical next steps or related questions
6. Keep it conversational and context-aware

Focus on being helpful while showing you understand the conversation flow.
"""
        
        response = self.llm_client.invoke(prompt)
        return response.content.strip()
    
    def _get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for session"""
        return self._conversation_memory.get(session_id, [])
    
    def _save_turn(self, session_id: str, query: str, response: str, execution_results: List[Dict[str, Any]] = None):
        """Save conversation turn with detailed context"""
        if session_id not in self._conversation_memory:
            self._conversation_memory[session_id] = []
        
        # Create rich conversation turn
        turn = {
            "timestamp": time.time(),
            "query": query,
            "response": response,
            "execution_results": execution_results or [],
            "agents_used": [r.get("agent_name") for r in (execution_results or []) if r.get("success")],
            "data_summary": self._create_data_summary(execution_results or [])
        }
        
        self._conversation_memory[session_id].append(turn)
        
        # Keep only last 10 turns in detailed memory
        if len(self._conversation_memory[session_id]) > 10:
            # Move older conversations to summary
            self._update_conversation_summary(session_id)
            self._conversation_memory[session_id] = self._conversation_memory[session_id][-10:]
    
    def _create_data_summary(self, execution_results: List[Dict[str, Any]]) -> str:
        """Create a brief summary of data retrieved"""
        if not execution_results:
            return "No data retrieved"
        
        successful_results = [r for r in execution_results if r.get("success")]
        if not successful_results:
            return "Data retrieval failed"
        
        total_records = sum(r.get("row_count", 0) for r in successful_results)
        agents = [r.get("agent_name", "Unknown") for r in successful_results]
        
        summary = f"{total_records} records from {', '.join(agents)}"
        
        # Add sample data context if available
        for result in successful_results:
            data = result.get("data", [])
            if data and isinstance(data, list) and len(data) > 0:
                first_record = data[0]
                if isinstance(first_record, dict):
                    key_fields = list(first_record.keys())[:3]  # First 3 fields
                    summary += f" (fields: {', '.join(key_fields)})"
                break
        
        return summary
    
    def _update_conversation_summary(self, session_id: str):
        """Update rolling summary of older conversations"""
        if session_id not in self._conversation_memory:
            return
        
        # Get conversations that will be removed (older than last 10)
        old_conversations = self._conversation_memory[session_id][:-10]
        
        if not old_conversations:
            return
        
        # Create summary of old conversations
        summary_points = []
        for turn in old_conversations:
            query_summary = turn['query'][:100]
            if turn.get('agents_used'):
                query_summary += f" (used: {', '.join(turn['agents_used'])})"
            summary_points.append(f"- {query_summary}")
        
        new_summary = "Previous conversation topics:\n" + "\n".join(summary_points[-20:])  # Keep last 20 topics
        
        # Combine with existing summary if any
        existing_summary = self._conversation_summaries.get(session_id, "")
        if existing_summary:
            # Use LLM to create consolidated summary
            combined_summary = self._consolidate_conversation_summary(existing_summary, new_summary)
            self._conversation_summaries[session_id] = combined_summary
        else:
            self._conversation_summaries[session_id] = new_summary
    
    def _consolidate_conversation_summary(self, existing_summary: str, new_summary: str) -> str:
        """Use LLM to consolidate conversation summaries"""
        try:
            prompt = f"""
Consolidate these conversation summaries into a concise overview:

EXISTING SUMMARY:
{existing_summary}

NEW TOPICS:
{new_summary}

Create a consolidated summary that:
1. Captures key themes and data types discussed
2. Notes main agents/databases used
3. Identifies recurring topics or follow-up patterns
4. Keeps it under 300 words

Focus on information that would help continue the conversation context.
"""
            
            response = self.llm_client.invoke(prompt)
            return response.content.strip()
        except Exception:
            # Fallback: simple concatenation
            return f"{existing_summary}\n\n{new_summary}"
    
    def _get_conversation_summary(self, session_id: str) -> str:
        """Get conversation summary for session"""
        return self._conversation_summaries.get(session_id, "")
    
    def _build_schema_summary(self, agent: Dict[str, Any]) -> str:
        """Build table schema summary for data agents"""
        try:
            agent_name = agent.get("name", "Unknown")
            database_type = agent.get("database_type", "unknown")
            connection_type = agent.get("connection_type", "unknown")
            
            # Get tables from cached data
            tables = agent.get("tables", [])
            
            if not tables:
                return f"{agent_name} ({database_type}/{connection_type}) - No table schema available"
            
            schema_parts = [f"{agent_name} Database ({database_type}/{connection_type}):"]
            
            # Build table summaries (limit to 5 tables to prevent token overflow)
            for table in tables[:5]:
                table_name = table.get("tableName", "")
                schema_name = table.get("schemaName", "public")
                row_count = table.get("rowCount", 0)
                
                if table_name:
                    full_table_name = f"{schema_name}.{table_name}" if schema_name not in ["public", "dbo"] else table_name
                    
                    # Get column information
                    columns = table.get("columns", [])
                    if columns:
                        # Show important columns with data types and constraints
                        col_info = []
                        for col in columns[:12]:  # Show more columns for better context
                            col_name = col.get("columnName", "")
                            data_type = col.get("dataType", "")
                            
                            if col.get("isPrimaryKey", False):
                                col_info.append(f"{col_name}({data_type}) PK")
                            elif col.get("isForeignKey", False):
                                col_info.append(f"{col_name}({data_type}) FK")
                            else:
                                col_info.append(f"{col_name}({data_type})")
                        
                        cols_text = ", ".join(col_info)
                        if len(columns) > 12:
                            cols_text += f" + {len(columns) - 12} more"
                        
                        table_summary = f"  • {full_table_name} ({row_count:,} rows): {cols_text}"
                    else:
                        table_summary = f"  • {full_table_name} ({row_count:,} rows) - schema not available"
                    
                    schema_parts.append(table_summary)
            
            if len(tables) > 5:
                schema_parts.append(f"  ... and {len(tables) - 5} more tables")
            
            return "\n".join(schema_parts)
            
        except Exception as e:
            return f"Schema unavailable: {str(e)}"
    
    def _extract_data_types_from_schema(self, agent: Dict[str, Any]) -> List[str]:
        """Extract data types/domains from agent tables for better matching"""
        try:
            data_types = set()
            tables = agent.get("tables", [])
            
            for table in tables:
                table_name = table.get("tableName", "").lower()
                
                # Infer data types from table names and descriptions
                if any(keyword in table_name for keyword in ["employee", "staff", "worker"]):
                    data_types.add("employees")
                if any(keyword in table_name for keyword in ["department", "dept"]):
                    data_types.add("departments")
                if any(keyword in table_name for keyword in ["customer", "client"]):
                    data_types.add("customers")
                if any(keyword in table_name for keyword in ["order", "sale", "transaction"]):
                    data_types.add("orders")
                if any(keyword in table_name for keyword in ["product", "item", "inventory"]):
                    data_types.add("products")
                if any(keyword in table_name for keyword in ["vendor", "supplier"]):
                    data_types.add("vendors")
                if any(keyword in table_name for keyword in ["contract", "agreement"]):
                    data_types.add("contracts")
                    
            return list(data_types)
            
        except Exception:
            return []
    
    def _get_application_endpoints(self, agent: Dict[str, Any]) -> List[str]:
        """Get application endpoints for API agents"""
        try:
            endpoints = []
            tools = agent.get("tools", [])
            
            for tool in tools[:5]:  # Limit to 5 endpoints
                endpoint_name = tool.get("name", "")
                method = tool.get("method", "")
                path = tool.get("path", "")
                description = tool.get("description", "")
                
                if endpoint_name:
                    endpoint_info = f"{method} {endpoint_name}"
                    if description:
                        endpoint_info += f" - {description[:50]}"
                    endpoints.append(endpoint_info)
                    
            return endpoints
            
        except Exception:
            return []


# Global instance
simple_orchestrator = SimpleOrchestrator()
