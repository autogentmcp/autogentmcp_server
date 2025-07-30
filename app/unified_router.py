"""
Unified router for both applications and data agents with confidence scoring.
Enhanced with multi-agent orchestration support.
"""
from typing import Dict, Any, List, Optional, Tuple
from app.ollama_client import ollama_client
from app.registry import fetch_agents_and_tools_from_registry
from app.data_agents_client import data_agents_client
import re

class UnifiedRouter:
    """Router that handles both applications and data agents with confidence scoring."""
    
    def __init__(self):
        pass
    
    async def route_query(self, query: str, session_id: str = "default", enable_orchestration: bool = True) -> Dict[str, Any]:
        """
        Route a query to the best agent (application or data agent) based on confidence scoring.
        Now with optional multi-agent orchestration support.
        
        Args:
            query: User query
            session_id: Session identifier
            enable_orchestration: Whether to enable multi-agent orchestration
            
        Returns:
            Dictionary containing routing results and response
        """
        # Input validation
        if not query or not query.strip():
            return {
                "query": query or "",
                "route_type": "error",
                "selected_agent": None,
                "confidence": 0,
                "agent_reason": "Empty query provided",
                "final_answer": "Please provide a valid query to process.",
                "error": "Empty query"
            }
        
        # Handle general conversational queries first
        conversational_response = self._handle_conversational_query(query, session_id)
        if conversational_response:
            return conversational_response
        
        try:
            # Check for multi-agent orchestration if enabled
            if enable_orchestration:
                try:
                    from app.langgraph_orchestrator import langgraph_orchestrator
                    
                    # Execute dynamic workflow with LangGraph
                    workflow_result = await langgraph_orchestrator.execute_workflow(query, session_id)
                    
                    if workflow_result.get("status") == "completed":
                        return {
                            "query": query,
                            "route_type": "langgraph_workflow",
                            "workflow_id": workflow_result["workflow_id"],
                            "execution_summary": workflow_result.get("execution_summary", {}),
                            "final_answer": workflow_result.get("final_answer", "Workflow completed"),
                            "collected_data": workflow_result.get("collected_data", {}),
                            "confidence": 95  # High confidence for successful workflows
                        }
                    elif workflow_result.get("user_input_needed"):
                        return {
                            "query": query,
                            "route_type": "langgraph_workflow",
                            "workflow_id": workflow_result["workflow_id"],
                            "status": "waiting_input",
                            "user_question": workflow_result.get("user_question"),
                            "partial_results": workflow_result.get("collected_data", {})
                        }
                    else:
                        # Workflow failed, fall back to simple routing
                        print(f"[UnifiedRouter] LangGraph workflow failed: {workflow_result.get('error', 'Unknown error')}")
                        
                except Exception as orchestration_error:
                    print(f"[UnifiedRouter] Error in LangGraph orchestration: {orchestration_error}")
                    # Continue with simple routing as fallback
            
            # Continue with existing simple routing logic
            # Extract keywords from query for matching
            try:
                keywords = self._extract_keywords(query)
            except Exception as keyword_error:
                print(f"[UnifiedRouter] Keyword extraction failed: {keyword_error}")
                keywords = []
            
            # Get candidates from both sources
            try:
                app_candidates = self._get_application_candidates(query, keywords)
            except Exception as app_error:
                print(f"[UnifiedRouter] Application candidate search failed: {app_error}")
                app_candidates = []
            
            try:
                data_agent_candidates = self._get_data_agent_candidates(keywords)
            except Exception as data_error:
                print(f"[UnifiedRouter] Data agent candidate search failed: {data_error}")
                data_agent_candidates = []
            
            # Combine and rank all candidates
            all_candidates = app_candidates + data_agent_candidates
            
            if not all_candidates:
                return self._create_no_match_response(query, session_id)
            
            # Sort by confidence score
            all_candidates.sort(key=lambda x: x["confidence"], reverse=True)
            best_candidate = all_candidates[0]
            
            # Emit routing decision event for streaming
            try:
                from app.workflow_streamer import workflow_streamer
                # Use session-based workflow ID so all session clients can see it
                workflow_id = f"session_{session_id}"
                
                # Create candidate summary for streaming
                candidate_summary = []
                for candidate in all_candidates[:5]:  # Top 5 candidates
                    # Extract name based on candidate type
                    if candidate["type"] == "data_agent":
                        candidate_name = candidate.get("agent", {}).get("name", "unknown")
                    else:  # application type
                        candidate_name = candidate.get("name", candidate.get("app_key", "unknown"))
                    
                    candidate_summary.append({
                        "type": candidate["type"],
                        "name": candidate_name,
                        "confidence": candidate["confidence"]
                    })
                
                # Extract agent name properly based on candidate type
                if best_candidate["type"] == "data_agent":
                    selected_agent_name = best_candidate.get("agent", {}).get("name", "unknown")
                else:  # application type
                    selected_agent_name = best_candidate.get("name", best_candidate.get("app_key", "unknown"))
                
                workflow_streamer.emit_routing_decision(
                    workflow_id=workflow_id,
                    session_id=session_id,
                    route_type=best_candidate["type"],
                    selected_agent=selected_agent_name,
                    confidence=best_candidate["confidence"],
                    reasoning=f"Selected best match from {len(all_candidates)} candidates",
                    candidates=candidate_summary
                )
            except Exception as streaming_error:
                print(f"[UnifiedRouter] Error emitting routing decision: {streaming_error}")
            
            # Route to appropriate handler
            try:
                if best_candidate["type"] == "application":
                    return self._handle_application_route(query, best_candidate, session_id)
                elif best_candidate["type"] == "data_agent":
                    return self._handle_data_agent_route(query, best_candidate, session_id)
                else:
                    return self._create_no_match_response(query, session_id)
            except Exception as routing_error:
                print(f"[UnifiedRouter] Error routing to {best_candidate['type']} agent: {routing_error}")
                return {
                    "query": query,
                    "route_type": "error",
                    "selected_agent": best_candidate.get("name", "unknown"),
                    "confidence": 0,
                    "agent_reason": f"Agent execution failed: {str(routing_error)}",
                    "final_answer": f"I encountered an error while processing your request with the selected agent. Please try again later.",
                    "error": str(routing_error)
                }
        
        except Exception as e:
            print(f"[UnifiedRouter] Unexpected routing error: {e}")
            return {
                "query": query,
                "route_type": "error",
                "selected_agent": None,
                "confidence": 0,
                "agent_reason": f"Routing system error: {str(e)}",
                "final_answer": "I encountered an unexpected error while processing your request. Please try again later or contact support if the issue persists.",
                "error": str(e)
            }

    def _handle_conversational_query(self, query: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Handle simple conversational queries that don't need agents or workflows."""
        query_lower = query.lower().strip()
        
        # Define conversational patterns
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        goodbyes = ["bye", "goodbye", "see you", "farewell", "good night"]
        how_are_you = ["how are you", "how's it going", "how are things", "what's up"]
        thank_you = ["thank you", "thanks", "thx", "appreciate it"]
        help_requests = ["help", "what can you do", "what are your capabilities", "how do you work"]
        
        # Check for exact matches or contains
        def matches_pattern(patterns, query_text):
            return any(pattern in query_text for pattern in patterns)
        
        try:
            from app.session_manager import session_manager
            
            final_answer = None
            
            if matches_pattern(greetings, query_lower):
                final_answer = """Hello! 👋 I'm your AI assistant, ready to help you with:

🔍 **Data Queries** - Access databases, generate reports, analyze data
🚀 **Application Integration** - Connect with various business applications  
🤖 **Multi-Agent Workflows** - Complex tasks requiring multiple data sources
📊 **Business Intelligence** - Charts, insights, and analytics

What would you like to work on today?"""
            
            elif matches_pattern(goodbyes, query_lower):
                final_answer = """Goodbye! 👋 It was great working with you today. 

Feel free to come back anytime you need help with data queries, application integration, or complex workflows. Have a wonderful day! 🌟"""
            
            elif matches_pattern(how_are_you, query_lower):
                final_answer = """I'm doing great, thank you for asking! 🤖✨

I'm running smoothly and ready to help you with:
• Database queries and data analysis
• Application integrations
• Multi-step workflows
• Business insights and reporting

How can I assist you today?"""
            
            elif matches_pattern(thank_you, query_lower):
                final_answer = """You're very welcome! 😊

I'm always here to help with your data queries, application integrations, and complex workflows. If you need anything else, just ask!"""
            
            elif matches_pattern(help_requests, query_lower) or query_lower in ["help", "?"]:
                final_answer = """I'm your intelligent assistant capable of handling various tasks! Here's what I can do:

## 🔍 **Data Queries**
- Connect to databases (SQL Server, PostgreSQL, MySQL, etc.)
- Generate SQL queries from natural language
- Create reports and analyze data trends

## 🚀 **Application Integration**
- Connect with business applications via APIs
- Automate workflows between different systems
- Retrieve and process application data

## 🤖 **Multi-Agent Workflows**
- Handle complex requests requiring multiple data sources
- Sequential processing with dependency management
- User interaction handling during workflows

## 📊 **Business Intelligence**
- Generate charts and visualizations
- Comparative analysis across data sources
- Financial reporting and insights

## 💬 **Example Queries**
- "Show me sales data from last month"
- "Compare revenue between Q1 and Q2"
- "List all customers in California with orders > $1000"
- "Generate a financial summary report"
- "What's our top-selling product category?"

Just ask me anything in natural language, and I'll figure out the best way to help you!"""
            
            if final_answer:
                session_manager.add_to_session(session_id, query, final_answer)
                return {
                    "query": query,
                    "route_type": "conversational",
                    "selected_agent": "built_in_conversational",
                    "confidence": 100,
                    "agent_reason": "Conversational query handled by built-in responses",
                    "final_answer": final_answer
                }
            
        except Exception as e:
            print(f"[UnifiedRouter] Error in conversational handler: {e}")
        
        return None  # Not a conversational query, continue with normal routing
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from the query."""
        # Simple keyword extraction - can be enhanced with NLP
        # Remove common words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "can", "may", "might", "must", "shall", "get", "show", "find", "list"
        }
        
        # Extract words and filter out stop words
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _get_application_candidates(self, query: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Get application candidates with confidence scores."""
        try:
            applications = fetch_agents_and_tools_from_registry()
            candidates = []
            
            for app_key, app_info in applications.items():
                score = self._calculate_application_confidence(app_info, query, keywords)
                if score > 0:
                    candidates.append({
                        "type": "application",
                        "app_key": app_key,
                        "agent": app_info,
                        "confidence": score
                    })
            
            return candidates
        except Exception as e:
            print(f"[UnifiedRouter] Error getting application candidates: {e}")
            return []
    
    def _get_data_agent_candidates(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Get data agent candidates with confidence scores."""
        try:
            return data_agents_client.search_data_agents_by_keywords(keywords)
        except Exception as e:
            print(f"[UnifiedRouter] Error getting data agent candidates: {e}")
            return []
    
    def _calculate_application_confidence(self, app_info: Dict[str, Any], query: str, keywords: List[str]) -> float:
        """
        Calculate confidence score for applications.
        
        Scoring logic:
        - App name match: 20 points
        - App description match: 15 points
        - Tool name match: 25 points
        - Tool description match: 10 points
        - Endpoint path relevance: 15 points
        """
        score = 0
        query_lower = query.lower()
        keywords_lower = [kw.lower() for kw in keywords]
        
        # Check app name and description
        app_name = app_info.get("name", "").lower()
        app_desc = app_info.get("description", "").lower()
        
        for keyword in keywords_lower:
            if keyword in app_name:
                score += 20
            if keyword in app_desc:
                score += 15
        
        # Check tools
        for tool in app_info.get("tools", []):
            tool_name = tool.get("name", "").lower()
            tool_desc = tool.get("description", "").lower()
            tool_path = tool.get("path", "").lower()
            
            for keyword in keywords_lower:
                if keyword in tool_name:
                    score += 25
                if keyword in tool_desc:
                    score += 10
                if keyword in tool_path:
                    score += 15
        
        return score
    
    def _handle_application_route(self, query: str, candidate: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle routing to an application."""
        from app.session_manager import session_manager
        from app.tool_selector import tool_selector
        from app.endpoint_invoker import endpoint_invoker
        
        app_key = candidate["app_key"]
        agent_info = candidate["agent"]
        
        try:
            # Get conversation history
            history = session_manager.get_history_string(session_id)
            
            # Select specific tool using existing logic
            _, tool_name, _, tool_info = tool_selector.select_agent_and_tool(query, history)
            
            if not tool_name or not tool_info:
                final_answer = f"Found application '{agent_info.get('name')}' but couldn't select appropriate tool."
                session_manager.add_to_session(session_id, query, final_answer)
                return {
                    "query": query,
                    "route_type": "application",
                    "selected_agent": app_key,
                    "confidence": candidate["confidence"],
                    "agent_reason": f"Application match (confidence: {candidate['confidence']:.1f})",
                    "selected_tool": None,
                    "call_result": None,
                    "final_answer": final_answer
                }
            
            # Prepare endpoint invocation
            method = tool_info.get("method", "GET").upper()
            resolved_endpoint = tool_info.get("resolved_endpoint")
            
            if not resolved_endpoint:
                base_domain = agent_info.get("base_domain", "")
                endpoint_path = tool_info.get("path", "")
                resolved_endpoint = base_domain + endpoint_path
            
            # Invoke endpoint
            call_result = endpoint_invoker.invoke_registry_endpoint(
                app_key=app_key,
                agent_info=agent_info,
                tool_info=tool_info,
                resolved_endpoint=resolved_endpoint,
                method=method,
                query_params=tool_info.get("query_params", {}),
                body_params=tool_info.get("body_params", {}),
                headers=tool_info.get("headers", {})
            )
            
            # Generate final answer
            final_prompt = ollama_client.create_final_answer_prompt(query, call_result)
            final_answer = ollama_client.invoke_with_text_response(final_prompt, allow_diagrams=True)
            
            # Update session
            session_manager.add_to_session(session_id, query, final_answer)
            
            return {
                "query": query,
                "route_type": "application",
                "selected_agent": app_key,
                "confidence": candidate["confidence"],
                "agent_reason": f"Application match (confidence: {candidate['confidence']:.1f})",
                "selected_tool": tool_name,
                "call_result": call_result,
                "final_answer": final_answer
            }
            
        except Exception as e:
            error_msg = f"Error processing application route: {str(e)}"
            print(f"[UnifiedRouter] {error_msg}")
            return {
                "query": query,
                "route_type": "application",
                "selected_agent": app_key,
                "confidence": candidate["confidence"],
                "error": error_msg,
                "final_answer": error_msg
            }
    
    def _handle_data_agent_route(self, query: str, candidate: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle routing to a data agent."""
        from app.session_manager import session_manager
        from app.database_query_executor import database_query_executor
        
        agent = candidate["agent"]
        agent_id = agent.get("id")
        
        try:
            # Get database connection info
            connection_info = data_agents_client.get_connection_info(agent_id)
            if not connection_info:
                error_msg = f"No connection information found for data agent '{agent.get('name')}'"
                return {
                    "query": query,
                    "route_type": "data_agent",
                    "selected_agent": agent_id,
                    "confidence": candidate["confidence"],
                    "error": error_msg,
                    "final_answer": error_msg
                }
            
            # Generate SQL query using LLM
            sql_query = self._generate_sql_query(query, agent, connection_info)
            
            if not sql_query:
                error_msg = "Could not generate appropriate SQL query"
                return {
                    "query": query,
                    "route_type": "data_agent",
                    "selected_agent": agent_id,
                    "confidence": candidate["confidence"],
                    "error": error_msg,
                    "final_answer": error_msg
                }
            
            # Emit SQL generation event for streaming
            try:
                from app.workflow_streamer import workflow_streamer
                workflow_id = f"session_{session_id}"
                workflow_streamer.emit_sql_generated(
                    workflow_id=workflow_id,
                    session_id=session_id,
                    sql_query=sql_query,
                    database_type=connection_info.get('connection_type', 'unknown')
                )
            except Exception as streaming_error:
                print(f"[UnifiedRouter] Error emitting SQL generation: {streaming_error}")
            
            # Execute query
            print(f"[DEBUG] Executing SQL query on {connection_info['connection_type']} database")
            query_result = database_query_executor.execute_query(
                vault_key=connection_info["vault_key"],
                connection_type=connection_info["connection_type"],
                sql_query=sql_query
            )
            
            if not query_result:
                print("[ERROR] No query result returned from database")
                query_result = {"error": "No results returned from database query"}
            
            print(f"[DEBUG] Query result type: {type(query_result)}, content preview: {str(query_result)[:200]}...")
            
            # Generate final answer
            print("[DEBUG] Generating final answer with LLM")
            final_prompt = ollama_client.create_data_answer_prompt(query, sql_query, query_result)
            final_answer = ollama_client.invoke_with_text_response(final_prompt, allow_diagrams=True)
            
            if not final_answer:
                print("[ERROR] No final answer generated from LLM")
                final_answer = f"Query executed successfully. SQL: {sql_query}. Result: {query_result}"
            
            print(f"[DEBUG] Final answer generated: {final_answer[:200]}...")
            
            # Update session
            session_manager.add_to_session(session_id, query, final_answer)
            
            return {
                "query": query,
                "route_type": "data_agent",
                "selected_agent": agent_id,
                "agent_name": agent.get("name"),
                "confidence": candidate["confidence"],
                "agent_reason": f"Data agent match (confidence: {candidate['confidence']:.1f})",
                "sql_query": sql_query,
                "query_result": query_result,
                "final_answer": final_answer
            }
            
        except Exception as e:
            error_msg = f"Error processing data agent route: {str(e)}"
            print(f"[UnifiedRouter] {error_msg}")
            return {
                "query": query,
                "route_type": "data_agent",
                "selected_agent": agent_id,
                "confidence": candidate["confidence"],
                "error": error_msg,
                "final_answer": error_msg
            }
    
    def _generate_sql_query(self, query: str, agent: Dict[str, Any], connection_info: Dict[str, Any]) -> Optional[str]:
        """Generate SQL query using LLM based on data agent schema with structured JSON response."""
        try:
            # Get structured table data directly instead of building text context
            tables_data = agent.get("tables", [])
            connection_type = connection_info.get('connection_type', 'unknown')
            
            print(f"[DEBUG] SQL Generation Debug Info:")
            print(f"[DEBUG] Connection Type: {connection_type}")
            print(f"[DEBUG] Number of tables in structured data: {len(tables_data)}")
            
            # Log table structure for debugging
            for i, table in enumerate(tables_data):
                table_name = table.get("tableName", "Unknown")
                columns = table.get("columns", [])
                print(f"[DEBUG] Table {i+1}: {table_name} ({len(columns)} columns)")
                for j, col in enumerate(columns[:5]):  # Show first 5 columns
                    col_name = col.get("columnName", "")
                    print(f"[DEBUG]   Column {j+1}: {col_name}")
                if len(columns) > 5:
                    print(f"[DEBUG]   ... and {len(columns) - 5} more columns")
            
            # Use the enhanced SQL generation prompt with structured data
            prompt = ollama_client.create_sql_generation_prompt(
                user_query=query,
                connection_type=connection_type,
                tables_data=tables_data,  # Pass structured data directly
                agent_name=agent.get('name', 'Unknown Agent'),
            )
            
            print(f"[DEBUG] Generated prompt length: {len(prompt)} characters")
            print(f"[DEBUG] Full prompt being sent to LLM:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)
            
            print(f"[DEBUG] Generating SQL with structured data for {connection_type}")
            print(f"[DEBUG] Using {len(tables_data)} tables from structured data")
            
            # Use JSON response method for structured output
            sql_response_json = ollama_client.invoke_with_json_response(prompt)
            
            if not sql_response_json:
                print("[ERROR] No JSON response from LLM")
                return None
            
            # Extract thought and query from JSON response
            thought = sql_response_json.get("thought", "")
            sql_query = sql_response_json.get("query", "")
            
            if thought:
                print(f"[DEBUG] LLM reasoning: {thought}")
            
            if not sql_query:
                print("[ERROR] No SQL query found in JSON response")
                return None
            
            # Basic cleanup of the SQL query
            sql_query = sql_query.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:].strip()
            if sql_query.startswith("```"):
                sql_query = sql_query[3:].strip()
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3].strip()
            
            print(f"[DEBUG] Generated SQL query: {sql_query}")
            return sql_query
            
        except Exception as e:
            print(f"[UnifiedRouter] Error generating SQL query: {e}")
            # Fallback to text-based approach if JSON parsing fails
            try:
                print("[DEBUG] Falling back to text-based SQL generation")
                sql_response = ollama_client.invoke_with_text_response(prompt)
                
                if not sql_response:
                    return None
                
                # Try to extract JSON from text response
                import json
                import re
                json_match = re.search(r'\{[^}]*"query"[^}]*\}', sql_response, re.DOTALL)
                if json_match:
                    try:
                        parsed_json = json.loads(json_match.group())
                        sql_query = parsed_json.get("query", "").strip()
                        thought = parsed_json.get("thought", "")
                        if thought:
                            print(f"[DEBUG] LLM reasoning (from text): {thought}")
                        if sql_query:
                            print(f"[DEBUG] Generated SQL query (from text): {sql_query}")
                            return sql_query
                    except json.JSONDecodeError:
                        pass
                
                # Last resort: assume the response is just SQL
                sql_query = sql_response.strip()
                print(f"[DEBUG] Generated SQL query (fallback): {sql_query}")
                return sql_query
                
            except Exception as fallback_error:
                print(f"[UnifiedRouter] Fallback SQL generation also failed: {fallback_error}")
                return None
    
    def _build_schema_context(self, agent: Dict[str, Any]) -> str:
        """Build comprehensive schema context for SQL generation including relationships."""
        context_parts = []
        
        # Build table information with columns
        for table in agent.get("tables", []):
            table_name = table.get("tableName")
            schema_name = table.get("schemaName", "")
            full_table_name = f"{schema_name}.{table_name}" if schema_name else table_name
            
            context_parts.append(f"Table: {full_table_name}")
            if table.get("description"):
                context_parts.append(f"  Description: {table.get('description')}")
            
            context_parts.append("  Columns:")
            for column in table.get("columns", []):
                col_name = column.get("columnName")
                col_type = column.get("dataType")
                is_pk = " (PRIMARY KEY)" if column.get("isPrimaryKey") else ""
                is_fk = " (FOREIGN KEY)" if column.get("isForeignKey") else ""
                nullable = " (NOT NULL)" if not column.get("isNullable") else ""
                default_val = f" DEFAULT {column.get('defaultValue')}" if column.get("defaultValue") else ""
                
                context_parts.append(f"    - {col_name}: {col_type}{is_pk}{is_fk}{nullable}{default_val}")
            
            # Add foreign key relationships if available
            relationships = table.get("relationships", [])
            if relationships:
                context_parts.append("  Relationships:")
                for rel in relationships:
                    rel_type = rel.get("relationshipType", "UNKNOWN")
                    target_table = rel.get("targetTable", "")
                    source_column = rel.get("sourceColumn", "")
                    target_column = rel.get("targetColumn", "")
                    
                    if target_table and source_column and target_column:
                        context_parts.append(f"    - {rel_type}: {source_column} -> {target_table}.{target_column}")
            
            context_parts.append("")
        
        # Add a relationships summary section
        context_parts.append("RELATIONSHIPS SUMMARY:")
        all_relationships = []
        
        for table in agent.get("tables", []):
            table_name = table.get("tableName")
            schema_name = table.get("schemaName", "")
            full_table_name = f"{schema_name}.{table_name}" if schema_name else table_name
            
            for rel in table.get("relationships", []):
                target_table = rel.get("targetTable", "")
                source_column = rel.get("sourceColumn", "")
                target_column = rel.get("targetColumn", "")
                
                if target_table and source_column and target_column:
                    # Add schema prefix if available
                    if schema_name and "." not in target_table:
                        target_table = f"{schema_name}.{target_table}"
                    
                    relationship_desc = f"{full_table_name}.{source_column} = {target_table}.{target_column}"
                    all_relationships.append(relationship_desc)
        
        if all_relationships:
            for rel in all_relationships:
                context_parts.append(f"- JOIN: {rel}")
        else:
            context_parts.append("- No explicit relationships defined. Use common column names for joins.")
        
        return "\n".join(context_parts)
    
    def _create_no_match_response(self, query: str, session_id: str) -> Dict[str, Any]:
        """Create response when no suitable agent is found."""
        try:
            from app.session_manager import session_manager
            
            final_answer = """I couldn't find an appropriate agent or data source to handle your request. Here are some things you can try:

1. **Rephrase your question** - Try using different keywords or being more specific
2. **Check service availability** - Some services might be temporarily unavailable
3. **Try a simpler request** - Break complex questions into smaller parts
4. **Contact support** - If this issue persists, please reach out for assistance

Example queries that work well:
- "Show me sales data from last month"
- "List all customers in California" 
- "What's the current inventory status?"
- "Generate a financial report"

Please try again with a different approach."""
            
            session_manager.add_to_session(session_id, query, final_answer)
            
            return {
                "query": query,
                "route_type": "none",
                "selected_agent": None,
                "confidence": 0,
                "agent_reason": "No suitable agent found for this query",
                "final_answer": final_answer,
                "suggestions": [
                    "Try rephrasing your question",
                    "Check if relevant services are available", 
                    "Break complex requests into simpler parts",
                    "Use more specific keywords"
                ]
            }
        except Exception as e:
            print(f"[UnifiedRouter] Error creating no match response: {e}")
            return {
                "query": query,
                "route_type": "error",
                "selected_agent": None,
                "confidence": 0,
                "agent_reason": f"Error creating response: {str(e)}",
                "final_answer": "I'm currently unable to process your request due to a system issue. Please try again later.",
                "error": str(e)
            }

# Global instance
unified_router = UnifiedRouter()
