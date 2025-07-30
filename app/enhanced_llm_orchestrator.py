"""
Enhanced LLM-Powered Multi-Hop Orchestrator
Builds upon the existing enhanced_orchestrator.py with:
- Advanced LLM-driven planning and routing
- Multi-hop reasoning and follow-up capabilities
- Result analysis and insight generation
- Feedback learning system
- Dynamic schema introspection
- Database-specific SQL dialect handling
"""

import json
import asyncio
import time
import sqlite3
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from app.workflow_streamer import workflow_streamer
from app.multimode_llm_client import MultiModeLLMClient, TaskType
from app.registry import fetch_agents_and_tools_from_registry, get_enhanced_agent_details_for_llm
from app.database_query_executor import DatabaseQueryExecutor
from app.vault_manager import vault_manager
from app.data_sampling_middleware import DataSamplingMiddleware, SamplingConfig
import re


@dataclass
class EnhancedAgentResult:
    """Enhanced result structure with additional metadata."""
    success: bool
    data: Any
    agent_id: str
    agent_name: str
    agent_type: str
    execution_time: float
    query_executed: Optional[str] = None
    error: Optional[str] = None
    row_count: Optional[int] = None
    insight_summary: Optional[str] = None


@dataclass
class ConversationContext:
    """Enhanced context for multi-turn conversations."""
    session_id: str
    workflow_id: str = ""
    original_query: str = ""
    refined_query: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    agent_results: List[EnhancedAgentResult] = field(default_factory=list)
    current_step: int = 0
    max_steps: int = 10
    start_time: float = field(default_factory=time.time)
    user_feedback: Optional[str] = None
    requires_follow_up: bool = False
    suggested_follow_ups: List[str] = field(default_factory=list)
    
    # Multi-agent orchestration fields
    execution_strategy: str = "single_agent"  # single_agent, parallel_agents, sequential_agents
    primary_agent_id: Optional[str] = None
    additional_agent_ids: List[str] = field(default_factory=list)
    agent_coordination: str = "independent"  # independent, results_merge, sequential_dependency
    
    # Enhanced orchestrator fields
    intent_category: Optional[str] = None
    query_type: Optional[str] = None
    selected_agent_id: Optional[str] = None
    selected_agent_name: Optional[str] = None
    multi_agent_reasoning: Optional[str] = None
    workflow_streamer: Optional[Any] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
        if self.conversation_history is None:
            self.conversation_history = []
        if self.agent_results is None:
            self.agent_results = []
        if self.suggested_follow_ups is None:
            self.suggested_follow_ups = []
        if self.additional_agent_ids is None:
            self.additional_agent_ids = []


class SessionContextManager:
    def create_new_conversation(self, session_id: str) -> str:
        """
        Create a new conversation for the given session.
        Returns a new conversation_id (uuid).
        """
        conversation_id = f"conv_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        session = self._sessions.get(session_id)
        if session is not None:
            if 'conversations' not in session:
                session['conversations'] = []
            session['conversations'].append({
                'conversation_id': conversation_id,
                'created_at': time.time(),
                'title': 'New Chat',
                'history': []
            })
            session['last_updated'] = time.time()
        else:
            # If session doesn't exist, create it
            self._sessions[session_id] = {
                'conversations': [{
                    'conversation_id': conversation_id,
                    'created_at': time.time(),
                    'title': 'New Chat',
                    'history': []
                }],
                'created_at': time.time(),
                'last_updated': time.time()
            }
        print(f"[SessionContextManager] Created new conversation: {conversation_id} for session: {session_id}")
        return conversation_id

    def get_user_conversations(self, session_id: str) -> list:
        """
        Return all conversations for a session.
        """
        session = self._sessions.get(session_id)
        if session and 'conversations' in session:
            # Add last_activity timestamp for frontend sorting
            conversations = []
            for conv in session['conversations']:
                conv_copy = conv.copy()
                # Use last message timestamp or created_at as last_activity
                if conv_copy.get('history') and len(conv_copy['history']) > 0:
                    last_message = conv_copy['history'][-1]
                    conv_copy['last_activity'] = last_message.get('timestamp', conv_copy.get('created_at', time.time()))
                else:
                    conv_copy['last_activity'] = conv_copy.get('created_at', time.time())
                conversations.append(conv_copy)
            return conversations
        return []

    def update_conversation_title(self, session_id: str, conversation_id: str, title: str):
        """
        Update the title of a specific conversation.
        """
        session = self._sessions.get(session_id)
        if session and 'conversations' in session:
            for conversation in session['conversations']:
                if conversation['conversation_id'] == conversation_id:
                    conversation['title'] = title
                    conversation['last_updated'] = time.time()
                    session['last_updated'] = time.time()
                    print(f"[SessionContextManager] Updated conversation {conversation_id} title to: '{title}'")
                    return True
        print(f"[SessionContextManager] Could not find conversation {conversation_id} in session {session_id}")
        return False
    def generate_session_from_fingerprint(self, fingerprint: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate a session_id and user_id from browser fingerprint.
        Uses fingerprint data if available, else falls back to uuid.
        """
        # Use a hash of the fingerprint for user_id if possible
        fingerprint_str = json.dumps(fingerprint, sort_keys=True)
        user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, fingerprint_str)) if fingerprint_str else str(uuid.uuid4())
        # Session id is always unique per request
        session_id = f"sess_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        # Store session
        self._sessions[session_id] = {
            'user_id': user_id,
            'fingerprint': fingerprint,
            'created_at': time.time(),
            'last_updated': time.time(),
            'conversation_history': [],
        }
        print(f"[SessionContextManager] Created session: user_id={user_id}, session_id={session_id}, fingerprint={fingerprint}")
        return user_id, session_id

    def _calculate_fingerprint_quality(self, fingerprint: Dict[str, Any]) -> int:
        """
        Dummy fingerprint quality metric (0-100).
        Returns higher value for more keys present.
        """
        if not fingerprint:
            return 0
        keys = ["user_agent", "screen_resolution", "timezone", "language", "platform", "timestamp"]
        score = sum(1 for k in keys if k in fingerprint)
        return int((score / len(keys)) * 100)


class SessionContextManager:
    """Manages session context for multi-turn conversations with enhanced history management."""
    
    def __init__(self):
        self._sessions = {}  # session_id -> session_data
        self._session_ttl = 3600  # 1 hour session timeout
        self._max_history_turns = 5  # Keep last 5 conversation turns
        self._max_summary_length = 500  # Maximum length for summarized history
    
    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get existing session context."""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            # Check if session has expired
            if time.time() - session.get('last_updated', 0) < self._session_ttl:
                return session
            else:
                # Session expired, remove it
                del self._sessions[session_id]
        return None
    
    def update_session_context(self, session_id: str, context: ConversationContext):
        """Update session context with latest conversation and maintain history."""
        # Get existing session or create new one
        existing_session = self._sessions.get(session_id, {})
        conversation_turns = existing_session.get('conversation_turns', [])
        
        # Create current turn data
        current_turn = {
            'timestamp': time.time(),
            'query': context.original_query,
            'refined_query': context.refined_query,
            'workflow_id': context.workflow_id,
            'agent_results': [
                {
                    'success': r.success,
                    'data': r.data,
                    'agent_id': r.agent_id,
                    'agent_name': r.agent_name,
                    'agent_type': r.agent_type,
                    'execution_time': r.execution_time,
                    'query_executed': r.query_executed,
                    'row_count': r.row_count,
                    'insight_summary': r.insight_summary
                } for r in context.agent_results
            ],
            'execution_strategy': getattr(context, 'execution_strategy', 'unknown'),
            'suggested_follow_ups': context.suggested_follow_ups
        }
        
        # Add current turn to conversation history
        conversation_turns.append(current_turn)
        
        # Maintain only the last N turns, summarize older ones
        if len(conversation_turns) > self._max_history_turns:
            # Summarize the oldest turns beyond our limit
            turns_to_summarize = conversation_turns[:-self._max_history_turns]
            conversation_turns = conversation_turns[-self._max_history_turns:]
            
            # Create or update summary of older conversations
            summary = self._summarize_conversation_turns(turns_to_summarize, existing_session.get('conversation_summary', ''))
            conversation_summary = summary
        else:
            conversation_summary = existing_session.get('conversation_summary', '')
        
        # Update session data
        session_data = {
            'last_updated': time.time(),
            'conversation_turns': conversation_turns,
            'conversation_summary': conversation_summary,
            'conversation_history': context.conversation_history,  # Keep for backward compatibility
            'agent_results': current_turn['agent_results'],  # Keep for backward compatibility
            'last_successful_agent': context.agent_results[-1].agent_id if context.agent_results and context.agent_results[-1].success else None,
            'last_query': context.original_query,
            'last_refined_query': context.refined_query,
            'workflow_id': context.workflow_id,
            'total_conversations': len(conversation_turns) + len(turns_to_summarize if len(conversation_turns) == self._max_history_turns else [])
        }
        
        self._sessions[session_id] = session_data
        print(f"💾 Updated session context for {session_id} (turns: {len(conversation_turns)}, total: {session_data['total_conversations']})")
    
    def _summarize_conversation_turns(self, turns_to_summarize: List[Dict], existing_summary: str = '') -> str:
        """Create a summary of conversation turns using LLM."""
        if not turns_to_summarize:
            return existing_summary
        
        try:
            from app.ollama_client import OllamaClient
            llm_client = OllamaClient()
            
            # Prepare turns data for summarization
            turns_text = []
            for i, turn in enumerate(turns_to_summarize):
                query = turn.get('query', 'Unknown query')
                agents_used = [r.get('agent_name', 'Unknown') for r in turn.get('agent_results', []) if r.get('success')]
                insights = [r.get('insight_summary', '') for r in turn.get('agent_results', []) if r.get('success') and r.get('insight_summary')]
                
                turn_summary = f"Turn {i+1}: User asked '{query}'"
                if agents_used:
                    turn_summary += f" | Used agents: {', '.join(agents_used[:2])}"
                if insights:
                    turn_summary += f" | Key insights: {' | '.join(insights[:1])}"
                
                turns_text.append(turn_summary)
            
            prompt = f"""
            Please create a concise summary of these previous conversation turns. Keep it under {self._max_summary_length} characters.
            
            Existing Summary: {existing_summary or 'None'}
            
            New Turns to Summarize:
            {chr(10).join(turns_text)}
            
            Create a comprehensive but concise summary that:
            1. Captures the main topics discussed
            2. Notes key data sources/agents used
            3. Highlights important findings or patterns
            4. Maintains continuity with existing summary
            
            Summary:
            """
            
            response = llm_client.invoke(prompt, timeout=30)
            summary = response.content.strip()[:self._max_summary_length]
            
            print(f"📝 Created conversation summary: {len(summary)} chars")
            return summary
            
        except Exception as e:
            print(f"⚠️ Failed to create conversation summary: {e}")
            # Fallback: simple text summary
            topics = set()
            agents = set()
            for turn in turns_to_summarize:
                if turn.get('query'):
                    # Extract key words from queries
                    words = turn['query'].lower().split()
                    for word in words:
                        if len(word) > 4 and word not in ['show', 'give', 'find', 'what', 'where', 'when', 'how']:
                            topics.add(word)
                
                for result in turn.get('agent_results', []):
                    if result.get('success') and result.get('agent_name'):
                        agents.add(result['agent_name'])
            
            fallback_summary = f"Previous discussions covered: {', '.join(list(topics)[:5])}. Used agents: {', '.join(list(agents)[:3])}."
            return (existing_summary + ' ' + fallback_summary)[:self._max_summary_length]
    
    def get_conversation_context_for_llm(self, session_id: str) -> Dict[str, Any]:
        """Get formatted conversation context for LLM prompts."""
        session = self.get_session_context(session_id)
        if not session:
            return {'has_context': False}
        
        context = {'has_context': True}
        
        # Add conversation summary if available
        summary = session.get('conversation_summary', '')
        if summary:
            context['conversation_summary'] = summary
        
        # Add recent conversation turns
        recent_turns = session.get('conversation_turns', [])
        if recent_turns:
            formatted_turns = []
            for i, turn in enumerate(recent_turns[-5:]):  # Last 5 turns
                turn_info = {
                    'turn_number': len(recent_turns) - 5 + i + 1 if len(recent_turns) > 5 else i + 1,
                    'query': turn.get('query', ''),
                    'agents_used': [r.get('agent_name', '') for r in turn.get('agent_results', []) if r.get('success')],
                    'key_insights': [r.get('insight_summary', '') for r in turn.get('agent_results', []) if r.get('success') and r.get('insight_summary')],
                    'execution_strategy': turn.get('execution_strategy', 'unknown'),
                    'timestamp': turn.get('timestamp', 0)
                }
                formatted_turns.append(turn_info)
            
            context['recent_turns'] = formatted_turns
            context['total_conversations'] = session.get('total_conversations', len(recent_turns))
        
        return context
    
    def detect_follow_up_intent(self, current_query: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if current query is a follow-up to previous conversation."""
        follow_up_indicators = [
            'yes', 'sure', 'please', 'go ahead', 'continue', 'proceed',
            'that would be great', 'sounds good', 'absolutely', 'definitely',
            'i would like that', 'let\'s do it', 'perfect', 'exactly',
            'report', 'dashboard', 'visualization', 'chart', 'graph',
            'more details', 'elaborate', 'expand on that'
        ]
        
        query_lower = current_query.lower().strip()
        
        # Check for simple affirmative responses
        is_affirmative = any(indicator in query_lower for indicator in follow_up_indicators)
        
        # Check if query is very short (likely a follow-up)
        is_short_response = len(query_lower.split()) <= 5
        
        # Get previous agent results
        last_agent_results = session_context.get('agent_results', [])
        has_previous_data = bool(last_agent_results and any(r.get('success', False) for r in last_agent_results))
        
        if is_affirmative and is_short_response and has_previous_data:
            return {
                'is_follow_up': True,
                'follow_up_type': 'affirmative_continuation',
                'previous_agent': session_context.get('last_successful_agent'),
                'previous_data': last_agent_results,
                'suggested_action': 'generate_report_dashboard'
            }
        
        return {'is_follow_up': False}


class FeedbackManager:
    """Manages user feedback and learning from interactions."""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for feedback storage."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Interaction logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interaction_logs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    workflow_id TEXT,
                    user_question TEXT,
                    refined_query TEXT,
                    selected_agents TEXT,  -- JSON array
                    agent_calls TEXT,      -- JSON array of agent executions
                    final_answer TEXT,
                    user_feedback TEXT,    -- thumbs_up, thumbs_down, etc.
                    execution_time REAL,
                    token_cost INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Agent performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    agent_type TEXT,
                    query_type TEXT,
                    success_rate REAL,
                    avg_execution_time REAL,
                    total_calls INTEGER,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Query patterns table for reuse
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE,
                    original_query TEXT,
                    successful_agent_path TEXT,  -- JSON
                    success_count INTEGER DEFAULT 0,
                    total_attempts INTEGER DEFAULT 0,
                    last_used DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def log_interaction(self, context: ConversationContext, feedback: str = None):
        """Log a complete interaction for learning purposes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                interaction_id = str(uuid.uuid4())
                selected_agents = [result.agent_id for result in context.agent_results]
                agent_calls = [
                    {
                        "agent_id": r.agent_id,
                        "agent_type": r.agent_type,
                        "query": r.query_executed,
                        "success": r.success,
                        "execution_time": r.execution_time,
                        "error": r.error
                    }
                    for r in context.agent_results
                ]
                
                cursor.execute("""
                    INSERT INTO interaction_logs 
                    (id, session_id, workflow_id, user_question, refined_query, 
                     selected_agents, agent_calls, user_feedback, execution_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction_id,
                    context.session_id,
                    context.workflow_id,
                    context.original_query,
                    context.refined_query,
                    json.dumps(selected_agents),
                    json.dumps(agent_calls),
                    feedback,
                    time.time() - context.start_time
                ))
                
                conn.commit()
                print(f"✅ Logged interaction: {interaction_id}")
                
        except Exception as e:
            print(f"❌ Failed to log interaction: {e}")
    
    def get_similar_queries(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar successful queries for reuse."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Simple similarity based on common words (can be enhanced with embeddings)
                words = set(query.lower().split())
                
                cursor.execute("""
                    SELECT user_question, selected_agents, agent_calls, user_feedback
                    FROM interaction_logs 
                    WHERE user_feedback IN ('thumbs_up', 'positive')
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit * 2,))
                
                results = []
                for row in cursor.fetchall():
                    question, agents, calls, feedback = row
                    question_words = set(question.lower().split())
                    similarity = len(words & question_words) / len(words | question_words)
                    
                    if similarity > 0.3:  # Threshold for similarity
                        results.append({
                            "question": question,
                            "agents": json.loads(agents),
                            "calls": json.loads(calls),
                            "similarity": similarity
                        })
                
                return sorted(results, key=lambda x: x["similarity"], reverse=True)[:limit]
                
        except Exception as e:
            print(f"❌ Error finding similar queries: {e}")
            return []


class SchemaIntrospector:
    """Schema information provider using cached registry data."""
    
    def __init__(self, db_executor: DatabaseQueryExecutor = None):
        # Keep db_executor for compatibility, but we'll primarily use cached data
        self.db_executor = db_executor
        self._schema_cache = {}
    
    def get_schema_summary(self, agent_id: str, force_refresh: bool = False) -> str:
        """Get schema summary from cached registry data."""
        if not force_refresh and agent_id in self._schema_cache:
            cached_time, schema = self._schema_cache[agent_id]
            if time.time() - cached_time < 3600:  # 1 hour cache
                return schema
        
        try:
            # Get agent details from registry cache (much faster than DB connection)
            agent_details = get_enhanced_agent_details_for_llm(agent_id)
            if not agent_details:
                return "Schema unavailable - agent not found in registry"
            
            # Generate schema summary from cached registry data
            schema_summary = self._build_schema_from_cached_data(agent_details)
            
            # Cache the result
            self._schema_cache[agent_id] = (time.time(), schema_summary)
            
            return schema_summary
            
        except Exception as e:
            print(f"❌ Schema summary generation failed for {agent_id}: {e}")
            return f"Schema generation error: {str(e)}"
    
    def _build_schema_from_cached_data(self, agent_details: Dict[str, Any]) -> str:
        """Build schema summary from cached registry data instead of DB introspection."""
        try:
            agent_name = agent_details.get("name", "Unknown")
            database_type = agent_details.get("database_type", "unknown")
            
            # Get tables from cached data
            tables = agent_details.get("tables", [])
            relations = agent_details.get("relations", [])
            
            if not tables:
                return f"Database: {agent_name} ({database_type}) - No table information available in registry cache"
            
            schema_parts = [f"Database: {agent_name} ({database_type})"]
            
            # Build table summaries from cached data
            for table in tables:  # Pass ALL tables for accurate schema representation
                table_name = table.get("tableName", "")
                schema_name = table.get("schemaName", "public")
                description = table.get("description", "")
                row_count = table.get("rowCount", 0)
                
                if table_name:
                    full_table_name = f"{schema_name}.{table_name}" if schema_name != "public" else table_name
                    
                    # Get column information
                    columns = table.get("columns", [])
                    if columns:
                        # Show key columns (primary keys, foreign keys, and first few regular columns)
                        key_columns = []
                        regular_columns = []
                        
                        for col in columns:
                            col_name = col.get("columnName", "")
                            data_type = col.get("dataType", "")
                            
                            if col.get("isPrimaryKey", False):
                                key_columns.append(f"{col_name}({data_type}) PK")
                            elif col.get("isForeignKey", False):
                                ref_table = col.get("referencedTable", "")
                                key_columns.append(f"{col_name}({data_type}) FK->{ref_table}")
                            else:
                                regular_columns.append(f"{col_name}({data_type})")
                        
                        # Combine key columns with some regular columns
                        shown_columns = key_columns + regular_columns[:8]  # Show max 8 regular columns
                        col_info = ", ".join(shown_columns)
                        
                        if len(columns) > len(shown_columns):
                            col_info += f", ... and {len(columns) - len(shown_columns)} more columns"
                        
                        table_summary = f"Table `{full_table_name}` ({row_count:,} rows) with columns: {col_info}"
                        
                        # Add description if available and meaningful
                        if description and len(description.strip()) > 10:
                            desc_preview = description[:100] + "..." if len(description) > 100 else description
                            table_summary += f" -- {desc_preview}"
                    else:
                        table_summary = f"Table `{full_table_name}` ({row_count:,} rows) - column details not available"
                    
                    schema_parts.append(table_summary)
            
            # Add sample queries from relations if available
            if relations:
                sample_queries = []
                for relation in relations[:3]:  # Show max 3 sample queries
                    example = relation.get("example", "")
                    if example and len(example.strip()) > 10:
                        sample_queries.append(example.strip())
                
                if sample_queries:
                    schema_parts.append("\\nSample Queries:")
                    for i, query in enumerate(sample_queries, 1):
                        query_preview = query[:150] + "..." if len(query) > 150 else query
                        schema_parts.append(f"{i}. {query_preview}")
            
            return "\\n".join(schema_parts)
            
        except Exception as e:
            return f"Error building schema from cached data: {str(e)}"


class SQLSafetyValidator:
    """Comprehensive SQL safety validator to prevent data modifications and schema changes."""
    
    def __init__(self):
        # Define dangerous SQL patterns that can modify data or schema
        self.dangerous_keywords = {
            # Data modification commands
            'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'UPSERT', 'REPLACE',
            
            # Schema modification commands
            'CREATE', 'DROP', 'ALTER', 'TRUNCATE', 'RENAME',
            
            # Database/schema management
            'USE', 'EXEC', 'EXECUTE', 'CALL', 'DO',
            
            # Transaction control that could be used maliciously
            'COMMIT', 'ROLLBACK', 'SAVEPOINT',
            
            # System/administrative commands
            'GRANT', 'REVOKE', 'DENY', 'BACKUP', 'RESTORE',
            
            # Stored procedures and functions that could modify data
            'PROCEDURE', 'FUNCTION', 'TRIGGER', 'INDEX', 'VIEW',
            
            # Database-specific dangerous commands
            'BULK', 'LOAD', 'IMPORT', 'EXPORT', 'COPY',
            
            # Comments that could hide dangerous commands
            'COMMENT', 'DESCRIBE', 'EXPLAIN', 'SHOW',  # These are usually safe but we'll check context
        }
        
        # Safe read-only commands
        self.safe_keywords = {
            'SELECT', 'WITH', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL',
            'GROUP', 'ORDER', 'HAVING', 'DISTINCT', 'UNION', 'INTERSECT', 'EXCEPT',
            'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'ASC', 'DESC', 'LIMIT', 'TOP',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COALESCE', 'ISNULL', 'CAST', 'CONVERT',
            'SUBSTRING', 'CONCAT', 'UPPER', 'LOWER', 'TRIM', 'LIKE', 'IN', 'EXISTS',
            'AND', 'OR', 'NOT', 'IS', 'NULL', 'BETWEEN', 'OVER', 'PARTITION', 'ROW_NUMBER',
            'RANK', 'DENSE_RANK', 'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE'
        }
        
        # Dangerous patterns that could bypass keyword detection
        self.dangerous_patterns = [
            # SQL injection attempts
            r";\s*(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER|TRUNCATE)",
            r"--.*?(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER|TRUNCATE)",
            r"/\*.*?(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER|TRUNCATE).*?\*/",
            
            # Union-based injection attempts to modify data
            r"UNION.*?(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)",
            
            # Nested queries that might hide dangerous operations
            r"\(\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)",
            
            # String concatenation that might hide commands
            r"CONCAT.*?(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER)",
            
            # Dynamic SQL execution
            r"EXEC\s*\(",
            r"EXECUTE\s*\(",
            r"sp_executesql",
            
            # File operations
            r"INTO\s+OUTFILE",
            r"LOAD\s+DATA",
            r"BULK\s+INSERT",
            
            # System commands
            r"xp_cmdshell",
            r"OPENROWSET",
            r"OPENDATASOURCE"
        ]
    
    def validate_sql_safety(self, sql_query: str) -> Dict[str, Any]:
        """
        Comprehensive validation to ensure SQL query is safe (read-only).
        
        Returns:
            Dict with 'is_safe', 'violations', 'sanitized_query', 'risk_level'
        """
        if not sql_query or not sql_query.strip():
            return {
                "is_safe": False,
                "violations": ["Empty or null SQL query"],
                "risk_level": "HIGH",
                "sanitized_query": None
            }
        
        # Normalize the query
        normalized_query = self._normalize_sql(sql_query)
        violations = []
        risk_level = "LOW"
        
        # Check 1: Dangerous keyword detection
        keyword_violations = self._check_dangerous_keywords(normalized_query)
        if keyword_violations:
            violations.extend(keyword_violations)
            risk_level = "HIGH"
        
        # Check 2: Dangerous pattern detection
        pattern_violations = self._check_dangerous_patterns(normalized_query)
        if pattern_violations:
            violations.extend(pattern_violations)
            risk_level = "HIGH"
        
        # Check 3: Ensure query starts with safe commands
        starts_safe = self._check_query_start(normalized_query)
        if not starts_safe:
            violations.append("Query does not start with a safe read-only command")
            risk_level = "HIGH"
        
        # Check 4: Validate parentheses and quote balance
        balance_check = self._check_sql_balance(sql_query)
        if not balance_check["is_balanced"]:
            violations.append(f"SQL structure issue: {balance_check['issue']}")
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
        
        # Check 5: Look for multiple statements (potential SQL injection)
        multi_statement_check = self._check_multiple_statements(normalized_query)
        if multi_statement_check:
            violations.extend(multi_statement_check)
            risk_level = "HIGH"
        
        # Final safety determination
        is_safe = len(violations) == 0
        
        return {
            "is_safe": is_safe,
            "violations": violations,
            "risk_level": risk_level,
            "sanitized_query": self._sanitize_query(sql_query) if is_safe else None,
            "original_query": sql_query
        }
    
    def _normalize_sql(self, sql_query: str) -> str:
        """Normalize SQL query for safety checking."""
        # Remove extra whitespace and normalize to uppercase for keyword detection
        normalized = re.sub(r'\s+', ' ', sql_query.strip()).upper()
        
        # Remove comments but keep track of them for pattern detection
        # We don't actually remove them as they might contain dangerous content
        return normalized
    
    def _check_dangerous_keywords(self, normalized_query: str) -> List[str]:
        """Check for dangerous keywords that could modify data."""
        violations = []
        words = re.findall(r'\b\w+\b', normalized_query)
        
        for word in words:
            if word in self.dangerous_keywords:
                # Special handling for some keywords that might be safe in certain contexts
                if word in ['SHOW', 'DESCRIBE', 'EXPLAIN']:
                    # These are generally safe for read-only operations
                    continue
                elif word == 'COMMENT':
                    # Comments themselves aren't dangerous, but check the content
                    continue
                else:
                    violations.append(f"Dangerous keyword detected: {word}")
        
        return violations
    
    def _check_dangerous_patterns(self, normalized_query: str) -> List[str]:
        """Check for dangerous patterns that might bypass keyword detection."""
        violations = []
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, normalized_query, re.IGNORECASE | re.DOTALL):
                violations.append(f"Dangerous pattern detected: {pattern}")
        
        return violations
    
    def _check_query_start(self, normalized_query: str) -> bool:
        """Ensure query starts with safe read-only commands."""
        # Remove leading comments and whitespace
        query_start = re.sub(r'^\s*(--.*?\n|/\*.*?\*/)*\s*', '', normalized_query, flags=re.DOTALL)
        
        safe_starts = ['SELECT', 'WITH', 'SHOW', 'DESCRIBE', 'EXPLAIN']
        
        for safe_start in safe_starts:
            if query_start.startswith(safe_start):
                return True
        
        return False
    
    def _check_sql_balance(self, sql_query: str) -> Dict[str, Any]:
        """Check if SQL has balanced parentheses and quotes."""
        paren_count = 0
        single_quote_count = 0
        double_quote_count = 0
        bracket_count = 0
        
        i = 0
        while i < len(sql_query):
            char = sql_query[i]
            
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            elif char == "'":
                # Skip escaped quotes
                if i > 0 and sql_query[i-1] == '\\':
                    i += 1
                    continue
                single_quote_count += 1
            elif char == '"':
                # Skip escaped quotes
                if i > 0 and sql_query[i-1] == '\\':
                    i += 1
                    continue
                double_quote_count += 1
            
            i += 1
        
        issues = []
        if paren_count != 0:
            issues.append(f"Unbalanced parentheses (difference: {paren_count})")
        if single_quote_count % 2 != 0:
            issues.append("Unbalanced single quotes")
        if double_quote_count % 2 != 0:
            issues.append("Unbalanced double quotes")
        if bracket_count != 0:
            issues.append(f"Unbalanced brackets (difference: {bracket_count})")
        
        return {
            "is_balanced": len(issues) == 0,
            "issue": "; ".join(issues) if issues else None
        }
    
    def _check_multiple_statements(self, normalized_query: str) -> List[str]:
        """Check for multiple SQL statements that could indicate injection."""
        violations = []
        
        # Look for semicolons that might separate statements
        # This is a simplified check - in reality, semicolons can appear in strings
        semicolon_positions = []
        in_string = False
        quote_char = None
        
        for i, char in enumerate(normalized_query):
            if char in ["'", '"'] and (i == 0 or normalized_query[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    quote_char = char
                elif char == quote_char:
                    in_string = False
                    quote_char = None
            elif char == ';' and not in_string:
                semicolon_positions.append(i)
        
        # If we find semicolons outside of strings, check what follows
        for pos in semicolon_positions:
            remaining = normalized_query[pos+1:].strip()
            if remaining and not remaining.startswith('--'):
                # There's more SQL after a semicolon
                violations.append("Multiple SQL statements detected - potential injection attempt")
                break
        
        return violations
    
    def _sanitize_query(self, sql_query: str) -> str:
        """Sanitize a safe query by removing potentially problematic elements."""
        # Remove comments for extra safety
        sanitized = re.sub(r'--.*?\n', '\n', sql_query)
        sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized.strip())
        
        # Ensure query ends with semicolon
        if not sanitized.endswith(';'):
            sanitized += ';'
        
        return sanitized


class LLMResultAnalyzer:
    """Analyzes query results using LLM for insights and follow-up suggestions."""
    
    def __init__(self, llm_client: MultiModeLLMClient):
        self.llm_client = llm_client
    
    def analyze_result(self, context: ConversationContext, result_data: Any) -> Dict[str, Any]:
        """Analyze query results and provide insights."""
        try:
            # Prepare data for LLM analysis
            data_summary = self._prepare_data_summary(result_data)
            
            prompt = f"""
            You are a data analyst providing insights on query results.
            
            Original Question: "{context.original_query}"
            
            Query Results Summary:
            {data_summary}
            
            Context from previous steps:
            {self._format_execution_context(context)}
            
            Please provide:
            1. Key insights from the data
            2. Notable patterns or anomalies
            3. Suggested follow-up questions
            4. Whether the original question was fully answered
            
            Respond in JSON format:
            {{
                "insights": ["insight 1", "insight 2", ...],
                "patterns": ["pattern 1", "pattern 2", ...],
                "anomalies": ["anomaly 1", "anomaly 2", ...],
                "follow_up_questions": ["question 1", "question 2", ...],
                "question_fully_answered": true|false,
                "confidence": 0.95,
                "summary": "Brief summary of findings"
            }}
            """
            
            try:
                response = self.llm_client.invoke_with_json_response(prompt, timeout=30)
                return response if response else self._fallback_analysis()
            except Exception as e:
                print(f"❌ LLM analysis failed: {e}")
                return self._fallback_analysis()
                
        except Exception as e:
            print(f"❌ Result analysis failed: {e}")
            return self._fallback_analysis()
    
    def _prepare_data_summary(self, data: Any, max_rows: int = 20) -> str:
        """Prepare a concise summary of data for LLM analysis."""
        if not data:
            return "No data returned"
        
        if isinstance(data, list):
            if not data:
                return "Empty result set"
            
            # Show structure and sample data
            sample_size = min(len(data), max_rows)
            summary = f"Result set with {len(data)} rows. Sample of first {sample_size} rows:\\n"
            
            for i, row in enumerate(data[:sample_size]):
                if isinstance(row, dict):
                    # Show key-value pairs
                    row_summary = ", ".join([f"{k}: {v}" for k, v in list(row.items())[:5]])
                    if len(row.items()) > 5:
                        row_summary += ", ..."
                    summary += f"Row {i+1}: {row_summary}\\n"
                else:
                    summary += f"Row {i+1}: {str(row)}\\n"
            
            return summary
        
        elif isinstance(data, dict):
            return f"Single result: {json.dumps(data, default=str, indent=2)}"
        
        else:
            return f"Result: {str(data)}"
    
    def _format_execution_context(self, context: ConversationContext) -> str:
        """Format execution context for LLM."""
        if not context.agent_results:
            return "No previous executions"
        
        context_parts = []
        for i, result in enumerate(context.agent_results):
            status = "✅" if result.success else "❌"
            context_parts.append(
                f"Step {i+1}: {status} {result.agent_name} - {result.insight_summary or 'Executed'}"
            )
        
        return "\\n".join(context_parts)
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        return {
            "insights": ["Data retrieved successfully"],
            "patterns": [],
            "anomalies": [],
            "follow_up_questions": [],
            "question_fully_answered": True,
            "confidence": 0.5,
            "summary": "Query executed successfully"
        }


class EnhancedLLMOrchestrator:
    """
    Enhanced orchestrator with advanced LLM planning, multi-hop reasoning,
    and feedback learning capabilities.
    """
    
    def __init__(self):
        self.llm_client = MultiModeLLMClient()
        self.db_executor = DatabaseQueryExecutor()
        self.feedback_manager = FeedbackManager()
        self.schema_introspector = SchemaIntrospector()  # No longer needs db_executor
        self.result_analyzer = LLMResultAnalyzer(self.llm_client)
        self.sql_validator = SQLSafetyValidator()  # Add SQL safety validator
        self.session_manager = SessionContextManager()  # Add session context manager
        self.data_sampling_middleware = DataSamplingMiddleware()  # Add data sampling middleware
        print("✅ Enhanced LLM Orchestrator initialized with cached schema support, SQL safety validation, session context management, and data sampling middleware")
    
    async def execute_enhanced_workflow(self, user_query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Simple AI assistant workflow that helps users by orchestrating data and application agents.
        Follows a clear 5-step process: Understand → Check Capability → Plan → Execute → Respond
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        workflow_id = str(uuid.uuid4())
        print(f"🚀 Starting AI assistant workflow: {workflow_id}")
        
        try:
            # Emit workflow started for streaming UI compatibility
            workflow_streamer.emit_workflow_started(
                workflow_id=workflow_id,
                session_id=session_id,
                title="AI Assistant Analysis",
                description="Intelligent query analysis and response generation",
                steps=4
            )
            
            # Step 1: Understand the User Request
            workflow_streamer.emit_step_started(
                workflow_id, session_id, "understand_request",
                "intent_analysis", "🧠 Understanding your request..."
            )
            
            analysis = await self._understand_user_request(user_query)
            
            workflow_streamer.emit_step_completed(
                workflow_id, session_id, "understand_request",
                "intent_analysis", "✅ Request understood"
            )
            
            # Step 2: Check if we need clarification or if it's outside capability  
            if analysis["status"] == "need_more_info":
                clarification_response = "I'd be happy to help! " + " ".join(analysis.get("clarification_needed", []))
                workflow_streamer.emit_workflow_completed(
                    workflow_id=workflow_id,
                    session_id=session_id,
                    final_answer=clarification_response,
                    execution_time=0.5
                )
                return {
                    "status": "success",
                    "greeting": "Hi there! How can I assist you today?",
                    "response": clarification_response,
                    "query_type": "clarification_needed"
                }
            
            if analysis["status"] == "outside_capability":
                capability_response = f"I understand you're asking about something, but {analysis.get('capability_reason', 'this request is outside my current capabilities.')} Is there something else I can help you with?"
                workflow_streamer.emit_workflow_completed(
                    workflow_id=workflow_id,
                    session_id=session_id,
                    final_answer=capability_response,
                    execution_time=0.5
                )
                return {
                    "status": "success", 
                    "greeting": "Hi there! How can I assist you today?",
                    "response": capability_response,
                    "query_type": "outside_capability"
                }
            
            # Step 3: Execute the plan if ready to proceed
            if analysis["status"] == "ready_to_proceed":
                plan = analysis.get("plan", {})
                
                workflow_streamer.emit_step_started(
                    workflow_id, session_id, "execute_plan",
                    "plan_execution", f"⚡ {plan.get('summary', 'Executing your request...')}"
                )
                
                execution_results = await self._execute_plan(plan, user_query, workflow_id, session_id)
                
                workflow_streamer.emit_step_completed(
                    workflow_id, session_id, "execute_plan",
                    "plan_execution", "✅ Execution completed"
                )
                
                # Step 4: Generate final response
                workflow_streamer.emit_step_started(
                    workflow_id, session_id, "generate_response",
                    "response_generation", "📝 Generating helpful response..."
                )
                
                final_response = await self._generate_helpful_response(user_query, execution_results, analysis)
                
                workflow_streamer.emit_step_completed(
                    workflow_id, session_id, "generate_response",
                    "response_generation", "✅ Response ready"
                )
                
                workflow_streamer.emit_workflow_completed(
                    workflow_id=workflow_id,
                    session_id=session_id,
                    final_answer=final_response,
                    execution_time=1.0
                )
                
                return {
                    "status": "success",
                    "greeting": analysis.get("greeting", "Hi there! How can I assist you today?"),
                    "response": final_response,
                    "plan_summary": plan.get("summary", ""),
                    "execution_type": plan.get("execution_type", "single_agent"),
                    "agents_used": [step.get("agent") for step in plan.get("steps", []) if step.get("agent") != "LLM"],
                    "query_type": "executed_successfully"
                }
            
            # Fallback
            fallback_response = "I'm here to help! What would you like to know?"
            workflow_streamer.emit_workflow_completed(
                workflow_id=workflow_id,
                session_id=session_id,
                final_answer=fallback_response,
                execution_time=0.2
            )
            return {
                "status": "success",
                "greeting": "Hi there! How can I assist you today?", 
                "response": fallback_response,
                "query_type": "general_greeting"
            }
            
        except Exception as e:
            print(f"❌ AI assistant workflow failed: {e}")
            error_response = "I apologize, but I encountered an issue processing your request. Could you please try rephrasing it?"
            workflow_streamer.emit_error(
                workflow_id, session_id, "workflow_error", str(e)
            )
            return {
                "status": "error",
                "greeting": "Hi there! How can I assist you today?",
                "response": error_response,
                "error": str(e)
            }
    
    async def _understand_user_request(self, user_query: str) -> Dict[str, Any]:
        """Step 1: Understand the User Request - Analyze intent and check if we have enough information."""
        from app.agents import fetch_agents_and_tools_from_registry
        
        # Get available capabilities
        agents = fetch_agents_and_tools_from_registry()
        agent_summaries = []
        
        for agent_id, agent in agents.items():
            if agent.get("agent_type") == "data_agent":
                agent_summaries.append({
                    "id": agent_id,
                    "name": agent.get("name", agent_id),
                    "type": "data",
                    "description": agent.get("description", ""),
                    "capabilities": "Query databases for business insights"
                })
            elif agent.get("agent_type") == "application":
                agent_summaries.append({
                    "id": agent_id,
                    "name": agent.get("name", agent_id), 
                    "type": "application",
                    "description": agent.get("description", ""),
                    "capabilities": list(agent.get("capabilities", {}).keys())[:3]
                })
        
        prompt = f"""
        You are an intelligent AI assistant that helps users by orchestrating a network of data and application agents.

        User Query: "{user_query}"
        
        Available Agents: {agent_summaries}
        
        Follow this process:
        
        1. **Understand the User Request**: Analyze the intent and check if you have enough information to proceed.
        2. **Check Capability**: Determine if the request is within system capabilities.
        3. **Plan if Ready**: If valid and clear, plan the agentic flow.
        
        Guidelines:
        - If greeting (hi, hello) → ready_to_proceed with greeting response
        - If vague ("help me", "information") → need_more_info with clarification
        - If general knowledge → ready_to_proceed with LLM response
        - If specific data request → ready_to_proceed with agent execution
        - If impossible request → outside_capability
        
        Respond with JSON exactly matching this structure:
        {{
            "greeting": "Hi there! How can I assist you today?",
            "status": "need_more_info" | "outside_capability" | "ready_to_proceed", 
            "clarification_needed": ["What specific data are you looking for?"],
            "capability_reason": "This requires real-time data we don't have access to",
            "plan": {{
                "summary": "I will query the inventory database to check stock levels",
                "execution_type": "single_agent" | "sequential" | "parallel" | "llm_only",
                "steps": [
                    {{
                        "step_id": "step1", 
                        "agent": "agent_id_or_LLM",
                        "action": "query inventory data",
                        "query": "refined query for the agent"
                    }}
                ]
            }}
        }}
        """
        
        try:
            response = self.llm_client.invoke_with_json_response(prompt, timeout=30)
            if response:
                print(f"✅ User request understood: {response.get('status')}")
                return response
        except Exception as e:
            print(f"❌ Request understanding failed: {e}")
        
        # Fallback analysis
        query_lower = user_query.lower().strip()
        if query_lower in ["hi", "hello", "hey"]:
            return {
                "greeting": "Hi there! How can I assist you today?",
                "status": "ready_to_proceed",
                "plan": {"execution_type": "llm_only", "steps": []}
            }
        elif len(user_query.split()) < 4:
            return {
                "greeting": "Hi there! How can I assist you today?", 
                "status": "need_more_info",
                "clarification_needed": ["Could you provide more details about what you're looking for?"]
            }
        else:
            return {
                "greeting": "Hi there! How can I assist you today?",
                "status": "ready_to_proceed", 
                "plan": {"execution_type": "llm_only", "steps": []}
            }
    
    async def _execute_plan(self, plan: Dict[str, Any], user_query: str, workflow_id: str, session_id: str) -> Dict[str, Any]:
        """Step 3: Execute the Plan - Coordinate agent calls and gather results."""
        execution_type = plan.get("execution_type", "llm_only")
        steps = plan.get("steps", [])
        
        # Emit execution plan display
        plan_summary = plan.get("summary", "Executing your request")
        agents_in_plan = [step.get("agent") for step in steps if step.get("agent") and step.get("agent") != "LLM"]
        
        if agents_in_plan:
            workflow_streamer.emit_step_started(
                workflow_id, session_id, "execution_plan",
                "execution_plan", f"📋 Execution Plan: {plan_summary}"
            )
            
            plan_details = f"Strategy: {execution_type}, Agents: {', '.join(agents_in_plan)}"
            workflow_streamer.emit_step_completed(
                workflow_id, session_id, "execution_plan",
                "execution_plan", f"✅ Plan Ready: {plan_details}"
            )
        
        results = {}
        
        if execution_type == "llm_only" or not steps:
            # Handle with LLM only - greetings, general knowledge, etc.
            results["llm_response"] = await self._generate_llm_response(user_query)
            return results
        
        if execution_type == "single_agent":
            # Execute single agent
            step = steps[0] if steps else {}
            agent_id = step.get("agent")
            query = step.get("query", user_query)
            
            if agent_id and agent_id != "LLM":
                result = await self._execute_agent_simple(agent_id, query)
                results[f"step_{step.get('step_id', '1')}"] = result
        
        elif execution_type == "sequential":
            # Execute agents in sequence
            for step in steps:
                agent_id = step.get("agent")
                if agent_id and agent_id != "LLM":
                    query = step.get("query", user_query)
                    result = await self._execute_agent_simple(agent_id, query)
                    results[f"step_{step.get('step_id')}"] = result
        
        elif execution_type == "parallel":
            # Execute agents in parallel
            tasks = []
            step_ids = []
            for step in steps:
                agent_id = step.get("agent")
                if agent_id and agent_id != "LLM":
                    query = step.get("query", user_query)
                    task = self._execute_agent_simple(agent_id, query)
                    tasks.append(task)
                    step_ids.append(step.get('step_id'))
            
            if tasks:
                parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(parallel_results):
                    if not isinstance(result, Exception):
                        results[f"step_{step_ids[i]}"] = result
        
        # Emit agents used summary if any agents were executed
        executed_agents = []
        total_records = 0
        total_time = 0.0
        
        for result_key, result_data in results.items():
            if isinstance(result_data, dict) and result_data.get("agent_name"):
                agent_name = result_data.get("agent_name", "Unknown")
                row_count = result_data.get("row_count", 0)
                exec_time = result_data.get("execution_time", 0)
                executed_agents.append(agent_name)
                total_records += row_count
                total_time += exec_time
        
        if executed_agents:
            workflow_streamer.emit_step_started(
                workflow_id, session_id, "agents_summary",
                "agents_summary", "📊 Summarizing Agent Results..."
            )
            
            agents_list = ", ".join(set(executed_agents))  # Remove duplicates
            summary_text = f"✅ Agents Used: {agents_list} | Total: {total_records} records in {total_time:.1f}s"
            
            workflow_streamer.emit_step_completed(
                workflow_id, session_id, "agents_summary",
                "agents_summary", summary_text
            )
        
        return results
    
    async def _execute_agent_simple(self, agent_id: str, query: str) -> Dict[str, Any]:
        """Simplified agent execution."""
        from app.agents import fetch_agents_and_tools_from_registry
        
        try:
            # Get agent details
            agents = fetch_agents_and_tools_from_registry()
            agent = agents.get(agent_id)
            
            if not agent:
                return {"success": False, "error": f"Agent {agent_id} not found"}
            
            agent_type = agent.get("agent_type", "unknown")
            
            if agent_type == "data_agent":
                result = await self._execute_data_agent(agent_id, query, None)
                return {
                    "success": True,
                    "agent_name": agent.get("name", agent_id),
                    "data": result.get("data"),
                    "row_count": result.get("row_count", 0),
                    "query_executed": result.get("query")
                }
            else:
                return {
                    "success": True,
                    "agent_name": agent.get("name", agent_id),
                    "data": {"message": f"Simulated execution for {query}"},
                    "row_count": 1
                }
                
        except Exception as e:
            print(f"❌ Agent {agent_id} execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_llm_response(self, user_query: str) -> str:
        """Generate LLM response for general queries, greetings, etc."""
        try:
            prompt = f"""
            You are a helpful AI assistant. The user said: "{user_query}"
            
            Provide a natural, friendly response. If it's:
            - A greeting: Respond warmly and mention you can help with data analysis
            - General knowledge: Answer directly
            - Vague request: This shouldn't happen (should be caught earlier)
            
            Keep it conversational and helpful.
            """
            
            response = self.llm_client.invoke_with_text_response(
                prompt, 
                task_type=TaskType.GENERAL
            )
            return response if response else "I'm here to help! What would you like to know?"
            
        except Exception as e:
            print(f"❌ LLM response generation failed: {e}")
            return "I'm here to help! What would you like to know?"
    
    async def _generate_helpful_response(self, user_query: str, execution_results: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Step 4: Respond Clearly - Present final answer in friendly, understandable format."""
        try:
            # Check if we have data results
            data_results = []
            for key, result in execution_results.items():
                if key != "llm_response" and result.get("success") and result.get("data"):
                    data_results.append(result)
            
            # If LLM-only response
            if "llm_response" in execution_results:
                return execution_results["llm_response"]
            
            # If we have data results, create business analysis
            if data_results:
                # Prepare data summary for LLM
                data_summary = []
                total_records = 0
                
                for result in data_results:
                    agent_name = result.get("agent_name", "Database")
                    row_count = result.get("row_count", 0)
                    total_records += row_count
                    
                    data_preview = str(result.get("data", []))[:300]
                    data_summary.append(f"{agent_name}: {row_count} records - {data_preview}...")
                
                prompt = f"""
                The user asked: "{user_query}"
                
                Plan executed: {analysis.get('plan', {}).get('summary', 'Data retrieval completed')}
                
                Data retrieved:
                {chr(10).join(data_summary)}
                
                Total records: {total_records}
                
                Task: Provide a clear, business-focused analysis of this data. Include:
                1. Direct answer to their question
                2. Key insights and patterns
                3. Specific recommendations if relevant
                4. Ask if they need further help
                
                Be natural, conversational, and helpful. Focus on business value.
                """
                
                response = self.llm_client.invoke_with_text_response(prompt)
                return response if response else f"I found {total_records} records that match your request. Would you like me to analyze specific aspects of this data?"
            
            # No data found
            return "I wasn't able to find data for your request. Could you provide more specific details about what you're looking for?"
            
        except Exception as e:
            print(f"❌ Response generation failed: {e}")
            return "I've processed your request. Is there anything specific you'd like to know more about?"
        """Step 1: Analyze user intent and refine query."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "analyze_intent",
            "intent_analysis", "🧠 Analyzing user intent and planning approach..."
        )
        
        # Get conversation context if available
        conversation_context = self.session_manager.get_conversation_context_for_llm(context.session_id)
        
        # Prepare conversation history for LLM
        conversation_history_text = ""
        if conversation_context.get('has_context'):
            conversation_history_text = "CONVERSATION CONTEXT:\n"
            
            # Add conversation summary if available
            if conversation_context.get('conversation_summary'):
                conversation_history_text += f"Previous Discussion Summary: {conversation_context['conversation_summary']}\n"
            
            # Add recent turns
            if conversation_context.get('recent_turns'):
                conversation_history_text += f"Recent Conversation Turns (last {len(conversation_context['recent_turns'])}):\n"
                for turn in conversation_context['recent_turns'][-3:]:  # Last 3 turns for context
                    turn_text = f"- Turn {turn['turn_number']}: '{turn['query']}'"
                    if turn['agents_used']:
                        turn_text += f" → Used: {', '.join(turn['agents_used'][:2])}"
                    conversation_history_text += turn_text + "\n"
                
                conversation_history_text += f"Total conversations: {conversation_context.get('total_conversations', 0)}\n\n"
        
        # Get available agents for context
        agents = fetch_agents_and_tools_from_registry()
        agent_summaries = []
        
        for agent_id, agent in agents.items():
            if agent.get("agent_type") == "data_agent":
                schema_summary = self.schema_introspector.get_schema_summary(agent_id)
                agent_summaries.append({
                    "id": agent_id,
                    "name": agent.get("name", agent_id),
                    "type": "data",
                    "database_type": agent.get("connection_type", "unknown"),
                    "schema_preview": schema_summary[:200] + "..." if len(schema_summary) > 200 else schema_summary
                })
            elif agent.get("agent_type") == "application":
                capabilities = list(agent.get("capabilities", {}).keys())
                agent_summaries.append({
                    "id": agent_id,
                    "name": agent.get("name", agent_id),
                    "type": "application",
                    "capabilities": capabilities[:5]  # Limit for prompt size
                })
        
        # SIMPLE APPROACH - check for vague queries first before LLM analysis
        vague_patterns = [
            "can i get", "help me", "what can you", "tell me about", 
            "information", "some data", "anything about", "i need help",
            "show me", "find me", "give me", "get me", "help with"
        ]
        
        query_lower = context.original_query.lower().strip()
        is_vague = any(pattern in query_lower for pattern in vague_patterns) and len(context.original_query.split()) < 8
        
        # Handle obviously vague queries immediately
        if query_lower in ["hi", "hello", "hey", "help"] or is_vague:
            print(f"📝 Detected vague query - setting to general_knowledge: {context.original_query}")
            context.execution_strategy = "general_knowledge"
            context.primary_agent_id = None
            context.additional_agent_ids = []
            context.refined_query = context.original_query
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, "analyze_intent",
                "intent_analysis", "✅ Identified as conversational query"
            )
            return
        
        prompt = f"""
        You are analyzing a user's query to decide how to help them.
        
        User Query: "{context.original_query}"
        
        Available Agents:
        {json.dumps(agent_summaries, indent=2)}
        
        SIMPLE RULES:
        1. If the query is vague/unclear -> ALWAYS set execution_strategy to "general_knowledge" and primary_agent_id to null
        2. If it's a specific data request -> pick the best agent  
        3. If it's general knowledge -> set execution_strategy to "general_knowledge"
        4. For greetings -> set execution_strategy to "general_knowledge"
        
        Examples:
        - "can I get information?" -> general_knowledge, null agent
        - "help me" -> general_knowledge, null agent  
        - "show me inventory" -> single_agent, pick inventory agent
        - "what is AI?" -> general_knowledge, null agent
        
        Respond with JSON:
        {{
            "intent_category": "general_knowledge|data_analysis",
            "refined_query": "cleaner version of the user query",
            "execution_strategy": "general_knowledge|single_agent|parallel_agents",
            "primary_agent_id": "select agent ID or null",
            "additional_agent_ids": [],
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self.llm_client.invoke_with_json_response(
                prompt, 
                timeout=30, 
                task_type=TaskType.INTENT_ANALYSIS
            )
            if response:
                context.refined_query = response.get("refined_query", context.original_query)
                context.requires_follow_up = response.get("requires_multi_step", False)
                context.suggested_follow_ups = response.get("expected_follow_ups", [])
                
                # Enhanced multi-agent support with better defaults
                context.execution_strategy = response.get("execution_strategy", "single_agent")
                context.primary_agent_id = response.get("primary_agent_id")
                context.additional_agent_ids = response.get("additional_agent_ids", [])
                context.agent_coordination = response.get("agent_coordination", "independent")
                
                # Debug: Log the full LLM response for troubleshooting
                print(f"🔍 LLM Response Keys: {list(response.keys())}")
                print(f"🔍 Execution Strategy from LLM: '{response.get('execution_strategy', 'NOT_SET')}'")
                print(f"🔍 Additional Agents from LLM: {response.get('additional_agent_ids', 'NOT_SET')}")
                
                print(f"✅ Intent analyzed: {response.get('intent_category')} -> Strategy: {context.execution_strategy}")
                print(f"🎯 Primary: {context.primary_agent_id}, Additional: {context.additional_agent_ids}")
                
                # Validate multi-agent setup
                if context.execution_strategy == "parallel_agents" and not context.additional_agent_ids:
                    print("⚠️ WARNING: parallel_agents strategy set but no additional_agent_ids provided")
                
                # Store planning result
                context.conversation_history.append({
                    "step": "intent_analysis",
                    "result": response,
                    "timestamp": datetime.now().isoformat()
                })
            
        except Exception as e:
            print(f"❌ Intent analysis failed: {e}")
            context.refined_query = context.original_query
            context.execution_strategy = "single_agent"
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "analyze_intent",
            "intent_analysis", 1.0
        )
    
    async def _step_check_patterns(self, context: ConversationContext):
        """Step 2: Check for similar successful query patterns."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "check_patterns",
            "pattern_matching", "🔍 Checking for similar successful patterns..."
        )
        
        similar_queries = self.feedback_manager.get_similar_queries(context.original_query)
        
        if similar_queries:
            print(f"📚 Found {len(similar_queries)} similar successful queries")
            context.conversation_history.append({
                "step": "pattern_matching",
                "similar_patterns": len(similar_queries),
                "top_pattern": similar_queries[0] if similar_queries else None,
                "timestamp": datetime.now().isoformat()
            })
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "check_patterns",
            "pattern_matching", 0.5
        )
    
    async def _step_execute_primary_agent(self, context: ConversationContext):
        """Step 3: Execute the primary agent."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "execute_primary",
            "primary_execution", "⚡ Executing primary agent..."
        )
        
        # Get the recommended agent from intent analysis
        planning_result = None
        for entry in context.conversation_history:
            if entry.get("step") == "intent_analysis":
                planning_result = entry.get("result")
                break
        
        if not planning_result:
            raise Exception("No planning result found")
        
        agent_id = planning_result.get("primary_agent_id")
        if not agent_id:
            raise Exception("No primary agent identified")
        
        # Execute the agent
        result = await self._execute_agent(agent_id, context.refined_query, context)
        context.agent_results.append(result)
        
        print(f"✅ Primary agent executed: {result.agent_name}")
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "execute_primary",
            "primary_execution", 2.0
        )
    
    async def _step_execute_parallel_agents(self, context: ConversationContext):
        """Execute multiple agents in parallel for cross-database analysis."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "execute_parallel",
            "parallel_execution", "⚡ Executing multiple agents in parallel..."
        )
        
        # Get all agents to execute
        all_agent_ids = [context.primary_agent_id] + context.additional_agent_ids
        
        print(f"🚀 Starting parallel execution of {len(all_agent_ids)} agents: {all_agent_ids}")
        
        # Create tasks for parallel execution
        tasks = []
        for agent_id in all_agent_ids:
            if agent_id:
                task = self._execute_agent(agent_id, context.refined_query, context)
                tasks.append(task)
        
        # Execute all agents in parallel
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"❌ Agent {all_agent_ids[i]} failed: {result}")
                    # Create error result
                    error_result = EnhancedAgentResult(
                        success=False,
                        data=None,
                        agent_id=all_agent_ids[i],
                        agent_name=f"Agent {all_agent_ids[i]}",
                        agent_type="data_agent",
                        execution_time=0.0,
                        error=str(result)
                    )
                    context.agent_results.append(error_result)
                else:
                    print(f"✅ Agent {result.agent_id} completed successfully")
                    context.agent_results.append(result)
        
        except Exception as e:
            print(f"❌ Parallel execution failed: {e}")
            raise
        
        print(f"🎯 Parallel execution completed. {len(context.agent_results)} results obtained.")
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "execute_parallel",
            "parallel_execution", 3.0
        )
    
    async def _step_execute_sequential_agents(self, context: ConversationContext):
        """Execute multiple agents sequentially with dependency handling."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "execute_sequential",
            "sequential_execution", "🔗 Executing agents sequentially..."
        )
        
        # Execute primary agent first
        if context.primary_agent_id:
            result = await self._execute_agent(context.primary_agent_id, context.refined_query, context)
            context.agent_results.append(result)
            print(f"✅ Primary agent executed: {result.agent_name}")
        
        # Execute additional agents with context from previous results
        for agent_id in context.additional_agent_ids:
            if agent_id:
                # Modify query based on previous results if needed
                enhanced_query = self._enhance_query_with_context(context.refined_query, context.agent_results)
                result = await self._execute_agent(agent_id, enhanced_query, context)
                context.agent_results.append(result)
                print(f"✅ Sequential agent executed: {result.agent_name}")
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "execute_sequential",
            "sequential_execution", 3.0
        )
    
    def _enhance_query_with_context(self, original_query: str, previous_results: List[EnhancedAgentResult]) -> str:
        """Enhance query with context from previous agent results for sequential execution."""
        if not previous_results:
            return original_query
        
        # Get the most recent successful result
        latest_result = None
        for result in reversed(previous_results):
            if result.success and result.data:
                latest_result = result
                break
        
        if not latest_result:
            return original_query
        
        # Use LLM to enhance the query based on previous results
        try:
            enhancement_prompt = f"""
            You are helping to create a follow-up query based on previous results in a sequential multi-agent workflow.
            
            Original Query: "{original_query}"
            
            Previous Agent: {latest_result.agent_name}
            Previous Results Summary: {str(latest_result.data)[:500]}...
            Row Count: {latest_result.row_count}
            
            Task: Create an enhanced version of the original query that takes into account the previous results.
            
            Examples of enhancements:
            - If previous results showed low stock items, enhance query to focus on those specific items
            - If previous results revealed patterns, refine query to investigate those patterns
            - If previous results were empty, adjust query to be more general
            
            Return only the enhanced query text, nothing else:
            """
            
            response = self.llm_client.invoke(enhancement_prompt, timeout=15)
            enhanced_query = response.content.strip()
            
            print(f"🔄 Enhanced query for sequential execution: {enhanced_query}")
            return enhanced_query
            
        except Exception as e:
            print(f"⚠️ Query enhancement failed, using original: {e}")
            return original_query
    
    async def _step_analyze_and_plan_followup(self, context: ConversationContext):
        """Step 4: Analyze results and determine if follow-up is needed."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "analyze_results",
            "result_analysis", "📊 Analyzing results and planning follow-ups..."
        )
        
        if not context.agent_results:
            return
        
        # Handle multi-agent result analysis
        if len(context.agent_results) > 1:
            combined_analysis = await self._analyze_multi_agent_results(context)
            
            # Create a combined insight summary
            all_summaries = []
            for result in context.agent_results:
                if result.success and result.data:
                    all_summaries.append(f"{result.agent_name}: {result.insight_summary or 'Data retrieved successfully'}")
            
            # Update the last result with combined analysis
            if context.agent_results:
                context.agent_results[-1].insight_summary = combined_analysis.get("combined_summary", "Multi-agent analysis completed")
                context.requires_follow_up = not combined_analysis.get("question_fully_answered", True)
                context.suggested_follow_ups.extend(combined_analysis.get("follow_up_questions", []))
        else:
            # Single agent analysis
            latest_result = context.agent_results[-1]
            
            # Skip complex analysis for general knowledge questions
            if context.execution_strategy == "general_knowledge":
                latest_result.insight_summary = "General knowledge response provided"
                context.requires_follow_up = False
                context.suggested_follow_ups = []
            else:
                # Data analysis logic
                analysis = self.result_analyzer.analyze_result(context, latest_result.data)
                latest_result.insight_summary = analysis.get("summary", "Analysis completed")
                context.requires_follow_up = not analysis.get("question_fully_answered", True)
                context.suggested_follow_ups.extend(analysis.get("follow_up_questions", []))
        
        # Store analysis
        context.conversation_history.append({
            "step": "result_analysis",
            "multi_agent": len(context.agent_results) > 1,
            "agent_count": len(context.agent_results),
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"📊 Analysis complete. Multi-agent: {len(context.agent_results) > 1}, Follow-up needed: {context.requires_follow_up}")
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "analyze_results",
            "result_analysis", 1.0
        )
    
    async def _analyze_multi_agent_results(self, context: ConversationContext) -> Dict[str, Any]:
        """Analyze and combine results from multiple agents."""
        successful_results = [r for r in context.agent_results if r.success and r.data]
        
        if not successful_results:
            return {"combined_summary": "No successful results to analyze", "question_fully_answered": False}
        
        # Prepare data for LLM analysis
        agent_summaries = []
        for result in successful_results:
            summary = {
                "agent_name": result.agent_name,
                "agent_type": result.agent_type,
                "data_preview": str(result.data)[:500] + "..." if len(str(result.data)) > 500 else str(result.data),
                "row_count": result.row_count or 0,
                "execution_time": result.execution_time
            }
            agent_summaries.append(summary)
        
        # Use LLM to create comprehensive analysis
        prompt = f"""
        Analyze the combined results from multiple data sources to answer this query:
        "{context.original_query}"
        
        Results from {len(successful_results)} agents:
        {json.dumps(agent_summaries, indent=2)}
        
        Tasks:
        1. Compare and contrast the data from different sources
        2. Identify key insights and patterns across all sources
        3. Provide specific recommendations (especially for inventory/supplier analysis)
        4. Determine if the question is fully answered
        5. Suggest follow-up questions if needed
        
        For inventory analysis specifically:
        - Identify items needing immediate attention across all databases
        - Compare stock levels between systems
        - Provide supplier consultation recommendations
        - Highlight discrepancies or concerns
        
        Respond with JSON:
        {{
            "combined_summary": "comprehensive analysis combining all data sources",
            "key_insights": ["insight1", "insight2"],
            "cross_database_comparison": "comparison findings",
            "priority_items": ["item1", "item2", "item3", "item4", "item5"],
            "supplier_recommendations": ["rec1", "rec2"],
            "question_fully_answered": true|false,
            "follow_up_questions": ["question1", "question2"],
            "confidence": 0.95
        }}
        """
        
        try:
            response = self.llm_client.invoke_with_json_response(prompt, timeout=30)
            return response or {"combined_summary": "Analysis completed", "question_fully_answered": True}
        except Exception as e:
            print(f"❌ Multi-agent analysis failed: {e}")
            return {"combined_summary": f"Analysis completed with {len(successful_results)} data sources", "question_fully_answered": True}
    
    async def _step_execute_followup(self, context: ConversationContext):
        """Step 5: Execute follow-up agent if needed."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "execute_followup",
            "followup_execution", "🔄 Executing follow-up analysis..."
        )
        
        # For demo purposes, we'll execute a simple follow-up
        # In practice, this would involve more sophisticated planning
        
        if context.suggested_follow_ups:
            # Take the first suggested follow-up
            followup_query = context.suggested_follow_ups[0]
            
            # Try to use the same agent for follow-up
            if context.agent_results:
                previous_agent_id = context.agent_results[-1].agent_id
                result = await self._execute_agent(previous_agent_id, followup_query, context)
                context.agent_results.append(result)
                
                print(f"✅ Follow-up executed: {result.agent_name}")
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "execute_followup",
            "followup_execution", 1.0
        )

    async def _handle_follow_up_request(self, user_query: str, session_id: str, follow_up_info: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle follow-up requests by continuing the previous conversation context."""
        workflow_id = str(uuid.uuid4())
        print(f"🔄 Handling follow-up request: {workflow_id}")
        
        # Create context from previous session
        context = ConversationContext(
            session_id=session_id,
            workflow_id=workflow_id,
            original_query=user_query
        )
        
        # Restore previous agent results for context
        previous_results = session_context.get('agent_results', [])
        for result_data in previous_results:
            if result_data.get('success', False):
                context.agent_results.append(EnhancedAgentResult(
                    success=result_data['success'],
                    data=result_data.get('data'),
                    agent_id=result_data.get('agent_id', 'unknown'),
                    agent_name=result_data.get('agent_name', 'Unknown Agent'),
                    agent_type=result_data.get('agent_type', 'unknown'),
                    execution_time=result_data.get('execution_time', 0.0),
                    query_executed=result_data.get('query_executed'),
                    row_count=result_data.get('row_count'),
                    insight_summary=result_data.get('insight_summary')
                ))
        
        try:
            # Emit workflow started for follow-up
            workflow_streamer.emit_workflow_started(
                workflow_id=workflow_id,
                session_id=session_id,
                title="Follow-up Analysis",
                description=f"Continuing previous conversation with: {user_query}",
                steps=3
            )
            
            # Use LLM to understand the user's follow-up intent and plan next action
            await self._step_plan_followup_action(context, session_context, follow_up_info)
            
            # Execute the planned follow-up action
            await self._step_execute_planned_followup(context)
            
            # Analyze and finalize results
            await self._step_analyze_and_plan_followup(context)
            
            # Update session context
            self.session_manager.update_session_context(session_id, context)
            
            # Emit completion
            execution_time = time.time() - context.start_time
            final_response = self._format_final_response(context)
            
            workflow_streamer.emit_workflow_completed(
                workflow_id=workflow_id,
                session_id=session_id,
                final_answer=final_response.get("final_answer", "Follow-up completed"),
                execution_time=execution_time
            )
            
            return final_response
            
        except Exception as e:
            print(f"❌ Follow-up workflow failed: {e}")
            workflow_streamer.emit_error(
                workflow_id, session_id, "followup_error", f"Follow-up failed: {str(e)}"
            )
            return {
                "status": "error",
                "error": str(e),
                "message": "Sorry, I couldn't help you with your request"
            }

    async def _step_plan_followup_action(self, context: ConversationContext, session_context: Dict[str, Any], follow_up_info: Dict[str, Any]):
        """Plan the follow-up action based on user input and previous context."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "plan_followup",
            "followup_planning", "🧠 Understanding your follow-up request..."
        )
        
        # Get enhanced conversation context
        conversation_context = self.session_manager.get_conversation_context_for_llm(context.session_id)
        
        # Prepare conversation history for LLM
        conversation_history_text = ""
        if conversation_context.get('has_context'):
            # Add conversation summary if available
            if conversation_context.get('conversation_summary'):
                conversation_history_text += f"CONVERSATION SUMMARY:\n{conversation_context['conversation_summary']}\n\n"
            
            # Add recent turns
            if conversation_context.get('recent_turns'):
                conversation_history_text += "RECENT CONVERSATION TURNS:\n"
                for turn in conversation_context['recent_turns']:
                    turn_text = f"Turn {turn['turn_number']}: User asked '{turn['query']}'"
                    if turn['agents_used']:
                        turn_text += f" | Agents: {', '.join(turn['agents_used'][:2])}"
                    if turn['key_insights']:
                        turn_text += f" | Insights: {' | '.join(turn['key_insights'][:1])}"
                    conversation_history_text += turn_text + "\n"
                conversation_history_text += f"\nTotal conversations in session: {conversation_context.get('total_conversations', 0)}\n"
        
        # Get available agents for context
        agents = fetch_agents_and_tools_from_registry()
        agent_summaries = []
        
        for agent_id, agent in agents.items():
            if agent.get("agent_type") == "data_agent":
                agent_summaries.append({
                    "id": agent_id,
                    "name": agent.get("name", agent_id),
                    "type": "data",
                    "database_type": agent.get("connection_type", "unknown")
                })
        
        prompt = f"""
        You are analyzing a follow-up request in an ongoing conversation. The user has provided additional input and you need to determine the best way to help them.

        {conversation_history_text}

        CURRENT USER INPUT: "{context.original_query}"

        AVAILABLE AGENTS:
        {json.dumps(agent_summaries, indent=2)}

        FOLLOW-UP CONTEXT:
        - Type: {follow_up_info.get('follow_up_type', 'general')}
        - Previous Agent: {follow_up_info.get('previous_agent', 'Unknown')}

        ANALYSIS NEEDED:
        1. What is the user asking for in their follow-up?
        2. Should we continue with the same agent(s) or use different ones?
        3. Is this a request for more detail, different analysis, or something new?
        4. How does this request relate to the conversation history?

        Consider the full conversation context when planning your response. If the user is asking for something related to previous discussions, leverage that context.

        Respond with JSON:
        {{
            "follow_up_type": "continue_analysis|new_analysis|clarification|none",
            "user_intent": "clear description of what the user wants",
            "recommended_action": "use_previous_agent|use_different_agent|provide_explanation|stop",
            "agent_id": "specific agent to use (if applicable)",
            "refined_query": "refined version of the user's request for execution",
            "reasoning": "why this approach was chosen, considering conversation history",
            "confidence": 0.1-1.0,
            "uses_conversation_context": true|false
        }}
        """
        
        try:
            response = self.llm_client.invoke_with_json_response(prompt, timeout=30)
            if response:
                context.follow_up_plan = response
                context.refined_query = response.get("refined_query", context.original_query)
                
                print(f"✅ Follow-up planned: {response.get('follow_up_type')} -> {response.get('recommended_action')}")
        
        except Exception as e:
            print(f"❌ Follow-up planning failed: {e}")
            # Default fallback plan
            context.follow_up_plan = {
                "follow_up_type": "continue_analysis",
                "recommended_action": "use_previous_agent",
                "refined_query": context.original_query,
                "confidence": 0.5
            }
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "plan_followup",
            "followup_planning", 1.0
        )

    async def _step_execute_planned_followup(self, context: ConversationContext):
        """Execute the planned follow-up action."""
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, "execute_planned_followup",
            "followup_execution", "⚡ Executing your follow-up request..."
        )
        
        plan = getattr(context, 'follow_up_plan', {})
        action = plan.get('recommended_action', 'provide_explanation')
        
        if action == "none" or plan.get('follow_up_type') == 'none':
            # User doesn't want to continue
            context.agent_results.append(EnhancedAgentResult(
                success=True,
                data={"message": "I understand you don't want to continue. Thank you for using our service!"},
                agent_id="system",
                agent_name="System",
                agent_type="system",
                execution_time=0.1,
                insight_summary="User chose not to continue"
            ))
            
        elif action == "provide_explanation":
            # Provide explanation based on previous results
            explanation = self._generate_explanation_from_context(context)
            context.agent_results.append(EnhancedAgentResult(
                success=True,
                data={"explanation": explanation},
                agent_id="system",
                agent_name="System",
                agent_type="system",
                execution_time=0.1,
                insight_summary="Provided explanation based on previous analysis"
            ))
            
        elif action in ["use_previous_agent", "use_different_agent"]:
            # Execute agent with the refined query
            agent_id = plan.get('agent_id')
            if not agent_id and context.agent_results:
                # Use the last successful agent
                agent_id = context.agent_results[-1].agent_id
            
            if agent_id:
                result = await self._execute_agent(agent_id, context.refined_query, context)
                context.agent_results.append(result)
                print(f"✅ Follow-up executed: {result.agent_name}")
        
        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, "execute_planned_followup",
            "followup_execution", 2.0
        )

    def _generate_explanation_from_context(self, context: ConversationContext) -> str:
        """Generate an explanation based on previous context and current request."""
        if not context.agent_results:
            return "I don't have previous results to explain. Could you please clarify what you'd like to know?"
        
        # Use the latest successful result for explanation
        latest_result = None
        for result in reversed(context.agent_results):
            if result.success and result.data:
                latest_result = result
                break
        
        if not latest_result:
            return "I don't have successful results to explain. Could you please rephrase your request?"
        
        try:
            # Use LLM to generate explanation
            explanation_prompt = f"""
            The user is asking for clarification about previous analysis results. Provide a clear, helpful explanation.

            User's Current Request: "{context.original_query}"
            
            Previous Analysis:
            - Agent Used: {latest_result.agent_name}
            - Query Executed: {latest_result.query_executed or 'N/A'}
            - Results Summary: {latest_result.insight_summary or 'Data retrieved successfully'}
            - Row Count: {latest_result.row_count or 'Unknown'}

            Provide a clear, conversational explanation that addresses the user's request based on the previous analysis.
            """
            
            response = self.llm_client.invoke(explanation_prompt, timeout=15)
            return response.content.strip()
            
        except Exception as e:
            print(f"⚠️ Explanation generation failed: {e}")
            return f"Based on the previous analysis using {latest_result.agent_name}, we found {latest_result.row_count or 'some'} results. The analysis showed: {latest_result.insight_summary or 'data was retrieved successfully'}."
    
    async def _execute_agent(self, agent_id: str, query: str, context: ConversationContext) -> EnhancedAgentResult:
        """Execute a specific agent with enhanced result tracking."""
        start_time = time.time()
        
        try:
            # Get agent details
            agents = fetch_agents_and_tools_from_registry()
            agent = agents.get(agent_id)
            
            if not agent:
                raise Exception(f"Agent {agent_id} not found")
            
            agent_type = agent.get("agent_type", "unknown")
            agent_name = agent.get("name", agent_id)
            
            # Emit detailed step: Selected Agent
            workflow_streamer.emit_step_started(
                context.workflow_id, context.session_id, f"agent_selected_{agent_id}",
                "agent_selection", f"🎯 Selected Agent: {agent_name}"
            )
            
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"agent_selected_{agent_id}",
                "agent_selection", f"✅ Using {agent_name} ({agent_type})"
            )
            
            # Execute based on agent type
            if agent_type == "data_agent":
                result_data = await self._execute_data_agent(agent_id, query, context)
            elif agent_type == "application":
                result_data = await self._execute_application_agent(agent_id, query, context)
            else:
                raise Exception(f"Unknown agent type: {agent_type}")
            
            execution_time = time.time() - start_time
            
            # Emit completion with agent summary
            workflow_streamer.emit_step_started(
                context.workflow_id, context.session_id, f"agent_summary_{agent_id}",
                "execution_summary", f"📊 Agent Execution Summary"
            )
            
            row_count = result_data.get("row_count", 0)
            query_executed = result_data.get("query", "")
            summary_msg = f"✅ {agent_name} completed: {row_count} records in {execution_time:.1f}s"
            
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"agent_summary_{agent_id}",
                "execution_summary", summary_msg
            )
            
            return EnhancedAgentResult(
                success=True,
                data=result_data.get("data"),
                agent_id=agent_id,
                agent_name=agent_name,
                agent_type=agent_type,
                execution_time=execution_time,
                query_executed=result_data.get("query"),
                row_count=result_data.get("row_count")
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Agent execution failed: {e}")
            
            # Emit failure step
            workflow_streamer.emit_step_started(
                context.workflow_id, context.session_id, f"agent_failed_{agent_id}",
                "execution_error", f"❌ Agent {agent.get('name', agent_id)} Failed"
            )
            
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"agent_failed_{agent_id}",
                "execution_error", f"❌ Error: {str(e)}"
            )
            
            return EnhancedAgentResult(
                success=False,
                data=None,
                agent_id=agent_id,
                agent_name=agent.get("name", agent_id) if 'agent' in locals() else agent_id,
                agent_type=agent.get("agent_type", "unknown") if 'agent' in locals() else "unknown",
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _execute_data_agent(self, agent_id: str, query: str, context: ConversationContext) -> Dict[str, Any]:
        """Execute a data agent with enhanced SQL generation."""
        # Get agent details with schema
        agent_details = get_enhanced_agent_details_for_llm(agent_id)
        if not agent_details:
            raise Exception(f"Agent details not found for {agent_id}")

        agent_name = agent_details.get('name', agent_id)
        
        # Emit step: Preparing agent execution
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, f"prepare_agent_{agent_id}",
            "agent_preparation", f"🔧 Preparing {agent_name} for execution..."
        )

        # Get fresh schema
        schema_summary = self.schema_introspector.get_schema_summary(agent_id)

        # Defensive: Ensure connection_type is present
        connection_type = agent_details.get("connection_type")
        if not connection_type:
            # Fallback to database_type if present
            connection_type = agent_details.get("database_type")
            print(f"[Orchestrator] WARNING: connection_type missing for agent {agent_id}, using database_type: {connection_type}")
        if not connection_type or connection_type == "unknown":
            raise Exception(f"Agent {agent_id} is missing a valid connection_type/database_type in registry cache.")

        workflow_streamer.emit_step_completed(
            context.workflow_id, context.session_id, f"prepare_agent_{agent_id}",
            "agent_preparation", f"✅ {agent_name} ready ({connection_type} database)"
        )

        # Emit step: Generating SQL Query
        workflow_streamer.emit_step_started(
            context.workflow_id, context.session_id, f"sql_generation_{agent_id}",
            "sql_generation", f"🧠 Generating SQL Query for {agent_name}..."
        )

        # Generate SQL using LLM with enhanced schema qualification
        # Get the full tables structure with columns data
        tables_data = agent_details.get('tables_with_columns', [])
        print(f"[DEBUG] Orchestrator: Agent {agent_id} has {len(tables_data)} tables")
        
        # Debug: Check if tables have columns
        for i, table in enumerate(tables_data[:3]):  # Check first 3 tables
            table_name = table.get("tableName", "unknown")
            columns = table.get("columns", [])
            print(f"[DEBUG] Orchestrator: Table '{table_name}' has {len(columns)} columns")
        
        
        print(f"[DEBUG] 🔍 ORCHESTRATOR SQL GENERATION DEBUG:")
        print(f"[DEBUG] Agent ID: {agent_id}")
        print(f"[DEBUG] Agent Details connection_type: {agent_details.get('connection_type')}")
        print(f"[DEBUG] Agent Details database_type: {agent_details.get('database_type')}")
        print(f"[DEBUG] Final connection_type being passed: {connection_type}")
        
        # Extract custom prompt from environment section
        environment_details = agent_details.get("environment", {})
        custom_prompt = environment_details.get("customPrompt", "")
        if custom_prompt:
            print(f"[DEBUG] Using custom prompt guidelines from environment")
        
        sql_prompt = self.llm_client.create_sql_generation_prompt(
            user_query=query,
            schema_context=schema_summary,
            connection_type=connection_type,
            tables_data=tables_data,
            agent_name=agent_details.get('name', agent_id),
            custom_prompt=custom_prompt
        )

        try:
            sql_response = self.llm_client.invoke_with_json_response(
                sql_prompt, 
                timeout=60, 
                task_type=TaskType.SQL_GENERATION
            )
            
            # Extract SQL from JSON response
            if sql_response and "query" in sql_response:
                sql_query = sql_response["query"].strip()
                print(f"[DEBUG] Orchestrator: LLM generated SQL: {sql_query}")
            else:
                # Fallback: try to extract from content if JSON parsing failed
                fallback_response = self.llm_client.invoke(sql_prompt, timeout=60)
                sql_query = fallback_response.content.strip()
                sql_query = self._clean_sql_query(sql_query)
                print(f"[DEBUG] Orchestrator: Fallback SQL: {sql_query}")

            # Clean the SQL query
            sql_query = self._clean_sql_query(sql_query)

            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"sql_generation_{agent_id}",
                "sql_generation", f"✅ SQL Query Generated: {sql_query[:80]}{'...' if len(sql_query) > 80 else ''}"
            )

            # Emit step: Validating and executing query
            workflow_streamer.emit_step_started(
                context.workflow_id, context.session_id, f"execute_query_{agent_id}",
                "query_execution", f"⚡ Executing Query on {agent_name}..."
            )

            # CRITICAL: Validate SQL safety before execution
            safety_check = self.sql_validator.validate_sql_safety(sql_query)

            if not safety_check["is_safe"]:
                error_msg = f"SQL Safety Violation: {'; '.join(safety_check['violations'])}"
                print(f"🚨 {error_msg}")
                print(f"🚨 Rejected Query: {sql_query}")
                
                workflow_streamer.emit_step_completed(
                    context.workflow_id, context.session_id, f"execute_query_{agent_id}",
                    "query_execution", f"❌ SQL Safety Violation: Query rejected"
                )
                raise Exception(f"Query rejected for safety: {error_msg}")

            # Use the sanitized query
            safe_sql_query = safety_check["sanitized_query"]
            print(f"✅ SQL Safety Check Passed")
            print(f"🛡️ Generated Safe SQL: {safe_sql_query[:100]}...")

            # Apply data sampling middleware for large datasets
            sampling_result = self.data_sampling_middleware.process_query(
                query=safe_sql_query,
                user_request=query,
                connection_type=connection_type
            )
            
            # Use the optimized query from sampling middleware
            optimized_sql = sampling_result.get("optimized_query", safe_sql_query)
            sampling_config = sampling_result.get("sampling_config")
            
            if sampling_config:
                print(f"📊 Data sampling applied: {sampling_config['strategy']} strategy")
                print(f"🎯 Sample size: {sampling_config.get('sample_size', 'N/A')}")

            # Execute the optimized query
            db_result = self.db_executor.execute_query(
                agent_details.get("vault_key"),
                connection_type,
                optimized_sql
            )

            if db_result.get("status") == "success":
                data = db_result.get("data", [])
                execution_time = db_result.get("execution_time", 0)
                
                # Post-process data if sampling was applied
                final_data = data
                if sampling_config and hasattr(self.data_sampling_middleware, 'post_process_results'):
                    final_data = self.data_sampling_middleware.post_process_results(
                        data, sampling_config, query
                    )
                
                row_count = len(final_data) if isinstance(final_data, list) else 1
                
                workflow_streamer.emit_step_completed(
                    context.workflow_id, context.session_id, f"execute_query_{agent_id}",
                    "query_execution", f"✅ Query Executed: {row_count} records retrieved in {execution_time:.1f}s"
                )
                
                return {
                    "query": optimized_sql,  # Return the executed query
                    "original_query": sql_query,
                    "safe_query": safe_sql_query,
                    "safety_check": safety_check,
                    "sampling_config": sampling_config,
                    "data": final_data,
                    "row_count": row_count,
                    "execution_time": execution_time,
                    "sampling_applied": sampling_config is not None
                }
            else:
                # Surface the full error message from db_result
                error_msg = db_result.get('error') or db_result.get('message') or 'Unknown error'
                print(f"❌ Database query failed: {error_msg}")
                
                workflow_streamer.emit_step_completed(
                    context.workflow_id, context.session_id, f"execute_query_{agent_id}",
                    "query_execution", f"❌ Query Failed: {error_msg}"
                )
                raise Exception(f"Database query failed: {error_msg}")

        except Exception as e:
            # Emit step: Error occurred
            workflow_streamer.emit_step_completed(
                context.workflow_id, context.session_id, f"sql_generation_{agent_id}",
                "sql_generation", f"❌ SQL Generation Failed: {str(e)}"
            )
            raise Exception(f"Data agent execution failed: {str(e)}")
    
    async def _execute_application_agent(self, agent_id: str, query: str, context: ConversationContext) -> Dict[str, Any]:
        """Execute an application agent."""
        # Simplified application agent execution
        # This would be expanded with actual API calling logic
        return {
            "query": f"API call for: {query}",
            "data": {"status": "simulated_success", "message": f"API call executed for {query}"},
            "row_count": 1
        }
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean up SQL query formatting."""
        if not sql_query:
            return sql_query
        
        # Remove common formatting issues
        sql_query = sql_query.replace("\\n", " ").replace("\\t", " ")
        sql_query = " ".join(sql_query.split())
        
        # Basic parentheses balancing
        open_parens = sql_query.count('(')
        close_parens = sql_query.count(')')
        if open_parens > close_parens:
            sql_query += ')' * (open_parens - close_parens)
        
        return sql_query
    
    def _format_final_response(self, context: ConversationContext) -> Dict[str, Any]:
        """Format the final response with all results and insights."""
        # Compile insights from all results
        insights = []
        total_execution_time = 0
        
        for result in context.agent_results:
            total_execution_time += result.execution_time
            if result.insight_summary:
                insights.append(result.insight_summary)
        
        # Get the latest analysis
        latest_analysis = None
        for entry in reversed(context.conversation_history):
            if entry.get("step") == "result_analysis":
                latest_analysis = entry.get("analysis")
                break
        
        # Generate natural language response from the results
        final_answer = self._generate_final_answer(context, latest_analysis)
        
        return {
            "status": "success",
            "workflow_id": context.workflow_id,
            "session_id": context.session_id,
            "original_query": context.original_query,
            "refined_query": context.refined_query,
            "final_answer": final_answer,  # Add the natural language response
            "total_steps": len(context.agent_results),
            "total_execution_time": total_execution_time,
            "agents_used": [
                {
                    "agent_id": r.agent_id,
                    "agent_name": r.agent_name,
                    "agent_type": r.agent_type,
                    "success": r.success,
                    "execution_time": r.execution_time
                }
                for r in context.agent_results
            ],
            "results": [
                {
                    "agent_name": r.agent_name,
                    "success": r.success,
                    "data": r.data,
                    "row_count": r.row_count,
                    "query_executed": r.query_executed,
                    "insight": r.insight_summary
                }
                for r in context.agent_results
            ],
            "analysis": latest_analysis,
            "suggested_follow_ups": context.suggested_follow_ups,
            "conversation_history": context.conversation_history
        }
    
    def _generate_final_answer(self, context: ConversationContext, analysis: Dict[str, Any]) -> str:
        """Generate simple response using LLM to decide everything."""
        try:
            # Let LLM decide everything in one call
            return self._generate_smart_response(context)
            
        except Exception as e:
            print(f"❌ Failed to generate final answer: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."
    
    def _generate_smart_response(self, context: ConversationContext) -> str:
        """Single LLM call to handle all query types intelligently - exactly as requested."""
        from app.agents import fetch_agents_and_tools_from_registry
        
        try:
            # Get available capabilities
            agents = fetch_agents_and_tools_from_registry()
            capabilities = []
            
            for agent_id, agent in agents.items():
                if agent.get("agent_type") in ["data_agent", "application"]:
                    capabilities.append({
                        "name": agent.get("name", agent_id),
                        "type": agent.get("agent_type"),
                        "description": agent.get("description", "")
                    })
            
            # Check if we have data results
            successful_results = [r for r in context.agent_results if r.success and r.data]
            has_data = bool(successful_results)
            
            # Single LLM prompt to handle everything - keep it simple as requested
            prompt = f"""
            You are a helpful AI assistant. The user asked: "{context.original_query}"
            
            Available capabilities: {capabilities}
            Has data results: {has_data}
            
            Instructions - Keep it simple:
            1. If it's a greeting (hi, hello) - respond warmly and mention you can help with data
            2. If it's vague ("can I get information?", "help me") - be conversational, ask how you can help, mention capabilities  
            3. If it's general knowledge - answer directly
            4. If we have data - provide business analysis
            5. If specific request failed - provide guidance
            
            Be natural and conversational. Don't overthink it.
            """
            
            if has_data:
                # Add simple data context 
                data_summary = f"Retrieved {len(successful_results)} datasets with {sum(len(r.data) if isinstance(r.data, list) else 1 for r in successful_results)} total records"
                prompt += f"\n\nData available: {data_summary}\nProvide useful business insights from this data."
            
            response = self.llm_client.invoke_with_text_response(
                prompt, 
                task_type=TaskType.CONVERSATION
            )
            return response if response else "I'm here to help! What would you like to know?"
            
        except Exception as e:
            print(f"❌ Smart response generation failed: {e}")
            return "I'm here to help! What would you like to know?"
    
    def _prepare_simple_data_context(self, results: List) -> str:
        """Simple data context for LLM."""
        context = ""
        for result in results:
            context += f"From {result.agent_name}: {len(result.data) if isinstance(result.data, list) else 1} records\n"
        return context
    
    def _is_vague_request(self, query: str) -> bool:
        """Check if the query is too vague to process without clarification."""
        vague_patterns = [
            "i need some data", "i need data", "give me data", "show me data",
            "i want information", "i need information", "help me", "i need help",
            "what can you do", "what do you have"
        ]
        return any(pattern in query.lower() for pattern in vague_patterns)
    
    def _generate_clarification_request(self, query: str) -> str:
        """Generate a helpful clarification request for vague queries with dynamic capabilities using LLM."""
        from app.agents import fetch_agents_and_tools_from_registry
        
        try:
            # Get available agents to show capabilities
            agents = fetch_agents_and_tools_from_registry()
            capabilities_context = []
            
            for agent_id, agent in agents.items():
                agent_type = agent.get("agent_type", "unknown")
                if agent_type in ["data_agent", "application"]:
                    agent_name = agent.get("name", agent_id)
                    description = agent.get("description", "")
                    
                    capability_info = {
                        "name": agent_name,
                        "type": agent_type,
                        "description": description
                    }
                    
                    if agent_type == "data_agent":
                        capability_info["database_type"] = agent.get("connection_type", agent.get("database_type", "database"))
                    else:
                        capability_info["application_type"] = agent.get("application_type", "API")
                    
                    capabilities_context.append(capability_info)
            
            # Use LLM to generate a natural, conversational response
            prompt = f"""
            The user asked: "{query}"
            
            This is a vague request that needs clarification. You need to respond in a helpful, conversational way that:
            1. Acknowledges their request positively (e.g., "Sure!" or "I'd be happy to help!")
            2. Shows what capabilities are available to them
            3. Asks for more specific information
            4. Provides helpful examples of what they could ask for
            
            Available capabilities:
            {capabilities_context}
            
            Generate a natural, friendly response that helps the user understand what they can ask for and guides them to be more specific. 
            Don't use bullet points or formal formatting - make it conversational and helpful.
            Include specific examples of questions they could ask based on the available capabilities.
            
            The tone should be like a helpful assistant who wants to understand exactly what the user needs.
            """
            
            response = self.llm_client.invoke_with_text_response(
                prompt, 
                task_type=TaskType.CONVERSATION
            )
            return response if response else self._generate_fallback_clarification()
            
        except Exception as e:
            print(f"❌ Failed to generate LLM clarification: {e}")
            return self._generate_fallback_clarification()
    
    def _generate_fallback_clarification(self) -> str:
        """Generate a simple fallback clarification when LLM fails."""
        return """Sure! I'd be happy to help you find the information you need. Could you please be more specific about what you're looking for? 

For example, you could ask about inventory levels, sales data, customer information, or any other business metrics. The more details you provide, the better I can assist you!"""
    
    def _generate_knowledge_response(self, query: str) -> str:
        """Generate a response using general knowledge (no agents used)."""
        try:
            prompt = f"""
            The user asked: "{query}"
            
            This appears to be a general knowledge question that doesn't require database access or specific data analysis.
            
            Instructions:
            1. Provide a direct, helpful answer based on general knowledge
            2. Use a natural, conversational tone
            3. Structure your response appropriately for the type of question
            4. Don't include methodology, key insights, or statistics sections unless the question specifically asks for them
            5. Be comprehensive but concise
            6. If it's a factual question, provide the facts clearly
            7. If it's a location question, give relevant location information
            
            Respond naturally without forcing any particular format.
            """
            
            response = self.llm_client.invoke_with_text_response(prompt)
            return response if response else "I'd be happy to help! Could you please provide more specific details about what you're looking for?"
            
        except Exception as e:
            print(f"❌ Knowledge response generation failed: {e}")
            return "I'd be happy to help! Could you please provide more specific details about what you're looking for?"
    
    def _generate_data_analysis_response(self, context: ConversationContext, successful_results: List) -> str:
        """Generate intelligent data analysis response using LLM to detect context and format appropriately."""
        try:
            # Prepare comprehensive data context for LLM
            data_context = self._prepare_comprehensive_data_context(successful_results)
            
            # Let the LLM analyze the intent and generate the appropriate response
            prompt = f"""
            You are an expert business analyst. Analyze the user's question and the retrieved data to provide the most helpful response.

            USER'S QUESTION: "{context.original_query}"

            DATA RETRIEVED:
            {data_context}

            INSTRUCTIONS:
            1. **Analyze the intent**: Determine what type of business analysis this is (inventory, sales, customer, financial, etc.)
            2. **Focus on business value**: Provide actionable insights rather than technical details
            3. **Identify critical items**: If this is inventory analysis, highlight items needing immediate attention
            4. **Provide recommendations**: Give specific, actionable advice
            5. **Use appropriate formatting**: Structure your response for the detected analysis type
            6. **Be business-focused**: Focus on impact, risks, and actions rather than methodology

            RESPONSE GUIDELINES:
            - For inventory analysis: Focus on critical items, supplier consultation, immediate actions
            - For sales analysis: Highlight top performers, trends, optimization opportunities  
            - For customer analysis: Identify key segments, high-value customers, retention strategies
            - For financial analysis: Focus on costs, margins, optimization opportunities
            - For general queries: Provide direct answers with relevant insights

            Generate a comprehensive, business-focused response that directly addresses what the user is asking for.
            """
            
            response = self.llm_client.invoke_with_text_response(
                prompt, 
                task_type=TaskType.DATA_ANSWER
            )
            return response if response else "I've analyzed your data and retrieved the requested information."
            
        except Exception as e:
            print(f"❌ Data analysis response generation failed: {e}")
            return self._generate_fallback_response(context)
    
    def _prepare_comprehensive_data_context(self, successful_results: List) -> str:
        """Prepare comprehensive data context for LLM analysis."""
        data_context = ""
        
        for result in successful_results:
            agent_name = result.agent_name
            row_count = result.row_count or (len(result.data) if isinstance(result.data, list) else 1)
            
            data_context += f"\n=== {agent_name} ===\n"
            data_context += f"Records Retrieved: {row_count}\n"
            
            # Include sample data for context
            if isinstance(result.data, list) and result.data:
                data_context += f"Sample Data (first 10 records):\n"
                for i, record in enumerate(result.data[:10]):
                    if isinstance(record, dict):
                        # Format record nicely for LLM
                        record_str = ", ".join([f"{k}: {v}" for k, v in record.items()])
                        data_context += f"  {i+1}. {record_str}\n"
                
                # Add field summary
                if isinstance(result.data[0], dict):
                    fields = list(result.data[0].keys())
                    data_context += f"Available Fields: {', '.join(fields)}\n"
            elif isinstance(result.data, dict):
                data_context += f"Data: {result.data}\n"
            
            data_context += f"Query Executed: {result.query_executed or 'N/A'}\n"
            data_context += "\n"
        
        return data_context
    
    def _generate_failure_guidance_response(self, context: ConversationContext) -> str:
        """Generate helpful guidance when data retrieval failed."""
        failed_agents = [r for r in context.agent_results if not r.success]
        
        if failed_agents:
            agent_names = [r.agent_name for r in failed_agents]
            return f"""I attempted to retrieve data from {', '.join(agent_names)} but encountered some issues. 

This could be due to:
• The specific data you're looking for might not be available in the current databases
• There might be connectivity issues with the data sources
• The query might need to be more specific or adjusted

Could you please:
1. Try rephrasing your question with more specific details
2. Let me know if you're looking for data from a particular system or time period
3. Specify what type of analysis or information would be most helpful

I'm here to help once we clarify what you're looking for!"""
        
        return "I wasn't able to retrieve the specific data you requested. Could you please provide more details about what you're looking for so I can better assist you?"
    
    async def _check_user_input_required(self, context: ConversationContext) -> Dict[str, Any]:
        """Check if user input is required for agent selection."""
        from app.agents import fetch_agents_and_tools_from_registry
        
        print(f"[DEBUG] _check_user_input_required called for context: {context.workflow_id}")
        
        # Get planning result from intent analysis
        planning_result = None
        for entry in context.conversation_history:
            if entry.get("step") == "intent_analysis":
                planning_result = entry.get("result")
                break
        
        print(f"[DEBUG] Planning result: {planning_result}")
        
        if not planning_result:
            print(f"[DEBUG] No planning result found - no user input required")
            return {"requires_input": False}
        
        execution_strategy = planning_result.get("execution_strategy")
        primary_agent_id = planning_result.get("primary_agent_id")
        intent_category = planning_result.get("intent_category")
        
        print(f"[DEBUG] Execution strategy: {execution_strategy}, Primary agent: {primary_agent_id}, Intent: {intent_category}")
        
        # For vague requests, let the workflow complete and return LLM-generated clarification in final_answer
        # instead of pausing for user input
        if self._is_vague_request(context.original_query):
            print(f"[DEBUG] Vague request detected - will handle in final_answer generation")
            return {"requires_input": False}
        
        # For specific data requests with no agent selected, require user input
        if intent_category == "data_analysis" and not primary_agent_id and not self._is_vague_request(context.original_query):
            print(f"[DEBUG] Specific data analysis request with no agent selected - checking for multiple sources")
            agents = fetch_agents_and_tools_from_registry()
            
            # Get all available agents (both data and application)
            available_agents = {
                agent_id: agent 
                for agent_id, agent in agents.items() 
                if agent.get("agent_type") in ["data_agent", "application"]
            }
            
            print(f"[DEBUG] Found {len(available_agents)} available sources: {list(available_agents.keys())}")
            
            if len(available_agents) > 1:
                print(f"[DEBUG] Multiple sources available for specific data request - USER INPUT REQUIRED")
                # Multiple sources available, user needs to choose
                agent_options = []
                for agent_id, agent in available_agents.items():
                    agent_type = agent.get("agent_type", "unknown")
                    
                    if agent_type == "data_agent":
                        # Database source
                        agent_options.append({
                            "agent_id": agent_id,
                            "name": agent.get("name", agent_id),
                            "source_type": "database",
                            "database_type": agent.get("connection_type", agent.get("database_type", "unknown")),
                            "description": agent.get("description", f"{agent.get('name', agent_id)} database")
                        })
                    elif agent_type == "application":
                        # Application source
                        agent_options.append({
                            "agent_id": agent_id,
                            "name": agent.get("name", agent_id),
                            "source_type": "application",
                            "application_type": agent.get("application_type", "API"),
                            "description": agent.get("description", f"{agent.get('name', agent_id)} application")
                        })
                
                input_request = {
                    "type": "source_selection",
                    "prompt": f"Please specify which data source you'd like to query.",
                    "context": agent_options,
                    "reasoning": f"I found {len(available_agents)} different sources that could help with your request. Please specify which one you'd like me to use.",
                    "workflow_id": context.workflow_id,
                    "step": 0
                }
                
                print(f"🚨 USER INPUT REQUIRED: Multiple sources available ({len(available_agents)} options)")
                return {
                    "requires_input": True,
                    "input_request": input_request
                }
        
        print(f"[DEBUG] No user input required")
        return {"requires_input": False}
    
    async def resume_enhanced_workflow_with_user_choice(self, workflow_id: str, session_id: str, user_choice: Dict[str, Any]) -> Dict[str, Any]:
        """Resume enhanced workflow after user makes a choice."""
        print(f"[EnhancedLLMOrchestrator] Resuming workflow {workflow_id} with user choice: {user_choice}")
        
        choice_type = user_choice.get("type")
        
        if choice_type in ["database_selection", "source_selection"]:
            chosen_agent_id = user_choice.get("agent_id")
            user_message = user_choice.get("message", "")
            
            print(f"[EnhancedLLMOrchestrator] User chose agent: {chosen_agent_id}, message: '{user_message}'")
            
            # Check if user cancelled
            if self._is_cancellation_response(user_message):
                return {
                    "status": "cancelled",
                    "message": "Sorry, we couldn't help you with your request. None of the available sources match what you're looking for.",
                    "workflow_id": workflow_id,
                    "session_id": session_id
                }
            
            # Parse natural language response to extract option choice if no direct agent_id
            if not chosen_agent_id and user_message:
                agents = fetch_agents_and_tools_from_registry()
                available_agents = {
                    agent_id: agent 
                    for agent_id, agent in agents.items() 
                    if agent.get("agent_type") in ["data_agent", "application"]
                }
                chosen_agent_id = self._parse_agent_choice(user_message, available_agents)
            
            if not chosen_agent_id:
                return {
                    "status": "error",
                    "message": f"I couldn't identify which source you selected from your response: '{user_message}'. Please try again with a clearer choice."
                }
            
            # Get the original query from session context
            session_context = self.session_manager.get_session_context(session_id)
            original_query = user_choice.get("original_query", session_context.get("last_query", user_message)) if session_context else user_message
            
            # Create new context with selected agent
            context = ConversationContext(
                session_id=session_id,
                workflow_id=workflow_id,
                original_query=original_query
            )
            
            # Set execution strategy to single agent with the chosen agent
            context.execution_strategy = "single_agent"
            context.primary_agent_id = chosen_agent_id
            context.refined_query = original_query
            
            try:
                # Execute the chosen agent
                result = await self._execute_agent(chosen_agent_id, original_query, context)
                context.agent_results.append(result)
                
                # Generate final response
                final_response = self._format_final_response(context)
                
                # Save session context
                self.session_manager.update_session_context(session_id, context)
                
                # Emit workflow completed
                execution_time = time.time() - context.start_time
                final_answer = final_response.get("results", [{}])[-1].get("insight", "Analysis completed") if final_response.get("results") else "Query completed"
                
                workflow_streamer.emit_workflow_completed(
                    workflow_id=workflow_id,
                    session_id=session_id,
                    final_answer=final_answer,
                    execution_time=execution_time
                )
                
                return final_response
                
            except Exception as e:
                print(f"❌ Enhanced workflow resumption failed: {e}")
                workflow_streamer.emit_error(
                    workflow_id, session_id, "resumption_error", f"Workflow resumption failed: {str(e)}"
                )
                return {
                    "status": "error",
                    "error": str(e),
                    "partial_results": [r.__dict__ for r in context.agent_results]
                }
        
        return {
            "status": "error",
            "message": f"Unknown choice type: {choice_type}"
        }
    
    def _is_cancellation_response(self, message: str) -> bool:
        """Check if user response indicates cancellation."""
        if not message:
            return False
        
        cancellation_words = ["none", "cancel", "nevermind", "never mind", "abort", "stop", "quit", "exit"]
        message_lower = message.lower().strip()
        
        return any(word in message_lower for word in cancellation_words)
    
    def _parse_agent_choice(self, message: str, available_agents: Dict[str, Any]) -> Optional[str]:
        """Parse user message to extract agent choice."""
        message_lower = message.lower().strip()
        
        # Try to match by agent name or ID
        for agent_id, agent in available_agents.items():
            agent_name = agent.get("name", agent_id).lower()
            if agent_name in message_lower or agent_id.lower() in message_lower:
                return agent_id
        
        # Try to match by database type
        for agent_id, agent in available_agents.items():
            db_type = agent.get("connection_type", agent.get("database_type", "")).lower()
            if db_type and db_type in message_lower:
                return agent_id
        
        # Try to match option patterns (option 1, option a, first option, etc.)
        import re
        option_patterns = [
            r'option\s*([1-9a-z])',
            r'choice\s*([1-9a-z])',
            r'number\s*([1-9])',
            r'^([1-9a-z])\.?\s*$',
            r'first|1st',
            r'second|2nd', 
            r'third|3rd'
        ]
        
        agent_list = list(available_agents.keys())
        
        for pattern in option_patterns:
            match = re.search(pattern, message_lower)
            if match:
                if pattern in ['first|1st']:
                    return agent_list[0] if agent_list else None
                elif pattern in ['second|2nd']:
                    return agent_list[1] if len(agent_list) > 1 else None
                elif pattern in ['third|3rd']:
                    return agent_list[2] if len(agent_list) > 2 else None
                else:
                    option_value = match.group(1)
                    try:
                        # Convert to index (1-based to 0-based)
                        if option_value.isdigit():
                            index = int(option_value) - 1
                            if 0 <= index < len(agent_list):
                                return agent_list[index]
                        elif option_value.isalpha() and len(option_value) == 1:
                            # Convert a, b, c to 0, 1, 2
                            index = ord(option_value.lower()) - ord('a')
                            if 0 <= index < len(agent_list):
                                return agent_list[index]
                    except (ValueError, IndexError):
                        continue
        
        return None
    
    def _generate_fallback_response(self, context: ConversationContext) -> str:
        """Generate a simple fallback response."""
        if context.agent_results and any(r.success for r in context.agent_results):
            return "I've processed your request and gathered some information. The data has been retrieved and is available for your review."
        else:
            return "I'd be happy to help! Could you please provide more specific details about what data or information you're looking for?"
    
    async def record_feedback(self, workflow_id: str, feedback: str) -> Dict[str, Any]:
        """Record user feedback for a completed workflow."""
        try:
            # In a real implementation, you'd look up the workflow from a store
            # For now, we'll just record the feedback
            
            with sqlite3.connect(self.feedback_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE interaction_logs 
                    SET user_feedback = ? 
                    WHERE workflow_id = ?
                """, (feedback, workflow_id))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    print(f"✅ Feedback recorded for workflow: {workflow_id}")
                    return {"status": "success", "message": "Feedback recorded"}
                else:
                    return {"status": "error", "message": "Workflow not found"}
                    
        except Exception as e:
            print(f"❌ Failed to record feedback: {e}")
            return {"status": "error", "message": str(e)}
