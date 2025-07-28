"""
SQLite-based conversation storage system
Implements ChatGPT-style conversation management with automatic summarization
"""

import sqlite3
import json
import uuid
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib

class ConversationStorage:
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    title TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    is_archived BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_session 
                ON conversations (session_id, updated_at DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages (conversation_id, created_at ASC)
            """)
    
    def create_session(self, session_id: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """Create a new session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, user_id) VALUES (?, ?)",
                (session_id, user_id)
            )
            return {"session_id": session_id, "user_id": user_id}
    
    def create_conversation(self, session_id: str) -> str:
        """Create a new conversation and return its ID."""
        conversation_id = f"conv_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:8]}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO conversations 
                   (conversation_id, session_id, title) 
                   VALUES (?, ?, ?)""",
                (conversation_id, session_id, "New Chat")
            )
            
            # Update session activity
            conn.execute(
                "UPDATE sessions SET last_activity = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )
        
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a message to a conversation."""
        message_id = f"msg_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:8]}"
        
        # Safely serialize metadata, removing circular references
        safe_metadata = {}
        if metadata:
            for key, value in metadata.items():
                try:
                    # Test if the value can be JSON serialized
                    json.dumps(value)
                    safe_metadata[key] = value
                except (TypeError, ValueError):
                    # Skip values that can't be serialized (like functions, circular refs)
                    safe_metadata[key] = str(value)[:200]  # Convert to string and truncate
        
        metadata_json = json.dumps(safe_metadata)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO messages 
                   (message_id, conversation_id, role, content, metadata) 
                   VALUES (?, ?, ?, ?, ?)""",
                (message_id, conversation_id, role, content, metadata_json)
            )
            
            # Update conversation message count and timestamp
            conn.execute(
                """UPDATE conversations 
                   SET message_count = message_count + 1, 
                       updated_at = CURRENT_TIMESTAMP 
                   WHERE conversation_id = ?""",
                (conversation_id,)
            )
            
            # Update session activity
            conn.execute(
                """UPDATE sessions 
                   SET last_activity = CURRENT_TIMESTAMP 
                   WHERE session_id = (
                       SELECT session_id FROM conversations WHERE conversation_id = ?
                   )""",
                (conversation_id,)
            )
        
        return message_id
    
    def get_conversations(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT conversation_id, title, summary, created_at, updated_at, message_count
                   FROM conversations 
                   WHERE session_id = ? AND is_archived = FALSE
                   ORDER BY updated_at DESC""",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT message_id, role, content, metadata, created_at
                   FROM messages 
                   WHERE conversation_id = ?
                   ORDER BY created_at ASC""",
                (conversation_id,)
            )
            
            messages = []
            for row in cursor.fetchall():
                message = dict(row)
                message['metadata'] = json.loads(message['metadata'] or '{}')
                messages.append(message)
            
            return messages
    
    def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                (title, conversation_id)
            )
    
    def generate_conversation_title_with_llm(self, conversation_id: str, backend_url: str = "http://localhost:8001") -> str:
        """Generate a title for a conversation using LLM."""
        messages = self.get_conversation_messages(conversation_id)
        
        if not messages:
            return "New Chat"
        
        # Find the first user message
        first_user_message = None
        for msg in messages:
            if msg['role'] == 'user':
                first_user_message = msg['content']
                break
        
        if not first_user_message:
            return "New Chat"
        
        try:
            # Make LLM call to generate title
            payload = {
                "query": f"Generate a short, descriptive title (max 6 words) for a conversation that starts with this user message: '{first_user_message}'. Only return the title, nothing else.",
                "session_id": "title_generation",
                "conversation_id": "temp",
                "include_analysis": False,
                "max_steps": 1
            }
            
            response = requests.post(
                f"{backend_url}/enhanced/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                generated_title = data.get("enhanced_result", {}).get("final_answer", "")
                
                # Clean up the title
                if generated_title:
                    title = generated_title.strip().strip('"').strip("'")
                    # Ensure it's not too long
                    if len(title) > 50:
                        title = title[:47] + "..."
                else:
                    title = self.generate_conversation_title(conversation_id)
            else:
                # Fallback to simple title generation
                title = self.generate_conversation_title(conversation_id)
                
        except Exception as e:
            print(f"[DEBUG] LLM title generation failed: {e}")
            # Fallback to simple title generation
            title = self.generate_conversation_title(conversation_id)
        
        # Update the conversation with the new title
        self.update_conversation_title(conversation_id, title)
        
        return title
    
    def generate_conversation_title(self, conversation_id: str) -> str:
        """Generate a simple title for a conversation based on the first few messages (fallback method)."""
        messages = self.get_conversation_messages(conversation_id)
        
        if not messages:
            return "New Chat"
        
        # Find the first user message
        first_user_message = None
        for msg in messages:
            if msg['role'] == 'user':
                first_user_message = msg['content']
                break
        
        if not first_user_message:
            return "New Chat"
        
        # Simple title generation - take first 50 chars and clean up
        title = first_user_message.strip()[:50]
        if len(first_user_message) > 50:
            title += "..."
        
        # Clean up the title
        title = title.replace('\n', ' ').replace('\r', ' ')
        while '  ' in title:
            title = title.replace('  ', ' ')
        
        # Update the conversation with the new title
        self.update_conversation_title(conversation_id, title)
        
        return title
    
    def generate_conversation_summary(self, conversation_id: str) -> str:
        """Generate a summary for a conversation (placeholder for future LLM integration)."""
        messages = self.get_conversation_messages(conversation_id)
        
        if len(messages) < 4:  # Don't summarize short conversations
            return ""
        
        # Simple summary - this could be enhanced with LLM summarization
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
        topics = []
        
        for msg in user_messages:
            # Extract key terms (very basic approach)
            words = msg.lower().split()
            key_words = [word for word in words if len(word) > 4 and word.isalpha()]
            topics.extend(key_words[:3])  # Take first 3 meaningful words
        
        if topics:
            # Remove duplicates and take most common
            unique_topics = list(set(topics))[:5]
            summary = f"Discussion about {', '.join(unique_topics)}"
        else:
            summary = f"Conversation with {len(messages)} messages"
        
        # Update the conversation with the summary
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET summary = ? WHERE conversation_id = ?",
                (summary, conversation_id)
            )
        
        return summary
    
    def delete_conversation(self, conversation_id: str):
        """Archive a conversation (soft delete)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET is_archived = TRUE WHERE conversation_id = ?",
                (conversation_id,)
            )
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """Clean up old sessions and their conversations."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE conversations SET is_archived = TRUE 
                   WHERE conversation_id IN (
                       SELECT c.conversation_id 
                       FROM conversations c
                       JOIN sessions s ON c.session_id = s.session_id
                       WHERE s.last_activity < datetime('now', '-{} days')
                   )""".format(days_old)
            )
