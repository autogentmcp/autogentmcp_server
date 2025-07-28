"""
SQL validation and security checks
"""

import re
from typing import Dict, List, Any, Optional

class SQLValidator:
    """Validates SQL queries for security and correctness"""
    
    # Dangerous SQL keywords that should be blocked
    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE',
        'EXEC', 'EXECUTE', 'MERGE', 'GRANT', 'REVOKE', 'DENY', 'BACKUP',
        'RESTORE', 'SHUTDOWN', 'KILL', 'BULK', 'OPENROWSET', 'OPENDATASOURCE'
    ]
    
    # System tables/schemas that should be protected
    PROTECTED_SCHEMAS = [
        'sys', 'information_schema', 'master', 'model', 'msdb', 'tempdb',
        'pg_catalog', 'pg_toast', 'pg_temp'
    ]
    
    @classmethod
    def validate_sql_query(cls, sql: str) -> Dict[str, Any]:
        """
        Validate SQL query for security and safety
        Returns: {"is_valid": bool, "error": str, "warnings": List[str]}
        """
        warnings = []
        
        if not sql or not sql.strip():
            return {"is_valid": False, "error": "Empty SQL query", "warnings": []}
        
        # Clean and normalize SQL
        sql_upper = sql.upper().strip()
        
        # Check for dangerous keywords
        for keyword in cls.DANGEROUS_KEYWORDS:
            if re.search(rf'\\b{keyword}\\b', sql_upper):
                return {
                    "is_valid": False, 
                    "error": f"Query contains forbidden operation: {keyword}. Only SELECT queries are allowed.",
                    "warnings": warnings
                }
        
        # Must start with SELECT (after cleaning comments)
        sql_clean = re.sub(r'--.*?\\n|/\\*.*?\\*/', '', sql_upper, flags=re.DOTALL)
        sql_clean = sql_clean.strip()
        
        if not sql_clean.startswith('SELECT') and not sql_clean.startswith('WITH'):
            return {
                "is_valid": False,
                "error": "Only SELECT and WITH queries are allowed",
                "warnings": warnings
            }
        
        # Check for system schema access
        for schema in cls.PROTECTED_SCHEMAS:
            if re.search(rf'\\b{schema}\\.', sql_upper):
                warnings.append(f"Query accesses system schema: {schema}")
        
        # Check for potential SQL injection patterns
        injection_patterns = [
            r"';\\s*--",  # Comment injection
            r"';\\s*/\\*",  # Block comment injection
            r"\\bunion\\s+select\\b",  # Union injection
            r"\\bor\\s+1\\s*=\\s*1\\b",  # Always true condition
            r"\\band\\s+1\\s*=\\s*1\\b"  # Always true condition
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql_upper):
                warnings.append("Query contains potential SQL injection pattern")
                break
        
        # Check for excessive complexity (basic heuristics)
        if sql.count('(') > 20:
            warnings.append("Query has high complexity - consider simplifying")
        
        if len(sql) > 5000:
            warnings.append("Query is very long - consider breaking into smaller parts")
        
        return {"is_valid": True, "error": None, "warnings": warnings}
    
    @classmethod
    def sanitize_sql_query(cls, sql: str) -> str:
        """
        Basic sanitization of SQL query
        """
        if not sql:
            return ""
        
        # Remove potential dangerous comments
        sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\\*.*?\\*/', '', sql, flags=re.DOTALL)
        
        # Clean extra whitespace
        sql = re.sub(r'\\s+', ' ', sql)
        sql = sql.strip()
        
        return sql
    
    @classmethod
    def extract_table_names(cls, sql: str) -> List[str]:
        """
        Extract table names from SQL query for validation
        """
        # Basic regex to find table names after FROM and JOIN
        pattern = r'\\b(?:FROM|JOIN)\\s+([\\w\\.]+)'
        matches = re.findall(pattern, sql.upper())
        
        # Clean schema prefixes and duplicates
        tables = []
        for match in matches:
            if '.' in match:
                table = match.split('.')[-1]  # Get table name without schema
            else:
                table = match
            if table not in tables:
                tables.append(table)
        
        return tables
