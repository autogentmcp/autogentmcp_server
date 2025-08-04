"""
Modular SQL Prompt Builder
Creates focused, context-specific prompts for SQL generation to reduce token count and complexity
"""

from typing import Dict, List, Any, Optional
import json

class SQLPromptBuilder:
    """Builds optimized SQL generation prompts based on query type and database dialect"""
    
    def __init__(self):
        self.base_prompt_template = """You are a SQL query generator. Generate a query based on the provided schema and user request.

## USER REQUEST: "{query}"

## AVAILABLE SCHEMA:
{schema}

{business_context}

{sample_queries}

## CRITICAL COLUMN VALIDATION RULES:
- You MUST ONLY use the exact column names listed in the schema above
- Do NOT make assumptions about column names (e.g., created_ts, updated_ts, etc.)
- If you need a timestamp/date column, use ONLY the columns that exist in the schema
- VERIFY each column name exists in the schema before using it
- When in doubt about column names, prefer the ones explicitly listed in the schema
- For BigQuery: TIMESTAMP_SUB only supports DAY, HOUR, MINUTE, SECOND intervals - use DATE_SUB with DATE() conversion for MONTH/YEAR

{validation_feedback}

## CRITICAL RULES:
- ONLY use exact table names and column names from the schema above
- Include schema prefix (e.g., "schema.table_name")
- Use table aliases (e.g., "t" for table)
- Target dialect: {dialect}
- Never use column names that don't exist in the provided schema

{specific_rules}

{custom_prompt}

## OUTPUT FORMAT:
Return JSON:
{{
  "status": "ready|needs_clarification|cannot_proceed",
  "reasoning": "Brief explanation",
  "query": "SQL query or empty if not ready",
  "tables_used": ["table1"],
  "output_format": ["table", "chart_type"],
  "chart_spec": {{
    "type": "chart_type", 
    "x": "column_name", 
    "y": "value_column", 
    "title": "Descriptive Chart Title",
    "x_label": "X-axis Label",
    "y_label": "Y-axis Label",
    "color_scheme": "blue|green|orange|purple",
    "chart_subtitle": "Additional context",
    "data_format": "currency|percentage|number|date",
    "aggregation_type": "sum|count|avg|min|max",
    "sort_order": "asc|desc",
    "show_values": true,
    "legend_position": "top|bottom|right|none"
  }}
}}

Generate the SQL now."""

    def build_prompt(self, 
                    query: str, 
                    schema: List[Dict[str, Any]], 
                    database_type: str = "unknown",
                    query_type: Optional[str] = None,
                    custom_prompt: str = "",
                    sample_queries: Optional[List[str]] = None,
                    business_context: str = "",
                    validation_feedback: str = "") -> str:
        """
        Build an optimized prompt based on query type and database
        
        Args:
            query: User's natural language query
            schema: Simplified schema structure
            database_type: Target database type (bigquery, mssql, postgresql, etc.)
            query_type: Detected query type (simple, aggregation, trend, etc.)
            custom_prompt: Agent-specific custom guidelines and instructions
            sample_queries: Example queries that work with this database
            business_context: Business context and table descriptions
            validation_feedback: Feedback from previous validation failures for retry attempts
        """
        
        # Filter schema to only relevant tables
        relevant_schema = self._filter_relevant_schema(query, schema)
        
        # Simplify schema to reduce tokens
        simplified_schema = self._simplify_schema(relevant_schema)
        schema_json = json.dumps(simplified_schema, indent=2)
        
        # Get database-specific rules
        specific_rules = self._get_database_rules(database_type, query_type)
        
        # Format custom prompt section
        custom_prompt_section = ""
        if custom_prompt and custom_prompt.strip():
            custom_prompt_section = f"\n## AGENT-SPECIFIC GUIDELINES:\n{custom_prompt.strip()}\n"
        
        # Format business context section
        business_context_section = ""
        if business_context and business_context.strip():
            business_context_section = f"\n## BUSINESS CONTEXT:\n{business_context.strip()}\n"
        
        # Format sample queries section
        sample_queries_section = ""
        if sample_queries and len(sample_queries) > 0:
            sample_queries_section = "\n## EXAMPLE QUERIES:\n"
            for i, example in enumerate(sample_queries[:5], 1):  # Limit to 5 examples
                sample_queries_section += f"\n### Example {i}:\n{example.strip()}\n"
            sample_queries_section += "\nUse these examples as reference for query patterns and structure.\n"
        
        return self.base_prompt_template.format(
            query=query,
            schema=schema_json,
            dialect=database_type,
            specific_rules=specific_rules,
            custom_prompt=custom_prompt_section,
            business_context=business_context_section,
            sample_queries=sample_queries_section,
            validation_feedback=validation_feedback
        )
    
    def _filter_relevant_schema(self, query: str, schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return all tables - let the LLM decide which ones are relevant"""
        # No filtering - pass ALL tables to LLM for maximum flexibility and dynamic functionality
        return schema
    
    def _simplify_schema(self, schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simplify schema to essential information only with explicit column validation"""
        simplified = []
        all_columns = []  # Track all available columns for validation
        
        for table in schema:  # Pass ALL tables for accurate query generation
            simplified_table = {
                "table": table.get("tableName", ""),
                "rows": table.get("rowCount", 0),
                "columns": []
            }
            
            # Include all columns for accurate schema representation
            columns = table.get("columns", [])
            for col in columns:
                col_name = col.get("name", "")
                col_info = {
                    "name": col_name,
                    "type": col.get("type", "")
                }
                
                # Track all column names for validation
                if col_name:
                    all_columns.append(f"{simplified_table['table']}.{col_name}")
                
                # Add AI description for better LLM understanding
                if col.get("aiDescription"):
                    col_info["description"] = col.get("aiDescription")
                
                # Add key info if present
                if col.get("keys"):
                    col_info["keys"] = col["keys"]
                    
                simplified_table["columns"].append(col_info)
            
            simplified.append(simplified_table)
        
        # Add explicit column validation section
        if all_columns:
            simplified.insert(0, {
                "VALIDATION_NOTE": "ONLY_USE_THESE_COLUMNS",
                "available_columns": all_columns[:50],  # Show first 50 columns for reference
                "total_columns": len(all_columns)
            })
        
        return simplified
    
    def _get_database_rules(self, database_type: str, query_type: Optional[str] = None) -> str:
        """Get specific rules based on database type and query type"""
        
        rules = []
        
        # Database-specific syntax rules
        if database_type.lower() in ["bigquery", "bq"]:
            rules.extend([
                "- Use `LIMIT N` at end of query",
                "- Use `DATE_TRUNC(date_col, MONTH)` for monthly aggregation",
                "- Use backticks for reserved words: `column`",
                "- For date arithmetic: Use `DATE_SUB(DATE(timestamp_col), INTERVAL 3 MONTH)` instead of TIMESTAMP_SUB with MONTH",
                "- For timestamp arithmetic: Use `TIMESTAMP_SUB(timestamp_col, INTERVAL 7 DAY)` for day/hour/minute intervals only",
                "- Convert TIMESTAMP to DATE for month/year operations: `DATE(timestamp_col)`",
                "- Use `DATETIME_SUB` for DATETIME types with any interval",
                "- Example: `WHERE DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH)`"
            ])
        elif database_type.lower() in ["mssql", "sqlserver"]:
            rules.extend([
                "- Use `SELECT TOP N` at start of query", 
                "- Use `FORMAT(date_col, 'yyyy-MM')` for monthly aggregation",
                "- Use square brackets for reserved words: [column]"
            ])
        elif database_type.lower() in ["postgresql", "postgres"]:
            rules.extend([
                "- Use `LIMIT N` at end of query",
                "- Use `DATE_TRUNC('month', date_col)` for monthly aggregation", 
                "- Use double quotes for case-sensitive names: \"Column\""
            ])
        elif database_type.lower() == "mysql":
            rules.extend([
                "- Use `LIMIT N` at end of query",
                "- Use `DATE_FORMAT(date_col, '%Y-%m')` for monthly aggregation",
                "- Use backticks for reserved words: `column`"
            ])
        else:
            rules.append("- Use standard SQL syntax")
        
        # Query-type specific rules
        if query_type == "simple":
            rules.extend([
                "- Select specific columns, avoid SELECT *",
                "- Add reasonable LIMIT (10-50 rows)",
                "- Use table aliases"
            ])
        elif query_type == "aggregation":
            rules.extend([
                "- Include GROUP BY for non-aggregate columns",
                "- Use meaningful column aliases",
                "- Recommend 'bar_chart' visualization"
            ])
        elif query_type == "trend":
            rules.extend([
                "- Include date/time columns",
                "- Order by date ascending",
                "- Recommend 'line_chart' visualization"
            ])
        elif query_type == "filter":
            rules.extend([
                "- Use appropriate WHERE conditions",
                "- Use LIKE for text searches with %",
                "- Consider case sensitivity"
            ])
        
        # Enhanced visualization rules with detailed chart specifications
        rules.extend([
            "",
            "## ENHANCED VISUALIZATION GUIDELINES:",
            "- **Time Series Data** → 'line_chart' with date/time on x-axis, metrics on y-axis",
            "  - Use colors: blue for single series, multi-color for multiple metrics",
            "  - Include trend subtitle: 'Trend analysis over time'",
            "  - Format dates appropriately (monthly, daily, yearly)",
            "",
            "- **Category Comparisons** → 'bar_chart' with categories on x-axis, values on y-axis", 
            "  - Use ascending/descending order based on values",
            "  - Include data labels on bars for clarity",
            "  - Use color scheme: green for positive metrics, blue for neutral",
            "",
            "- **Single Key Metrics** → 'metric' display with large number format",
            "  - Include percentage change if comparing periods",
            "  - Use appropriate data format (currency, percentage, count)",
            "",
            "- **Top N Results** → 'horizontal_bar_chart' for better label readability",
            "  - Sort by value descending",
            "  - Limit to top 10-15 items for clarity",
            "",
            "- **Distribution Analysis** → 'pie_chart' for percentages/shares",
            "  - Only use when showing parts of a whole",
            "  - Limit to 5-7 categories for readability",
            "",
            "- **Always include 'table' as primary format for data reference",
            "- **Chart titles should be descriptive and business-focused**",
            "- **Use appropriate data formatting**: currency ($), percentages (%), numbers (K, M)",
            "- **Consider color accessibility**: use distinct colors for different categories"
        ])
        
        return "\n".join(rules)
    
    def detect_query_type(self, query: str) -> str:
        """Detect the type of query to optimize prompt"""
        query_lower = query.lower()
        
        # Simple list/show queries (most common)
        if any(phrase in query_lower for phrase in ["show me", "list", "display", "get", "all"]):
            # Check if it's also an aggregation or trend
            if any(word in query_lower for word in ["count", "sum", "total", "how many", "how much"]):
                return "aggregation"
            elif any(word in query_lower for word in ["trend", "over time", "monthly", "daily", "growth", "change"]):
                return "trend"
            else:
                return "simple"
        
        # Trend/time-based queries
        if any(word in query_lower for word in ["trend", "over time", "monthly", "daily", "yearly", "growth", "change", "timeline"]):
            return "trend"
        
        # Aggregation queries  
        if any(word in query_lower for word in ["count", "sum", "total", "average", "avg", "group by", "how many", "how much", "per"]):
            return "aggregation"
        
        # Filter queries
        if any(word in query_lower for word in ["where", "filter", "find", "search", "like", "contains", "with", "by"]):
            return "filter"
        
        # Default to simple
        return "simple"


# Global instance
sql_prompt_builder = SQLPromptBuilder()
