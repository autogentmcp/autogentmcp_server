"""
Data Sampling and Aggregation Middleware for Enhanced LLM Orchestrator
Implements intelligent sampling strategies to handle large datasets efficiently.
"""

import re
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

class SamplingStrategy(Enum):
    """Different sampling strategies available."""
    AGGREGATION = "aggregation"
    TOP_N = "top_n"
    RANDOM_SAMPLE = "random_sample"
    TIME_WINDOW = "time_window"
    HIERARCHICAL = "hierarchical"
    NONE = "none"

@dataclass
class SamplingConfig:
    """Configuration for sampling behavior."""
    max_rows_for_llm: int = 20
    max_tokens_estimate: int = 4000
    aggregation_threshold: int = 1000
    enable_auto_sampling: bool = True
    preferred_strategies: List[SamplingStrategy] = None
    
    def __post_init__(self):
        if self.preferred_strategies is None:
            self.preferred_strategies = [
                SamplingStrategy.AGGREGATION,
                SamplingStrategy.TOP_N,
                SamplingStrategy.TIME_WINDOW
            ]

@dataclass
class SamplingResult:
    """Result of sampling operation."""
    original_query: str
    modified_query: str
    strategy_used: SamplingStrategy
    estimated_reduction: float
    explanation: str
    should_apply_post_processing: bool = False
    post_processing_instructions: str = ""

class DataSamplingMiddleware:
    """
    Middleware that intelligently samples and aggregates data queries
    to optimize for LLM processing while maintaining insight quality.
    """
    
    def __init__(self, config: SamplingConfig = None):
        self.config = config or SamplingConfig()
        self.query_patterns = self._init_query_patterns()
        
    def _init_query_patterns(self) -> Dict[str, Dict]:
        """Initialize regex patterns for different query types."""
        return {
            "inventory_analysis": {
                "patterns": [
                    r"inventory|stock|quantity.*hand|reorder",
                    r"warehouse|store.*inventory",
                    r"low.*stock|out.*stock|stockout"
                ],
                "preferred_strategy": SamplingStrategy.AGGREGATION,
                "group_by_fields": ["warehouse_id", "category", "store_id", "product_category"]
            },
            "sales_analysis": {
                "patterns": [
                    r"sales|revenue|orders",
                    r"top.*selling|best.*perform",
                    r"sales.*trend|monthly.*sales"
                ],
                "preferred_strategy": SamplingStrategy.TIME_WINDOW,
                "time_fields": ["sale_date", "order_date", "created_at", "timestamp"]
            },
            "customer_analysis": {
                "patterns": [
                    r"customer|client|user.*behavior",
                    r"customer.*segment|loyalty",
                    r"repeat.*customer"
                ],
                "preferred_strategy": SamplingStrategy.HIERARCHICAL,
                "hierarchy_fields": ["customer_tier", "region", "segment"]
            },
            "performance_metrics": {
                "patterns": [
                    r"performance|metrics|kpi",
                    r"dashboard|report|summary",
                    r"overview|status"
                ],
                "preferred_strategy": SamplingStrategy.AGGREGATION,
                "aggregate_functions": ["COUNT", "SUM", "AVG", "MAX", "MIN"]
            }
        }
    
    def analyze_query_intent(self, user_query: str, sql_query: str) -> Tuple[str, SamplingStrategy]:
        """Analyze the query to determine best sampling strategy."""
        user_query_lower = user_query.lower()
        sql_query_lower = sql_query.lower()
        
        for category, config in self.query_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, user_query_lower) or re.search(pattern, sql_query_lower):
                    return category, config["preferred_strategy"]
        
        # Default strategy for unknown patterns
        return "general", SamplingStrategy.TOP_N
    
    def should_apply_sampling(self, sql_query: str, estimated_rows: int = None) -> bool:
        """Determine if sampling should be applied to the query."""
        if not self.config.enable_auto_sampling:
            return False
        
        # Check for existing LIMIT clauses
        if re.search(r'\bLIMIT\s+\d+', sql_query, re.IGNORECASE):
            limit_match = re.search(r'\bLIMIT\s+(\d+)', sql_query, re.IGNORECASE)
            if limit_match:
                limit_value = int(limit_match.group(1))
                return limit_value > self.config.max_rows_for_llm
        
        # Check for aggregation functions (usually don't need sampling)
        if re.search(r'\b(COUNT|SUM|AVG|MAX|MIN|GROUP BY)\b', sql_query, re.IGNORECASE):
            # Only sample if it's a complex aggregation likely to return many rows
            return bool(re.search(r'GROUP BY', sql_query, re.IGNORECASE))
        
        # If we have estimated rows, use that
        if estimated_rows:
            return estimated_rows > self.config.aggregation_threshold
        
        # Default: apply sampling for safety
        return True
    
    def apply_sampling_strategy(self, user_query: str, sql_query: str, 
                              connection_type: str, estimated_rows: int = None) -> SamplingResult:
        """Apply the appropriate sampling strategy to the SQL query."""
        
        if not self.should_apply_sampling(sql_query, estimated_rows):
            return SamplingResult(
                original_query=sql_query,
                modified_query=sql_query,
                strategy_used=SamplingStrategy.NONE,
                estimated_reduction=0.0,
                explanation="No sampling needed - query already optimized or small result set expected"
            )
        
        # Analyze query intent
        category, preferred_strategy = self.analyze_query_intent(user_query, sql_query)
        
        # Apply the appropriate strategy
        if preferred_strategy == SamplingStrategy.AGGREGATION:
            return self._apply_aggregation_strategy(user_query, sql_query, connection_type, category)
        elif preferred_strategy == SamplingStrategy.TIME_WINDOW:
            return self._apply_time_window_strategy(user_query, sql_query, connection_type, category)
        elif preferred_strategy == SamplingStrategy.HIERARCHICAL:
            return self._apply_hierarchical_strategy(user_query, sql_query, connection_type, category)
        else:
            return self._apply_top_n_strategy(user_query, sql_query, connection_type)
    
    def _apply_aggregation_strategy(self, user_query: str, sql_query: str, 
                                  connection_type: str, category: str) -> SamplingResult:
        """Apply aggregation-based sampling."""
        
        # Check if query already has aggregation
        if re.search(r'\b(GROUP BY|COUNT|SUM|AVG)\b', sql_query, re.IGNORECASE):
            # Add ORDER BY and LIMIT to existing aggregation
            modified_query = self._add_limit_to_aggregation(sql_query, connection_type)
            explanation = f"Added LIMIT to existing aggregation query for {category} analysis"
        else:
            # Transform to aggregation query
            modified_query = self._transform_to_aggregation(sql_query, connection_type, category)
            explanation = f"Transformed to aggregation query for efficient {category} analysis"
        
        return SamplingResult(
            original_query=sql_query,
            modified_query=modified_query,
            strategy_used=SamplingStrategy.AGGREGATION,
            estimated_reduction=0.9,  # 90% reduction typical for aggregation
            explanation=explanation,
            should_apply_post_processing=True,
            post_processing_instructions="Results represent aggregated data. Use for pattern identification and high-level insights."
        )
    
    def _apply_time_window_strategy(self, user_query: str, sql_query: str, 
                                  connection_type: str, category: str) -> SamplingResult:
        """Apply time-based windowing."""
        
        # Detect time fields in the query
        time_fields = ["created_at", "updated_at", "sale_date", "order_date", "timestamp", "date"]
        detected_time_field = None
        
        for field in time_fields:
            if field in sql_query.lower():
                detected_time_field = field
                break
        
        if detected_time_field:
            # Add time window constraint
            time_constraint = self._generate_time_constraint(detected_time_field, connection_type)
            modified_query = self._add_time_constraint(sql_query, time_constraint)
            explanation = f"Added 7-day time window for recent {category} data"
        else:
            # Fallback to TOP N if no time field detected
            modified_query = self._add_simple_limit(sql_query, connection_type, self.config.max_rows_for_llm)
            explanation = f"Applied TOP {self.config.max_rows_for_llm} sampling (no time field detected)"
        
        return SamplingResult(
            original_query=sql_query,
            modified_query=modified_query,
            strategy_used=SamplingStrategy.TIME_WINDOW,
            estimated_reduction=0.7,  # 70% reduction typical for time windows
            explanation=explanation
        )
    
    def _apply_hierarchical_strategy(self, user_query: str, sql_query: str, 
                                   connection_type: str, category: str) -> SamplingResult:
        """Apply hierarchical sampling."""
        
        # Try to add GROUP BY for hierarchical analysis
        hierarchy_fields = ["category", "region", "department", "store_id", "customer_tier"]
        detected_field = None
        
        for field in hierarchy_fields:
            if field in sql_query.lower():
                detected_field = field
                break
        
        if detected_field:
            modified_query = self._add_hierarchical_grouping(sql_query, detected_field, connection_type)
            explanation = f"Applied hierarchical grouping by {detected_field} for {category} breakdown"
        else:
            # Fallback to aggregation
            modified_query = self._add_simple_limit(sql_query, connection_type, self.config.max_rows_for_llm)
            explanation = f"Applied TOP {self.config.max_rows_for_llm} sampling (hierarchical field not detected)"
        
        return SamplingResult(
            original_query=sql_query,
            modified_query=modified_query,
            strategy_used=SamplingStrategy.HIERARCHICAL,
            estimated_reduction=0.8,  # 80% reduction typical for hierarchical
            explanation=explanation,
            should_apply_post_processing=True,
            post_processing_instructions="Results show hierarchical breakdown. Useful for category-level analysis and drilling down."
        )
    
    def _apply_top_n_strategy(self, user_query: str, sql_query: str, connection_type: str) -> SamplingResult:
        """Apply simple TOP N sampling."""
        
        modified_query = self._add_simple_limit(sql_query, connection_type, self.config.max_rows_for_llm)
        
        return SamplingResult(
            original_query=sql_query,
            modified_query=modified_query,
            strategy_used=SamplingStrategy.TOP_N,
            estimated_reduction=0.5,  # Conservative estimate
            explanation=f"Applied TOP {self.config.max_rows_for_llm} sampling for manageable result set"
        )
    
    def _add_limit_to_aggregation(self, sql_query: str, connection_type: str) -> str:
        """Add LIMIT to existing aggregation query."""
        # Check if already has ORDER BY
        if not re.search(r'\bORDER BY\b', sql_query, re.IGNORECASE):
            # Add ORDER BY for aggregations (usually by count or sum)
            if "COUNT(" in sql_query.upper():
                sql_query = sql_query.rstrip(';') + " ORDER BY COUNT(*) DESC"
            elif "SUM(" in sql_query.upper():
                sql_query = sql_query.rstrip(';') + " ORDER BY 2 DESC"  # Order by second column (sum)
        
        return self._add_simple_limit(sql_query, connection_type, self.config.max_rows_for_llm)
    
    def _transform_to_aggregation(self, sql_query: str, connection_type: str, category: str) -> str:
        """Transform a detail query to an aggregation query."""
        
        # This is a simplified transformation - in practice, you'd want more sophisticated logic
        # based on the table structure and query pattern
        
        # Extract table name
        table_match = re.search(r'\bFROM\s+(\w+(?:\.\w+)?)', sql_query, re.IGNORECASE)
        if not table_match:
            return self._add_simple_limit(sql_query, connection_type, self.config.max_rows_for_llm)
        
        table_name = table_match.group(1)
        
        # Generate appropriate aggregation based on category
        if category == "inventory_analysis":
            agg_query = f"""
            SELECT 
                COALESCE(category, 'Unknown') as category,
                COUNT(*) as item_count,
                SUM(CASE WHEN quantity_on_hand < reorder_level THEN 1 ELSE 0 END) as low_stock_items,
                AVG(quantity_on_hand) as avg_quantity
            FROM {table_name}
            GROUP BY category
            ORDER BY low_stock_items DESC, item_count DESC
            """
        elif category == "sales_analysis":
            agg_query = f"""
            SELECT 
                DATE(COALESCE(sale_date, order_date, created_at)) as sale_date,
                COUNT(*) as transaction_count,
                SUM(total_amount) as total_sales,
                AVG(total_amount) as avg_transaction
            FROM {table_name}
            WHERE DATE(COALESCE(sale_date, order_date, created_at)) >= CURRENT_DATE - INTERVAL 30 DAY
            GROUP BY DATE(COALESCE(sale_date, order_date, created_at))
            ORDER BY sale_date DESC
            """
        else:
            # Fallback to simple aggregation
            agg_query = f"""
            SELECT 
                COUNT(*) as total_records,
                'Summary' as analysis_type
            FROM {table_name}
            """
        
        return self._add_simple_limit(agg_query.strip(), connection_type, self.config.max_rows_for_llm)
    
    def _generate_time_constraint(self, time_field: str, connection_type: str) -> str:
        """Generate appropriate time constraint for the database type."""
        
        if connection_type.lower() in ['postgresql', 'mysql']:
            return f"{time_field} >= CURRENT_DATE - INTERVAL '7 days'"
        elif connection_type.lower() == 'sqlserver':
            return f"{time_field} >= DATEADD(day, -7, GETDATE())"
        elif connection_type.lower() == 'bigquery':
            return f"{time_field} >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)"
        else:
            # Generic fallback
            return f"{time_field} >= CURRENT_DATE - 7"
    
    def _add_time_constraint(self, sql_query: str, time_constraint: str) -> str:
        """Add time constraint to the WHERE clause."""
        
        if re.search(r'\bWHERE\b', sql_query, re.IGNORECASE):
            # Add to existing WHERE clause
            sql_query = re.sub(
                r'(\bWHERE\b)', 
                f'\\1 {time_constraint} AND', 
                sql_query, 
                flags=re.IGNORECASE
            )
        else:
            # Add new WHERE clause before GROUP BY, ORDER BY, or end
            where_position = -1
            for keyword in ['GROUP BY', 'ORDER BY', 'LIMIT', ';']:
                match = re.search(rf'\b{keyword}\b', sql_query, re.IGNORECASE)
                if match:
                    where_position = match.start()
                    break
            
            if where_position > 0:
                sql_query = sql_query[:where_position] + f" WHERE {time_constraint} " + sql_query[where_position:]
            else:
                sql_query = sql_query.rstrip(';') + f" WHERE {time_constraint}"
        
        return self._add_simple_limit(sql_query, 'postgresql', self.config.max_rows_for_llm)
    
    def _add_hierarchical_grouping(self, sql_query: str, group_field: str, connection_type: str) -> str:
        """Add GROUP BY for hierarchical analysis."""
        
        if not re.search(r'\bGROUP BY\b', sql_query, re.IGNORECASE):
            # Transform SELECT to include grouping
            # This is simplified - you'd want more sophisticated parsing
            
            select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
            if select_match:
                # Replace SELECT with aggregation
                agg_select = f"SELECT {group_field}, COUNT(*) as item_count"
                sql_query = re.sub(
                    r'SELECT\s+.+?\s+FROM',
                    f'{agg_select} FROM',
                    sql_query,
                    flags=re.IGNORECASE | re.DOTALL
                )
                
                # Add GROUP BY
                sql_query = sql_query.rstrip(';') + f" GROUP BY {group_field} ORDER BY item_count DESC"
        
        return self._add_simple_limit(sql_query, connection_type, self.config.max_rows_for_llm)
    
    def _add_simple_limit(self, sql_query: str, connection_type: str, limit: int) -> str:
        """Add appropriate LIMIT clause based on database type using enhanced dialect handling."""
        return self.get_limited_query(sql_query, connection_type, limit)
    
    def get_limited_query(self, raw_sql: str, dialect: str, limit: int = 100) -> str:
        """Enhanced SQL limit handling for different database dialects."""
        sql_clean = raw_sql.strip().rstrip(';')

        if dialect.lower() in ["postgresql", "mysql", "bigquery", "clickhouse", "databricks"]:
            if "limit" not in sql_clean.lower():
                return f"{sql_clean} LIMIT {limit}"
            return sql_clean

        elif dialect.lower() in ["mssql", "sqlserver"]:
            # Only modify if no TOP already
            if "top" not in sql_clean.lower():
                sql_lines = sql_clean.split()
                # Insert TOP after SELECT
                for i, word in enumerate(sql_lines):
                    if word.lower() == "select":
                        sql_lines.insert(i + 1, f"TOP {limit}")
                        break
                return " ".join(sql_lines)
            return sql_clean

        elif dialect.lower() in ["oracle"]:
            # Simplified: Only works for newer versions
            if "fetch first" not in sql_clean.lower():
                return f"{sql_clean} FETCH FIRST {limit} ROWS ONLY"
            return sql_clean

        else:
            # Fallback to LIMIT for unknown dialects
            if "limit" not in sql_clean.lower():
                return f"{sql_clean} LIMIT {limit}"
            return sql_clean
    
    def aggregate_if_needed(self, df: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
        """
        Applies auto-aggregation if the dataset is too large.
        """
        if len(df) <= max_rows:
            return df  # Already small enough

        if 'category' in df.columns:
            return df.groupby('category').agg('sum').reset_index().head(20)

        elif 'warehouse_id' in df.columns and 'stock_level' in df.columns:
            return df.groupby('warehouse_id')['stock_level'].mean().reset_index().head(20)

        else:
            # fallback to random sampling
            return df.sample(n=min(max_rows, len(df)))
    
    def sample_and_sanitize_result(self, agent_result: pd.DataFrame, max_rows: int = 50) -> str:
        """
        Clean, truncate, and format the result as markdown or text for LLMs.
        """
        sampled = self.aggregate_if_needed(agent_result, max_rows)
        try:
            return sampled.to_markdown(index=False)
        except ImportError:
            # Fallback to CSV format if tabulate is not available
            return sampled.to_csv(index=False)
    
    def query_executor_with_sampling(self, raw_sql: str, db_connector, default_limit: int = 100) -> pd.DataFrame:
        """
        Full pipeline: modifies query, executes it, and samples the result.
        """
        safe_sql = self.get_limited_query(raw_sql, "postgresql", default_limit)  # Default dialect
        df = db_connector.run_query(safe_sql)  # must return pandas DataFrame
        return self.aggregate_if_needed(df)
    
    def process_query(self, query: str, user_request: str, connection_type: str, 
                     estimated_rows: int = None) -> Dict[str, Any]:
        """
        Main entry point for processing queries with sampling.
        Returns optimized query and sampling configuration.
        """
        # Apply sampling strategy
        sampling_result = self.apply_sampling_strategy(
            user_request, query, connection_type, estimated_rows
        )
        
        # Create sampling config for metadata
        category, strategy = self.analyze_query_intent(user_request, query)
        config = self._create_sampling_config(strategy, user_request, estimated_rows or 1000)
        
        return {
            "optimized_query": sampling_result.modified_query,
            "original_query": sampling_result.original_query,
            "sampling_config": config,
            "strategy_used": sampling_result.strategy_used.value,
            "estimated_reduction": sampling_result.estimated_reduction,
            "explanation": sampling_result.explanation,
            "category": category
        }
    
    def _create_sampling_config(self, strategy: SamplingStrategy, user_request: str, 
                               estimated_rows: int) -> Dict[str, Any]:
        """Create sampling configuration based on strategy and request."""
        base_config = {
            "strategy": strategy.value,
            "max_rows": self.config.max_rows_for_llm,
            "estimated_rows": estimated_rows
        }
        
        if strategy == SamplingStrategy.AGGREGATION:
            base_config.update({
                "aggregation_functions": ["COUNT(*)", "SUM", "AVG"],
                "group_by_suggested": True
            })
        elif strategy == SamplingStrategy.TOP_N:
            base_config.update({
                "sample_size": self.config.max_rows_for_llm,
                "order_by": "recommended"
            })
        elif strategy == SamplingStrategy.TIME_WINDOW:
            base_config.update({
                "time_window": "7 days",
                "time_column": "created_at"
            })
        elif strategy == SamplingStrategy.RANDOM_SAMPLE:
            base_config.update({
                "sample_size": self.config.max_rows_for_llm,
                "seed": 42
            })
        elif strategy == SamplingStrategy.HIERARCHICAL:
            base_config.update({
                "hierarchy_levels": ["category", "subcategory"],
                "max_per_level": 10
            })
        
        return base_config
    
    def estimate_token_usage(self, row_count: int, avg_row_size: int = 200) -> int:
        """Estimate token usage for given number of rows."""
        # Rough estimation: 4 characters per token, JSON overhead
        estimated_chars = row_count * avg_row_size * 1.3  # 30% JSON overhead
        return int(estimated_chars / 4)
    
    def post_process_results(self, results: Any, sampling_config: Dict[str, Any], 
                           user_query: str) -> Any:
        """
        Post-process results with enhanced handling for pandas DataFrames and large datasets.
        """
        # Handle pandas DataFrame input
        if isinstance(results, pd.DataFrame):
            # Apply aggregation if needed
            processed_df = self.aggregate_if_needed(results, sampling_config.get("max_rows", 100))
            
            # Convert to list of dictionaries for JSON serialization
            processed_results = processed_df.to_dict('records')
            
            # Add sampling metadata
            return {
                "results": processed_results,
                "sampling_info": {
                    "strategy_used": sampling_config.get("strategy", "none"),
                    "original_row_count": len(results),
                    "sampled_row_count": len(processed_results),
                    "reduction_ratio": 1 - (len(processed_results) / len(results)) if len(results) > 0 else 0,
                    "aggregation_applied": len(results) > sampling_config.get("max_rows", 100),
                    "processing_notes": f"Data processed for optimal LLM analysis. Original: {len(results)} rows, Processed: {len(processed_results)} rows"
                },
                "markdown_summary": self.sample_and_sanitize_result(processed_df, 20)  # Top 20 for summary
            }
        
        # Handle list of dictionaries (standard JSON results)
        elif isinstance(results, list):
            max_rows = sampling_config.get("max_rows", 100)
            
            if len(results) <= max_rows:
                return {
                    "results": results,
                    "sampling_info": {
                        "strategy_used": sampling_config.get("strategy", "none"),
                        "row_count": len(results),
                        "is_sampled": False,
                        "processing_notes": f"Dataset is within optimal size ({len(results)} rows)"
                    }
                }
            else:
                # Apply sampling to list
                sampled_results = results[:max_rows]  # Simple truncation
                
                return {
                    "results": sampled_results,
                    "sampling_info": {
                        "strategy_used": sampling_config.get("strategy", "truncation"),
                        "original_row_count": len(results),
                        "sampled_row_count": len(sampled_results),
                        "reduction_ratio": 1 - (len(sampled_results) / len(results)),
                        "is_sampled": True,
                        "processing_notes": f"Large dataset truncated for LLM processing. Showing first {len(sampled_results)} of {len(results)} records"
                    }
                }
        
        # Fallback for other data types
        else:
            return {
                "results": results,
                "sampling_info": {
                    "strategy_used": "none",
                    "processing_notes": "No sampling applied - data type not recognized for optimization"
                }
            }

# Factory function for easy integration
def create_sampling_middleware(max_rows: int = 20, enable_auto: bool = True) -> DataSamplingMiddleware:
    """Create a configured sampling middleware instance."""
    config = SamplingConfig(
        max_rows_for_llm=max_rows,
        enable_auto_sampling=enable_auto
    )
    return DataSamplingMiddleware(config)
