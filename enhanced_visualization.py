"""
Enhanced Visualization for MCP Chat Interface
Handles dynamic chart rendering based on LLM recommendations
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Any, Optional

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

def render_data_visualization(data: List[Dict[str, Any]], visualization_spec: Dict[str, Any]) -> None:
    """
    Render data visualization based on LLM-generated specifications
    
    Args:
        data: List of dictionaries containing the query results
        visualization_spec: Specification from LLM containing output_format and chart_spec
    """
    if not data:
        st.warning("No data to visualize")
        return
    
    # Convert to DataFrame
    try:
        df = pd.DataFrame(data)
        if df.empty:
            st.warning("Empty dataset")
            return
    except Exception as e:
        st.error(f"Error converting data to DataFrame: {e}")
        return
    
    # Ensure visualization_spec is a valid dictionary
    if visualization_spec is None:
        visualization_spec = {}
    
    output_formats = visualization_spec.get("output_format", ["table"])
    chart_spec = visualization_spec.get("chart_spec", {})
    
    # Ensure chart_spec is never None
    if chart_spec is None:
        chart_spec = {}
    
    # Always show table first (good for debugging)
    if "table" in output_formats:
        # Extra safety check for chart_spec
        title = chart_spec.get("title", "Data Results") if chart_spec else "Data Results"
        render_data_table(df, title)
    
    # Render charts based on recommendations
    if "metric" in output_formats:
        render_metric_display(df, chart_spec)
    
    if "line_chart" in output_formats:
        render_line_chart(df, chart_spec)
    
    if "bar_chart" in output_formats:
        render_bar_chart(df, chart_spec)
    
    if "pie_chart" in output_formats:
        render_pie_chart(df, chart_spec)
    
    # Show chart specification for debugging (if debug mode is on)
    if st.session_state.get('show_debug', False):
        with st.expander("🔍 Visualization Debug", expanded=False):
            st.json({
                "output_formats": output_formats,
                "chart_spec": chart_spec,
                "data_shape": df.shape,
                "columns": list(df.columns)
            })

def render_data_table(df: pd.DataFrame, title: str = "Data Table") -> None:
    """Render a data table with enhanced formatting"""
    st.subheader(f"📋 {title}")
    
    # Show data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        # Show memory usage if available
        try:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory", f"{memory_mb:.1f} MB")
        except:
            st.metric("Memory", "N/A")
    
    # Display the full table - don't truncate for token generation trends
    st.dataframe(df, use_container_width=True, height=400)
    
    # For debugging: show column types and sample data
    if st.session_state.get('show_debug', False):
        with st.expander("🔍 Data Debug Info"):
            st.write("**Column Types:**")
            st.write(df.dtypes)
            st.write("**Sample Data (first 3 rows):**")
            st.write(df.head(3))

def render_metric_display(df: pd.DataFrame, chart_spec: Dict[str, Any]) -> None:
    """Render metrics for KPI-style data"""
    # Ensure chart_spec is not None
    if chart_spec is None:
        chart_spec = {}
        
    st.subheader(f"📊 {chart_spec.get('title', 'Key Metrics')}")
    
    # Try to find numeric columns for metrics
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found for metrics display")
        return
    
    # Create metric display
    cols = st.columns(min(len(numeric_cols), 4))  # Max 4 columns
    
    for i, col_name in enumerate(numeric_cols[:4]):  # Limit to 4 metrics
        with cols[i % len(cols)]:
            value = df[col_name].iloc[0] if len(df) == 1 else df[col_name].sum()
            
            # Format the value nicely
            if isinstance(value, float):
                if value > 1000000:
                    display_value = f"{value/1000000:.1f}M"
                elif value > 1000:
                    display_value = f"{value/1000:.1f}K"
                else:
                    display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            
            st.metric(
                label=col_name.replace("_", " ").title(),
                value=display_value
            )

def render_line_chart(df: pd.DataFrame, chart_spec: Dict[str, Any]) -> None:
    """Render line chart using Altair or Streamlit fallback"""
    # Ensure chart_spec is not None
    if chart_spec is None:
        chart_spec = {}
        
    title = chart_spec.get("title", "Line Chart")
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    
    st.subheader(f"📈 {title}")
    
    if not x_col or not y_col:
        # Auto-detect columns
        cols = df.columns.tolist()
        if len(cols) >= 2:
            x_col = cols[0]
            y_col = cols[1]
        else:
            st.warning("Need at least 2 columns for line chart")
            return
    
    # Validate columns exist
    if x_col not in df.columns or y_col not in df.columns:
        st.warning(f"Columns {x_col} or {y_col} not found in data")
        return
    
    if HAS_ALTAIR:
        try:
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X(x_col, title=x_col.replace("_", " ").title()),
                    y=alt.Y(y_col, title=y_col.replace("_", " ").title()),
                    tooltip=[x_col, y_col]
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering Altair chart: {e}")
            # Fallback to Streamlit
            st.line_chart(df.set_index(x_col)[y_col])
    else:
        # Streamlit fallback
        try:
            st.line_chart(df.set_index(x_col)[y_col])
        except Exception as e:
            st.error(f"Error rendering line chart: {e}")

def render_bar_chart(df: pd.DataFrame, chart_spec: Dict[str, Any]) -> None:
    """Render bar chart using Altair or Streamlit fallback"""
    # Ensure chart_spec is not None
    if chart_spec is None:
        chart_spec = {}
        
    title = chart_spec.get("title", "Bar Chart")
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    
    st.subheader(f"📊 {title}")
    
    if not x_col or not y_col:
        # Auto-detect columns
        cols = df.columns.tolist()
        if len(cols) >= 2:
            x_col = cols[0]
            y_col = cols[1]
        else:
            st.warning("Need at least 2 columns for bar chart")
            return
    
    # Validate columns exist
    if x_col not in df.columns or y_col not in df.columns:
        st.warning(f"Columns {x_col} or {y_col} not found in data")
        return
    
    if HAS_ALTAIR:
        try:
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X(x_col, title=x_col.replace("_", " ").title()),
                    y=alt.Y(y_col, title=y_col.replace("_", " ").title()),
                    tooltip=[x_col, y_col]
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering Altair chart: {e}")
            # Fallback to Streamlit
            st.bar_chart(df.set_index(x_col)[y_col])
    else:
        # Streamlit fallback
        try:
            st.bar_chart(df.set_index(x_col)[y_col])
        except Exception as e:
            st.error(f"Error rendering bar chart: {e}")

def render_pie_chart(df: pd.DataFrame, chart_spec: Dict[str, Any]) -> None:
    """Render pie chart using Altair"""
    # Ensure chart_spec is not None
    if chart_spec is None:
        chart_spec = {}
        
    title = chart_spec.get("title", "Pie Chart")
    
    st.subheader(f"🥧 {title}")
    
    if not HAS_ALTAIR:
        st.warning("Pie charts require Altair. Please install: pip install altair")
        return
    
    # For pie charts, we typically need a category and a value
    cols = df.columns.tolist()
    if len(cols) < 2:
        st.warning("Need at least 2 columns for pie chart")
        return
    
    # Auto-detect or use specified columns
    category_col = chart_spec.get("x", cols[0])
    value_col = chart_spec.get("y", cols[1])
    
    # Validate columns exist
    if category_col not in df.columns or value_col not in df.columns:
        st.warning(f"Columns {category_col} or {value_col} not found in data")
        return
    
    try:
        chart = (
            alt.Chart(df)
            .mark_arc()
            .encode(
                theta=alt.Theta(value_col, title=value_col.replace("_", " ").title()),
                color=alt.Color(category_col, title=category_col.replace("_", " ").title()),
                tooltip=[category_col, value_col]
            )
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering pie chart: {e}")

def render_enhanced_results(results: List[Dict[str, Any]]) -> None:
    """
    Render enhanced results with automatic visualization
    
    Args:
        results: List of agent execution results with visualization metadata
    """
    if not results:
        st.warning("No results to display")
        return
    
    for i, result in enumerate(results):
        if not result.get("success"):
            continue
            
        agent_name = result.get("agent_name", f"Agent {i+1}")
        data = result.get("data", [])
        visualization = result.get("visualization", {})
        
        # Ensure visualization is a valid dict
        if visualization is None:
            visualization = {}
        
        # Create a section for each agent's results
        if len(results) > 1:
            st.markdown(f"### {agent_name} Results")
        
        if data:
            try:
                # Debug: Show what data we're working with
                print(f"[DEBUG] Rendering data for {agent_name}: {len(data)} records")
                if len(data) > 0:
                    print(f"[DEBUG] Sample record: {data[0]}")
                    print(f"[DEBUG] Visualization spec: {visualization}")
                
                render_data_visualization(data, visualization)
                
                # Show raw data preview if debug mode is enabled
                if st.session_state.get('show_debug', False):
                    with st.expander(f"🔍 Raw Data Preview for {agent_name}"):
                        st.write(f"**Total Records:** {len(data)}")
                        st.write(f"**Visualization Spec:** {visualization}")
                        if data:
                            st.write("**First 5 Records:**")
                            st.json(data[:5])
                        
            except Exception as e:
                st.error(f"Error rendering visualization for {agent_name}: {e}")
                st.write("**Raw Data (fallback display):**")
                # Show all data, not just first 5 for token trends
                if len(data) <= 20:  # Show all if reasonable size
                    st.json(data)
                else:
                    st.write(f"**Total records:** {len(data)}")
                    st.write("**First 10 records:**")
                    st.json(data[:10])
                    st.write("**Last 5 records:**")
                    st.json(data[-5:])
        else:
            st.warning(f"No data returned from {agent_name}")
        
        # Add separator if multiple results
        if i < len(results) - 1:
            st.markdown("---")

def install_requirements():
    """Display installation requirements for enhanced visualization"""
    st.info("""
    **Enhanced Visualization Requirements:**
    
    For the best visualization experience, install these packages:
    ```bash
    pip install altair pandas
    ```
    
    Current status:
    - pandas: ✅ Available
    - altair: {'✅ Available' if HAS_ALTAIR else '❌ Not installed'}
    """)
