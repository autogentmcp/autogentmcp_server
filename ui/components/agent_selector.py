"""
Agent selection UI component
"""

import streamlit as st
from typing import Dict, List, Any
from datetime import datetime

from ..styles.styles import AGENT_SELECTION_STYLES


class AgentSelector:
    """Component for rendering agent selection interface"""
    
    def render_agent_selection_ui(self, workflow_result: Dict[str, Any]) -> bool:
        """
        Render a user-friendly agent selection interface when clarification is needed.
        
        Args:
            workflow_result: The workflow result containing agent options
            
        Returns:
            True if user made a selection, False otherwise
        """
        user_interface = workflow_result.get("user_interface", {})
        available_agents = workflow_result.get("available_agents", [])
        
        if user_interface.get("type") == "agent_selection" and available_agents:
            # Apply agent selection styles
            st.markdown(AGENT_SELECTION_STYLES, unsafe_allow_html=True)
            
            st.markdown("### 🎯 Available Agents")
            st.markdown("I found multiple agents that could help with your request. Please select the one that best fits your needs:")
            
            # Create selection interface
            options = user_interface.get("options", [])
            
            for i, option in enumerate(options):
                agent_name = option.get("name", "Unknown Agent")
                description = option.get("description", "No description available")
                best_for = option.get("best_for", "")
                confidence = option.get("recommendation_confidence", 0.5)
                
                # Determine if this is a recommended agent
                is_recommended = confidence > 0.7
                
                # Create an expander for each agent option
                with st.expander(
                    f"🤖 {agent_name} {'⭐' if is_recommended else ''}", 
                    expanded=(i == 0 and is_recommended)
                ):
                    # Agent card content
                    st.markdown(
                        f"""
                        <div class="agent-card {'recommended' if is_recommended else ''}">
                            <div class="agent-name">{agent_name}</div>
                            <div class="agent-description">{description}</div>
                            {f'<div class="agent-description"><strong>Best for:</strong> {best_for}</div>' if best_for else ''}
                            <div class="agent-confidence">Confidence: {confidence:.1%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Selection button
                    button_type = "primary" if is_recommended else "secondary"
                    if st.button(
                        f"Select {agent_name}", 
                        key=f"select_agent_{i}", 
                        type=button_type
                    ):
                        # Store the selected agent choice
                        st.session_state.agent_selection = {
                            "selected_agent": agent_name,
                            "agent_index": i,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.success(f"✅ Selected: **{agent_name}**")
                        st.rerun()
                        return True
            
            st.markdown("---")
            st.info("💡 **Tip**: Click on an agent above to see more details, then use the 'Select' button to proceed.")
            return False
        
        return False
    
    def get_selected_agent(self) -> Dict[str, Any]:
        """Get the currently selected agent from session state"""
        return st.session_state.get('agent_selection', {})
    
    def clear_selection(self):
        """Clear the current agent selection"""
        if 'agent_selection' in st.session_state:
            del st.session_state.agent_selection
