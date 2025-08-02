"""
Registry module - re-exports from client for backwards compatibility.
"""

from .client import fetch_agents_and_tools_from_registry

__all__ = ['fetch_agents_and_tools_from_registry']