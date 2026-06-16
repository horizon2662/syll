"""Enhanced Aloha Act modules — backward-compatible extensions.

All modules here extend the original aloha/act components via inheritance
or composition. The original code is never modified directly.

Enable features via EnhancedConfig (all flags default to True).
"""

from syll.agent.aloha.act.enhanced.config import EnhancedConfig

__all__ = ["EnhancedConfig"]
