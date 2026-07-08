"""GUI configuration re-exports.

The canonical schema lives in ``syll.config.schema``; this module re-exports it
so that ``syll.agent.gui`` can be read as a self-contained layer.
"""

from __future__ import annotations

from syll.agent.aloha.act.enhanced.config import EnhancedConfig
from syll.config.schema import GuiAgentConfig, GuiConfig


def enhanced_config_from_gui_config(gui_config: GuiConfig) -> EnhancedConfig:
    """Build the legacy ``EnhancedConfig`` from the canonical ``GuiConfig``.

    Bridges the old standalone config section with the new schema location
    (``tools.gui.agent``). Falls back to the legacy config file when the schema
    agent block is missing.
    """
    agent_cfg = getattr(gui_config, "agent", None)
    if agent_cfg is not None:
        return EnhancedConfig.from_gui_agent_config(agent_cfg)
    return EnhancedConfig.from_config_file()


__all__ = ["EnhancedConfig", "GuiAgentConfig", "GuiConfig", "enhanced_config_from_gui_config"]
