"""Configuration for the enhanced Aloha Act pipeline.

Reads the ``enhanced`` section of ``~/.syll/config.json`` (see that file for
the full key list). All flags default to ``True`` so the pipeline is active
out of the box; set ``"allEnabled": false`` to disable everything.

Usage::

    EnhancedConfig()                          # everything on (default)
    EnhancedConfig.all_disabled()             # everything off
    EnhancedConfig(enable_llm_verify=False)   # pick and choose
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import get_type_hints

_CONFIG_FILE = Path.home() / ".syll" / "config.json"


def _load_enhanced_json() -> dict:
    """Load the ``enhanced`` section from ``~/.syll/config.json``.

    Returns an empty dict if the file or section is missing.
    """
    try:
        if _CONFIG_FILE.exists():
            cfg = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            return cfg.get("enhanced", {})
    except Exception:
        pass
    return {}


@dataclass
class EnhancedConfig:
    """Feature flags and thresholds for the enhanced Aloha Act pipeline.

    All boolean flags default to ``True`` so the full enhanced pipeline is
    active out of the box.  Values are first read from ``config.json``
    (under the ``enhanced`` key), then overridden by any explicit constructor
    arguments.

    Set ``allEnabled: false`` in config (or use :meth:`all_disabled`) to
    revert to the original behaviour.
    """

    # ---------- master switch ----------
    ALL_ENABLED: bool = True

    # ---------- Phase 1: TVAE verification ----------
    enable_tvae_verification: bool = True
    enable_llm_verify: bool = True
    enable_prompt_delta: bool = True

    # ---------- Phase 2: Structured planning & memory ----------
    enable_plan_persistence: bool = True
    enable_structured_memory: bool = True

    # ---------- Phase 4: Semantic trace & spatial analysis ----------
    enable_semantic_trace: bool = True
    enable_spatial_context: bool = True

    # ---------- Tunables ----------
    max_consecutive_failures: int = 3
    screenshot_delay_seconds: float = 1.0
    pixel_diff_threshold: float = 0.005

    def __post_init__(self) -> None:
        # ALL_ENABLED=False forces every boolean feature flag off (tunables kept).
        if not self.ALL_ENABLED:
            hints = get_type_hints(type(self))
            for name, typ in hints.items():
                if name != "ALL_ENABLED" and typ is bool:
                    setattr(self, name, False)

    @classmethod
    def from_config_file(cls, **overrides) -> EnhancedConfig:
        """Construct an EnhancedConfig by reading ``~/.syll/config.json``.

        Explicit keyword arguments take highest priority, then config file
        values, then built-in defaults.
        """
        file_vals = _load_enhanced_json()
        # JSON key "allEnabled" maps to the ALL_ENABLED field.
        if "allEnabled" in file_vals:
            file_vals["ALL_ENABLED"] = file_vals.pop("allEnabled")

        known = {f.name for f in fields(cls)}
        merged = {**file_vals, **overrides}
        return cls(**{k: v for k, v in merged.items() if k in known})

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def all_enabled(cls) -> EnhancedConfig:
        """Return a config with every feature turned on."""
        return cls(ALL_ENABLED=True)

    @classmethod
    def all_disabled(cls) -> EnhancedConfig:
        """Return a config with every feature turned off (original behaviour)."""
        return cls(ALL_ENABLED=False)

    @classmethod
    def from_gui_agent_config(cls, cfg: "GuiAgentConfig") -> EnhancedConfig:
        """Build an EnhancedConfig from the canonical schema config.

        This bridges the old standalone ``EnhancedConfig`` dataclass with the
        new ``tools.gui.agent`` schema field, so existing code that consumes
        EnhancedConfig continues to work while configuration converges.
        """
        return cls(
            ALL_ENABLED=cfg.all_enabled,
            enable_tvae_verification=cfg.enable_tvae_verification,
            enable_llm_verify=cfg.enable_llm_verify,
            enable_prompt_delta=cfg.enable_prompt_delta,
            enable_plan_persistence=cfg.enable_plan_persistence,
            enable_structured_memory=cfg.enable_structured_memory,
            enable_semantic_trace=cfg.enable_semantic_trace,
            enable_spatial_context=cfg.enable_spatial_context,
            max_consecutive_failures=cfg.max_consecutive_failures,
            screenshot_delay_seconds=cfg.screenshot_delay_seconds,
            pixel_diff_threshold=cfg.pixel_diff_threshold,
        )
