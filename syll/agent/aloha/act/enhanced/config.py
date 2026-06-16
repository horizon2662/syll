"""Configuration for enhanced Aloha Act features.

Reads from ``~/.syll/config.json`` under the ``enhanced`` key.
All flags default to ``True`` so the full enhanced pipeline is
active out of the box even without a config file.

Usage::

    # Everything on (default):
    cfg = EnhancedConfig()

    # Everything off:
    cfg = EnhancedConfig.all_disabled()

    # Pick and choose:
    cfg = EnhancedConfig(enable_tvae_verification=True, enable_gui_subagent=False)

Config file example (``~/.syll/config.json``)::

    {
      "enhanced": {
        "allEnabled": true,
        "enable_tvae_verification": true,
        "enable_verified_planner": true,
        "enable_prompt_delta": true,
        "enable_plan_persistence": true,
        "enable_structured_memory": true,
        "enable_gui_subagent": true,
        "enable_semantic_trace": true,
        "enable_spatial_context": true,
        "max_consecutive_failures": 3,
        "screenshot_delay_seconds": 1.0,
        "pixel_diff_threshold": 0.005
      }
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
    enable_tvae_verification: bool = True  # ActionVerifier pixel-diff + LLM check
    enable_verified_planner: bool = True   # VerifiedPlanner (inherits AlohaPlanner)
    enable_prompt_delta: bool = True       # Action-type-specific prompt hints

    # ---------- Phase 2: Structured planning & memory ----------
    enable_plan_persistence: bool = True   # PlanManager (md file per skill)
    enable_structured_memory: bool = True  # StructuredMemory (trajectory compression)

    # ---------- Phase 3: Sub-agent execution ----------
    enable_gui_subagent: bool = True       # GUIExecuteSubAgent (isolated TVAE loop)

    # ---------- Phase 4: Semantic trace & spatial analysis ----------
    enable_semantic_trace: bool = True     # EnhancedTraceGenerator
    enable_spatial_context: bool = True    # SpatialAnalyzer (VLM UI description)

    # ---------- Tunables ----------
    max_consecutive_failures: int = 3
    screenshot_delay_seconds: float = 1.0
    pixel_diff_threshold: float = 0.005

    # ------------------------------------------------------------------
    # Constructor: merge config.json values before applying overrides
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """When ALL_ENABLED is False, force every feature flag off."""
        if not self.ALL_ENABLED:
            self.enable_tvae_verification = False
            self.enable_verified_planner = False
            self.enable_prompt_delta = False
            self.enable_plan_persistence = False
            self.enable_structured_memory = False
            self.enable_gui_subagent = False
            self.enable_semantic_trace = False
            self.enable_spatial_context = False

    @classmethod
    def from_config_file(cls, **overrides) -> EnhancedConfig:
        """Construct an EnhancedConfig by reading ``~/.syll/config.json``.

        Explicit keyword arguments take highest priority, then config file
        values, then built-in defaults.
        """
        file_vals = _load_enhanced_json()

        # Map JSON key "allEnabled" → Python field "ALL_ENABLED"
        if "allEnabled" in file_vals:
            file_vals["ALL_ENABLED"] = file_vals.pop("allEnabled")

        # Config file values fill in gaps; explicit overrides win
        merged = {**file_vals, **overrides}

        # Filter to only known fields
        known = {
            "ALL_ENABLED",
            "enable_tvae_verification",
            "enable_verified_planner",
            "enable_prompt_delta",
            "enable_plan_persistence",
            "enable_structured_memory",
            "enable_gui_subagent",
            "enable_semantic_trace",
            "enable_spatial_context",
            "max_consecutive_failures",
            "screenshot_delay_seconds",
            "pixel_diff_threshold",
        }
        filtered = {k: v for k, v in merged.items() if k in known}

        return cls(**filtered)

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

    @property
    def tvae_active(self) -> bool:
        """Shortcut: are TVAE verification components active?"""
        return self.enable_tvae_verification and self.enable_verified_planner

    @property
    def phase2_active(self) -> bool:
        """Shortcut: are plan persistence and structured memory active?"""
        return self.enable_plan_persistence and self.enable_structured_memory

    @property
    def phase4_active(self) -> bool:
        """Shortcut: are semantic trace and spatial analysis active?"""
        return self.enable_semantic_trace and self.enable_spatial_context
