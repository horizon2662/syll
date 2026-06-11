"""Configuration for enhanced Aloha Act features.

Set ``ALL_ENABLED = False`` below to deactivate every enhanced feature.
Individual flags can still be overridden after construction.

Usage::

    # Everything on (default):
    cfg = EnhancedConfig()

    # Everything off:
    cfg = EnhancedConfig.all_disabled()

    # Pick and choose:
    cfg = EnhancedConfig(enable_tvae_verification=True, enable_gui_subagent=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnhancedConfig:
    """Feature flags and thresholds for the enhanced Aloha Act pipeline.

    All boolean flags default to ``True`` so the full enhanced pipeline is
    active out of the box.  Set ``ALL_ENABLED = False`` at the module level
    (or use :meth:`all_disabled`) to revert to the original behaviour.
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
    # Convenience constructors
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
