"""验证 enhanced 模块的导入和基本功能。

运行方式:
    cd c:\Users\11320\.venv\Lib\site-packages
    python -m syll.agent.aloha.act.enhanced.verify
"""

import sys
import tempfile
from pathlib import Path

# 确保正确的 site-packages 在 path 中
site_packages = str(Path(__file__).parent.parent.parent.parent.parent.parent.parent)
if site_packages not in sys.path:
    sys.path.insert(0, site_packages)

results = []


def test(name: str, fn) -> bool:
    """运行一个测试并记录结果。"""
    try:
        fn()
        results.append((name, "✅ PASS", ""))
        print(f"  ✅ {name}")
        return True
    except Exception as e:
        results.append((name, "❌ FAIL", str(e)))
        print(f"  ❌ {name}: {e}")
        return False


def main():
    print("=" * 60)
    print("Enhanced Aloha Act Modules — Verification")
    print("=" * 60)

    # ---- Phase 0: Config ----
    print("\n[Phase 0] Config")
    test("EnhancedConfig import", lambda: __import__(
        "syll.agent.aloha.act.enhanced.config", fromlist=["EnhancedConfig"]
    ))
    from syll.agent.aloha.act.enhanced.config import EnhancedConfig
    test("EnhancedConfig defaults all True", lambda: (
        cfg := EnhancedConfig(),
        all([
            cfg.enable_tvae_verification,
            cfg.enable_plan_persistence,
            cfg.enable_structured_memory,
            cfg.enable_gui_subagent,
            cfg.enable_prompt_delta,
            cfg.enable_spatial_context,
            cfg.enable_semantic_trace,
        ])[-1]
    ))
    test("EnhancedConfig.all_enabled()", lambda: (
        cfg := EnhancedConfig.all_enabled(),
        all([
            cfg.enable_tvae_verification,
            cfg.enable_plan_persistence,
            cfg.enable_structured_memory,
            cfg.enable_gui_subagent,
            cfg.enable_prompt_delta,
            cfg.enable_spatial_context,
            cfg.enable_semantic_trace,
        ])[-1]
    ))
    test("EnhancedConfig.all_disabled()", lambda: (
        cfg := EnhancedConfig.all_disabled(),
        not any([
            cfg.enable_tvae_verification,
            cfg.enable_plan_persistence,
            cfg.enable_structured_memory,
            cfg.enable_gui_subagent,
            cfg.enable_prompt_delta,
            cfg.enable_spatial_context,
            cfg.enable_semantic_trace,
        ])[-1]
    ))

    # ---- Phase 1: TVAE ----
    print("\n[Phase 1] TVAE Verification")
    test("ActionVerifier import", lambda: __import__(
        "syll.agent.aloha.act.enhanced.action_verifier", fromlist=["ActionVerifier"]
    ))
    from syll.agent.aloha.act.enhanced.action_verifier import (
        ActionVerifier, VerifyResult, VerifyStatus,
    )
    test("ActionVerifier instantiation", lambda: ActionVerifier())
    test("VerifyStatus enum values", lambda: (
        VerifyStatus.SUCCESS.value == "SUCCESS"
        and VerifyStatus.NO_CHANGE.value == "NO_CHANGE"
        and VerifyStatus.UNCERTAIN.value == "UNCERTAIN"
    ))
    test("VerifiedPlanner import", lambda: __import__(
        "syll.agent.aloha.act.enhanced.verified_planner",
        fromlist=["VerifiedPlanner"],
    ))
    from syll.agent.aloha.act.enhanced.verified_planner import VerifiedPlanner
    test("VerifiedPlanner inherits AlohaPlanner", lambda: issubclass(
        VerifiedPlanner,
        __import__(
            "syll.agent.aloha.act.planner",
            fromlist=["AlohaPlanner"],
        ).AlohaPlanner,
    ))
    test("VerifiedPlanner instantiation", lambda: VerifiedPlanner(model="test"))

    # ---- Phase 1: Prompt templates ----
    print("\n[Phase 1] Prompt Templates")
    prompt_dir = Path(__file__).parent / "prompt_templates" / "verified_planner"
    test("system.txt exists", lambda: (
        (prompt_dir / "system.txt").exists()
    ))
    test("user.txt exists", lambda: (
        (prompt_dir / "user.txt").exists()
    ))
    test("system.txt contains TVAE", lambda: (
        "TVAE" in (prompt_dir / "system.txt").read_text()
    ))
    test("user.txt contains Expectation", lambda: (
        "Expectation" in (prompt_dir / "user.txt").read_text()
    ))

    # ---- Phase 1: Prompt deltas ----
    print("\n[Phase 1] Prompt Deltas")
    delta_dir = Path(__file__).parent / "prompt_deltas"
    for name in ("click", "drag", "type", "scroll"):
        test(f"{name}_delta.txt exists", lambda n=name: (
            delta_dir / f"{n}_delta.txt"
        ).exists())

    # ---- Phase 2: Plan persistence ----
    print("\n[Phase 2] Plan Manager + Structured Memory")
    test("PlanManager import", lambda: __import__(
        "syll.agent.aloha.act.enhanced.plan_manager",
        fromlist=["PlanManager"],
    ))
    from syll.agent.aloha.act.enhanced.plan_manager import (
        PlanManager, ExecutionPlan, PlanStep,
    )
    test("PlanManager create_plan", lambda: (
        pm := PlanManager(workspace=Path(tempfile.mkdtemp())),
        plan := pm.create_plan("test_skill", "Test task", ["Step 1", "Step 2"]),
        len(plan.steps) == 2,
    )[-1])
    test("PlanManager save/load roundtrip", lambda: (
        pm := PlanManager(workspace=Path(tempfile.mkdtemp())),
        plan := pm.create_plan("test", "Task", ["A", "B", "C"]),
        path := pm.save_plan(plan),
        loaded := pm.load_plan("test"),
        loaded is not None and len(loaded.steps) == 3,
    )[-1])
    test("PlanManager update_step", lambda: (
        pm := PlanManager(workspace=Path(tempfile.mkdtemp())),
        plan := pm.create_plan("test", "Task", ["A", "B"]),
        pm.update_step(plan, 1, "DONE", "OK"),
        plan.steps[0].status == "DONE" and plan.steps[1].status == "CURRENT",
    )[-1])
    test("PlanManager get_current_step", lambda: (
        pm := PlanManager(workspace=Path(tempfile.mkdtemp())),
        plan := pm.create_plan("test", "Task", ["A", "B", "C"]),
        step := pm.get_current_step(plan),
        step.index == 1,
    )[-1])

    # Structured Memory
    test("StructuredMemory import", lambda: __import__(
        "syll.agent.aloha.act.enhanced.structured_memory",
        fromlist=["StructuredMemory"],
    ))
    from syll.agent.aloha.act.enhanced.structured_memory import StructuredMemory
    test("StructuredMemory inherits MemoryStore", lambda: issubclass(
        StructuredMemory,
        __import__("syll.agent.memory", fromlist=["MemoryStore"]).MemoryStore,
    ))
    test("StructuredMemory record + get_context", lambda: (
        mem := StructuredMemory(workspace=Path(tempfile.mkdtemp())),
        mem.record_step(1, "Click File", "Menu opens", "SUCCESS"),
        mem.record_step(2, "Click Save", "Dialog opens", "NO_CHANGE", "Missed"),
        ctx := mem.get_execution_context(max_recent_steps=2),
        "Step 1" in ctx and "Step 2" in ctx,
    )[-1])

    # ---- Phase 3: GUI Execute Sub-Agent ----
    print("\n[Phase 3] GUI Execute Sub-Agent")
    test("GUIExecuteSubAgent import", lambda: __import__(
        "syll.agent.aloha.act.enhanced.gui_execute_subagent",
        fromlist=["GUIExecuteSubAgent", "StepResult"],
    ))
    from syll.agent.aloha.act.enhanced.gui_execute_subagent import (
        GUIExecuteSubAgent, StepResult,
    )
    test("StepResult dataclass", lambda: (
        r := StepResult(step_index=1, action_success=True, retries_used=0),
        r.step_index == 1 and r.action_success,
    )[-1])

    # ---- Phase 4: Enhanced Trace Generator ----
    print("\n[Phase 4] Enhanced Trace + Spatial Analyzer")
    test("EnhancedTraceGenerator import", lambda: __import__(
        "syll.agent.aloha.act.enhanced.enhanced_trace_generator",
        fromlist=["EnhancedTraceGenerator"],
    ))
    test("SpatialAnalyzer import", lambda: __import__(
        "syll.agent.aloha.act.enhanced.spatial_analyzer",
        fromlist=["SpatialAnalyzer"],
    ))
    from syll.agent.aloha.act.enhanced.enhanced_trace_generator import (
        EnhancedTraceGenerator,
        _looks_like_coordinates,
    )
    test("_looks_like_coordinates heuristic", lambda: (
        _looks_like_coordinates("click(500, 30)") is True
        and _looks_like_coordinates("Click the File menu") is False
        and _looks_like_coordinates("[100,200]") is True
    ))

    # ---- Summary ----
    print("\n" + "=" * 60)
    passed = sum(1 for _, status, _ in results if status == "✅ PASS")
    failed = sum(1 for _, status, _ in results if status == "❌ FAIL")
    total = len(results)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        print("\nFailed tests:")
        for name, status, error in results:
            if status == "❌ FAIL":
                print(f"  {name}: {error}")
    print("=" * 60)


if __name__ == "__main__":
    main()
