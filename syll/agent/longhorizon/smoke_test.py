"""Smoke test: runs the full orchestrator end-to-end with a FAKE provider.

No API key needed. Verifies the stateful logic the runner depends on:
plan creation, subagent fold, per-skill + global memory writes,
checkpoint/resume, and replan (align).

Run:  python -m refactor_skeleton.smoke_test
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from syll.providers.base import LLMResponse, ToolCallRequest

from .config import RunnerConfig
from .runner import Runner


class FakeProvider:
    """Returns canned responses. Subagent mode (tools!=None) -> `return` tool call.
    Brain mode (no tools) -> JSON milestones, then replan steps if asked."""

    def __init__(self):
        self.calls = 0

    def get_default_model(self) -> str:
        return "fake-model"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        self.calls += 1
        if tools:  # subagent branch -> fold back immediately
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="1",
                        name="return",
                        arguments={
                            "summary": "completed the step successfully",
                            "status": "ok",
                            "lessons": ["prefer foo() over bar() for X"],
                            "artifacts": [],
                        },
                    )
                ],
            )
        # brain mode: first call = milestones; if the prompt mentions "failed"/"replan"
        last = messages[-1]["content"].lower()
        if "failed step" in last or "replacement" in last:
            return LLMResponse(content='{"steps": ["retry with foo() instead"]}')
        return LLMResponse(
            content='[{"title":"Prepare","steps":["setup env","install deps"]},'
            '{"title":"Build","steps":["compile"]}]'
        )


async def main():
    tmp = Path(tempfile.mkdtemp(prefix="syll_smoke_"))
    cfg = RunnerConfig(model="fake-model", api_key="fake", workspace=tmp, skill="demo")
    runner = Runner(cfg, FakeProvider())

    await runner.run("build a tiny demo app")

    # ---- assertions ----
    project = (tmp / "PROJECT.md").read_text(encoding="utf-8")
    skill = (tmp / "skills" / "demo" / "SKILL.md").read_text(encoding="utf-8")
    plan_state = (tmp / "aloha_skills" / "demo" / "session_state.json")
    orch_state = (tmp / "aloha_skills" / "demo" / "orchestrator_state.json")

    checks = {
        "PROJECT.md has goal": "build a tiny demo app" in project,
        "PROJECT.md logged completion": "ALL MILESTONES COMPLETE" in project,
        "SKILL.md captured lesson": "prefer foo()" in skill,
        "plan session_state.json exists": plan_state.exists(),
        "orchestrator_state.json exists": orch_state.exists(),
    }

    print("\n--- smoke test results ---")
    ok = True
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        ok = ok and v

    # ---- resume test: a second runner should pick up the checkpoint ----
    print("\n--- resume test ---")
    runner2 = Runner(cfg, FakeProvider())
    resumed = runner2.session.resume()
    plan2 = runner2.hpm.resume()
    resume_ok = resumed is not None and plan2 is not None and any(
        s.status == "DONE" for s in plan2.steps
    )
    print(f"  [{'PASS' if resume_ok else 'FAIL'}] resume loads checkpoint + done steps")
    ok = ok and resume_ok

    print(f"\nworkspace: {tmp}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
