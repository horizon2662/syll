"""Skill document edit operations — adapted from SkillOpt's ReflACT pipeline.

Provides atomic edit primitives (append, insert_after, replace, delete) and
patch application for Markdown skill documents.  Includes a SLOW_UPDATE
protected region mechanism so optimizer edits cannot clobber stable content
that has been explicitly locked by a previous slow-update epoch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

EditOp = Literal["append", "insert_after", "replace", "delete"]

SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"


# ── Edit data types ──────────────────────────────────────────────────────────


@dataclass
class Edit:
    """A single edit operation on a skill document."""

    op: EditOp
    content: str = ""
    target: str = ""
    source_type: Literal["failure", "success"] | None = None
    merge_level: int | None = None
    support_count: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Edit:
        return cls(
            op=d.get("op", "append"),
            content=d.get("content", ""),
            target=d.get("target", ""),
            source_type=d.get("source_type"),
            merge_level=d.get("merge_level"),
            support_count=d.get("support_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op, "content": self.content}
        if self.target:
            d["target"] = self.target
        if self.source_type is not None:
            d["source_type"] = self.source_type
        if self.merge_level is not None:
            d["merge_level"] = self.merge_level
        if self.support_count is not None:
            d["support_count"] = self.support_count
        return d


@dataclass
class Patch:
    """A set of edits with reasoning — output of Aggregate, input to Update."""

    edits: list[Edit] = field(default_factory=list)
    reasoning: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Patch:
        edits_raw = d.get("edits", [])
        return cls(
            edits=[Edit.from_dict(e) if isinstance(e, dict) else e for e in edits_raw],
            reasoning=d.get("reasoning", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "edits": [e.to_dict() if isinstance(e, Edit) else e for e in self.edits],
        }


# ── Gate result ─────────────────────────────────────────────────────────────


GateAction = Literal["accept_new_best", "accept", "reject"]


@dataclass(frozen=True)
class GateResult:
    """Outcome of the validation gate for one optimization cycle."""

    action: GateAction
    candidate_score: float
    current_score: float
    best_score: float
    best_step: int


# ── Protected region helpers ─────────────────────────────────────────────


def _is_in_slow_update_region(skill: str, target: str) -> bool:
    """Check if *target* text falls within the protected slow-update region."""
    start_idx = skill.find(SLOW_UPDATE_START)
    end_idx = skill.find(SLOW_UPDATE_END)
    if start_idx == -1 or end_idx == -1:
        return False
    target_idx = skill.find(target)
    if target_idx == -1:
        return False
    region_end = end_idx + len(SLOW_UPDATE_END)
    return start_idx <= target_idx < region_end


def _strip_slow_update_markers(text: str) -> str:
    """Remove SLOW_UPDATE markers so they don't get duplicated."""
    return text.replace(SLOW_UPDATE_START, "").replace(SLOW_UPDATE_END, "")


# ── Single edit application ────────────────────────────────────────────────


def apply_edit(skill: str, edit: Edit | dict) -> str:
    """Apply a single edit to a skill document.

    Edits targeting the protected slow-update region are silently skipped.
    """
    updated, _ = _apply_edit_with_report(skill, edit)
    return updated


def _apply_edit_with_report(skill: str, edit: Edit | dict) -> tuple[str, dict]:
    """Apply edit and return (updated_skill, report_dict)."""
    op, content, target = _edit_fields(edit)
    report: dict[str, Any] = {
        "op": op,
        "target": target[:200],
        "content_preview": content[:200],
        "status": "unknown",
    }

    # Protected region guard
    if target and _is_in_slow_update_region(skill, target):
        report["status"] = "skipped_protected_slow_update_region"
        return skill, report

    # ── append ──────────────────────────────────────────────────────────
    if op == "append":
        content = _strip_slow_update_markers(content)
        su_start = skill.find(SLOW_UPDATE_START)
        if su_start != -1:
            before = skill[:su_start].rstrip()
            after = skill[su_start:]
            report["status"] = "applied_append_before_slow_update"
            return before + "\n\n" + content + "\n\n" + after, report
        report["status"] = "applied_append"
        return skill.rstrip() + "\n\n" + content + "\n", report

    # ── insert_after ────────────────────────────────────────────────────
    if op == "insert_after":
        if not target or target not in skill:
            # Fallback: insert before slow-update or append at end
            su_start = skill.find(SLOW_UPDATE_START)
            if su_start != -1:
                before = skill[:su_start].rstrip()
                after = skill[su_start:]
                report["status"] = "applied_insert_after_fallback_before_slow_update"
                return before + "\n\n" + content + "\n\n" + after, report
            report["status"] = "applied_insert_after_fallback_append"
            return skill.rstrip() + "\n\n" + content + "\n", report
        idx = skill.index(target) + len(target)
        newline = skill.find("\n", idx)
        insert_at = newline + 1 if newline != -1 else len(skill)
        report["status"] = "applied_insert_after"
        return skill[:insert_at] + "\n" + content + "\n" + skill[insert_at:], report

    # ── replace ────────────────────────────────────────────────────────
    if op == "replace":
        if not target:
            report["status"] = "skipped_replace_missing_target"
            return skill, report
        if target not in skill:
            report["status"] = "skipped_replace_target_not_found"
            return skill, report
        report["status"] = "applied_replace"
        return skill.replace(target, content, 1), report

    # ── delete ──────────────────────────────────────────────────────────
    if op == "delete":
        if not target:
            report["status"] = "skipped_delete_missing_target"
            return skill, report
        if target not in skill:
            report["status"] = "skipped_delete_target_not_found"
            return skill, report
        report["status"] = "applied_delete"
        return skill.replace(target, "", 1), report

    report["status"] = "skipped_unknown_op"
    return skill, report


# ── Patch (multi-edit) application ────────────────────────────────────────


def apply_patch(skill: str, patch: Patch | dict) -> str:
    """Apply all edits in a patch sequentially."""
    updated, _ = apply_patch_with_report(skill, patch)
    return updated


def apply_patch_with_report(
    skill: str,
    patch: Patch | dict,
) -> tuple[str, list[dict]]:
    """Apply a patch and return per-edit report dicts for observability."""
    edits = patch.edits if isinstance(patch, Patch) else patch.get("edits", [])
    reports: list[dict] = []
    for idx, edit in enumerate(edits, 1):
        try:
            skill, report = _apply_edit_with_report(skill, edit)
            report["index"] = idx
        except Exception as exc:
            report = {
                "index": idx,
                "op": "",
                "target": "",
                "content_preview": "",
                "status": "error",
                "error": str(exc),
            }
        reports.append(report)
    return skill, reports


# ── Helpers ────────────────────────────────────────────────────────────────


def _edit_fields(edit: Edit | dict) -> tuple[str, str, str]:
    """Extract (op, content, target) from Edit or dict."""
    if isinstance(edit, Edit):
        op = edit.op
        content = _strip_slow_update_markers(edit.content.strip())
        target = edit.target
    else:
        op = edit.get("op", "")
        content = _strip_slow_update_markers(edit.get("content", "").strip())
        target = edit.get("target", "")
    return op, content, target


def validate_skill_document(text: str) -> tuple[bool, str]:
    """Check if *text* looks like a valid skill document.

    Returns (is_valid, reason).
    """
    if not text or not text.strip():
        return False, "Document is empty after applying edits"
    # Check for obvious corruption markers
    if text.count("#") < 1:
        return False, "Document has no Markdown headings"
    # Check for excessive null/control characters
    control_chars = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text))
    if control_chars > 5:
        return False, f"Document contains {control_chars} control characters (likely corrupted)"
    return True, "OK"
