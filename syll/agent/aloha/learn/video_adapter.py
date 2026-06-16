"""Video adapter: brings external tutorial videos into the Aloha pipeline.

Two-phase analysis:
  Phase 1 — Quick Scan:  sample every N seconds, LLM identifies key transition frames
  Phase 2 — Detailed Analysis: for each keyframe ± context window,
            reuse TraceGenerator to produce structured step descriptions

Reuses:
  - VideoScreenshotExtractor._get_frame_at() for frame extraction
  - TraceGenerator.generate_trace() for step description generation
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
from loguru import logger


# ── Frame extraction helpers ──────────────────────────────────────────────

def _get_video_info(video_path: str | Path) -> dict[str, Any]:
    """Return fps, total_frames, duration_s, width, height."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "total_frames": total, "duration_s": total / fps,
            "width": w, "height": h}


def _extract_frame(video_path: str | Path, timestamp_s: float) -> str | None:
    """Extract a single frame at the given timestamp, return base64 JPEG."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        # Resize to reasonable size for LLM vision
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    finally:
        cap.release()


def _extract_frame_file(video_path: str | Path, timestamp_s: float,
                        output_dir: Path, label: str = "") -> Path | None:
    """Extract a frame to a JPG file, return path or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        fname = label or f"{timestamp_s:.2f}s"
        out_path = output_dir / f"{fname}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return out_path
    finally:
        cap.release()


def _save_frames_batch(video_path: str | Path, timestamps: list[float],
                       output_dir: Path) -> list[Path]:
    """Extract multiple frames to files, return list of saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ts in timestamps:
        p = _extract_frame_file(video_path, ts, output_dir, label=f"{ts:.3f}s")
        if p:
            saved.append(p)
    return saved


# ── Video download ────────────────────────────────────────────────────────

def download_video(url: str, output_dir: str | Path | None = None,
                   cookie_file: str | None = None) -> Path:
    """Download video via yt-dlp. Returns path to the downloaded MP4.

    Args:
        url: Video URL (bilibili, youtube, douyin, etc.)
        output_dir: Target directory. Defaults to ~/.syll/video_cache/
        cookie_file: Optional Netscape cookie file for authenticated sites.

    Raises:
        RuntimeError: If yt-dlp is not installed or download fails.
    """
    import shutil
    if not shutil.which("yt-dlp"):
        raise RuntimeError(
            "yt-dlp is required but not found. "
            "Install with: pip install yt-dlp"
        )

    output_dir = Path(output_dir) if output_dir else Path.home() / ".syll" / "video_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "-o", str(output_dir / "%(id)s.%(ext)s"),
    ]
    if cookie_file:
        cmd.extend(["--cookies", cookie_file])
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

    # Find the downloaded file
    candidates = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("yt-dlp completed but no MP4 found")
    return candidates[0]


# ── Two-phase analyzer ────────────────────────────────────────────────────

class VideoAnalyzer:
    """Two-phase video analysis that feeds into the Aloha trace pipeline."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        api_base: str | None = None,
        scan_interval: float = 5.0,
        context_window: float = 2.0,
    ):
        """
        Args:
            model: LiteLLM model identifier for vision calls.
            api_key: API key for the model.
            api_base: Custom API base URL.
            scan_interval: Seconds between frames in Phase 1 quick scan.
            context_window: Seconds around each keyframe for Phase 2 detail.
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.scan_interval = scan_interval
        self.context_window = context_window

    # ── LLM helper ────────────────────────────────────────────────────

    async def _call_vision(self, prompt: str, images: list[str],
                           temperature: float = 0.2) -> str:
        """Call LiteLLM with text + images."""
        import litellm

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img_b64 in images:
            content.append({"type": "image_url", "image_url": {"url": img_b64}})

        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=2000,
            **kwargs,
        )
        return response.choices[0].message.content

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Extract first valid JSON from LLM output."""
        if not text:
            return None
        # Try code blocks first
        for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.S):
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
        # Try raw braces
        for m in re.finditer(r"\{.*\}", text, flags=re.S):
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
        for m in re.finditer(r"\[.*\]", text, flags=re.S):
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
        return None

    # ── Phase 1: Quick Scan ───────────────────────────────────────────

    async def quick_scan(self, video_path: str | Path,
                         task_description: str = "") -> list[float]:
        """Phase 1: Sample frames at intervals → LLM identifies key transition points.

        Returns:
            List of timestamps (seconds) where key transitions occur.
        """
        video_path = Path(video_path)
        info = _get_video_info(video_path)
        duration = info["duration_s"]
        logger.info(f"[VideoAnalyzer] Phase 1: scanning {duration:.1f}s video "
                     f"(interval={self.scan_interval}s)")

        # Generate scan timestamps
        scan_times = []
        t = self.scan_interval
        while t < duration:
            scan_times.append(t)
            t += self.scan_interval

        if not scan_times:
            logger.warning("[VideoAnalyzer] Video too short for scanning")
            return []

        # Extract frames as base64
        frames_b64: list[tuple[float, str]] = []
        for ts in scan_times:
            b64 = _extract_frame(video_path, ts)
            if b64:
                frames_b64.append((ts, b64))

        if not frames_b64:
            return []

        # Send in batches of ~10 frames per LLM call
        batch_size = 10
        all_keyframes: list[float] = []

        for i in range(0, len(frames_b64), batch_size):
            batch = frames_b64[i:i + batch_size]
            prompt = self._build_scan_prompt(
                len(batch), task_description, info["duration_s"]
            )
            images = [b64 for _, b64 in batch]
            timestamps = [ts for ts, _ in batch]

            try:
                raw = await self._call_vision(prompt, images, temperature=0.1)
                parsed = self._extract_json(raw)

                if isinstance(parsed, list):
                    # Direct list of indices
                    for idx in parsed:
                        if isinstance(idx, int) and 0 <= idx < len(timestamps):
                            all_keyframes.append(timestamps[idx])
                elif isinstance(parsed, dict):
                    # {"keyframes": [0, 3, 7]}
                    indices = parsed.get("keyframes") or parsed.get("indices") or []
                    for idx in indices:
                        if isinstance(idx, int) and 0 <= idx < len(timestamps):
                            all_keyframes.append(timestamps[idx])
                        elif isinstance(idx, (int, float)):
                            all_keyframes.append(float(idx))
            except Exception as e:
                logger.warning(f"[VideoAnalyzer] Phase 1 batch failed: {e}")
                continue

        # Deduplicate and sort
        all_keyframes = sorted(set(all_keyframes))
        logger.info(f"[VideoAnalyzer] Phase 1 done: {len(all_keyframes)} keyframes "
                     f"from {len(frames_b64)} sampled frames")
        return all_keyframes

    def _build_scan_prompt(self, frame_count: int, task: str,
                           total_duration: float) -> str:
        return f"""Analyze these {frame_count} screenshots sampled from a tutorial video{" about: " + task if task else ""}.
Total video duration: {total_duration:.1f}s.

Each image corresponds to a specific timestamp (shown in order: frame 0, 1, 2, ...).

Identify frames where a **significant UI state transition** occurs, such as:
- A menu, dialog, or panel opens or closes
- A new window or tab appears
- Tool selection changes in the toolbar
- Content area updates visibly (new content, file opened, etc.)
- A button is clearly being clicked (hover/pressed state)

Return ONLY a JSON array of frame indices that are key transition points.
Example: [0, 3, 7, 12]

If no clear transitions, return an empty array: []
Do NOT include frames that look the same as the previous one."""

    # ── Phase 2: Detailed Analysis ────────────────────────────────────

    async def detailed_analysis(
        self,
        video_path: str | Path,
        keyframe_timestamps: list[float],
        task_description: str = "",
        screenshots_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Phase 2: For each keyframe, analyze ±context_window frames in detail.

        Uses Aloha's TraceGenerator prompt structure for consistent output.

        Args:
            video_path: Path to the video file.
            keyframe_timestamps: Output from quick_scan().
            task_description: What the video teaches.
            screenshots_dir: Where to save frame images for the skill.

        Returns:
            List of step dicts: {step_idx, timestamp, action, observation,
                                 think, expectation, screenshot_path}
        """
        video_path = Path(video_path)
        info = _get_video_info(video_path)
        if screenshots_dir is None:
            screenshots_dir = video_path.parent / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[VideoAnalyzer] Phase 2: analyzing {len(keyframe_timestamps)} "
                     f"keyframes with ±{self.context_window}s context")

        # Load TraceGenerator's default prompt for consistent formatting
        prompt_path = Path(__file__).parent / "default_prompt.json"
        default_prompt = {}
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                default_prompt = json.load(f)

        steps: list[dict[str, Any]] = []
        step_idx = 1
        recent_steps: list[dict] = []

        for kf_ts in keyframe_timestamps:
            # Extract context window frames: keyframe ± context_window
            context_times = [
                max(0, kf_ts - self.context_window),
                kf_ts,
                min(info["duration_s"], kf_ts + self.context_window),
            ]
            context_times = sorted(set(context_times))

            # Extract frames to files + base64
            frame_b64_list: list[str] = []
            frame_paths: list[Path | None] = []
            for ct in context_times:
                b64 = _extract_frame(video_path, ct)
                frame_b64_list.append(b64) if b64 else frame_b64_list.append("")
                fp = _extract_frame_file(video_path, ct, screenshots_dir,
                                         label=f"step{step_idx:03d}_{ct:.2f}s")
                frame_paths.append(fp)

            if not any(frame_b64_list):
                continue

            # Build detail prompt using Aloha's style
            base_prompt = default_prompt.get("Base Prompt", "")
            prompt = self._build_detail_prompt(
                step_idx, kf_ts, task_description, recent_steps, base_prompt
            )

            # Filter out empty images
            valid_images = [img for img in frame_b64_list if img]

            try:
                raw = await self._call_vision(prompt, valid_images, temperature=0.15)
                data = self._extract_json(raw)

                if isinstance(data, dict):
                    step = {
                        "step_idx": step_idx,
                        "timestamp": kf_ts,
                        "action": data.get("Action") or data.get("action") or "",
                        "observation": data.get("Observation") or data.get("observation") or "",
                        "think": data.get("Think") or data.get("think") or "",
                        "expectation": data.get("Expectation") or data.get("expectation") or "",
                        "screenshot_path": str(frame_paths[1]) if len(frame_paths) > 1 and frame_paths[1] else None,
                        "context_frames": [str(p) for p in frame_paths if p],
                    }
                    # Sanitize: strip coordinate leaks
                    step = self._sanitize_step(step)
                    steps.append(step)
                    recent_steps.append({
                        "step_idx": step_idx,
                        "Observation": step["observation"],
                        "Action": step["action"],
                    })
                    # Keep only last 3 recent steps for context
                    if len(recent_steps) > 3:
                        recent_steps = recent_steps[-3:]
                    step_idx += 1

                    logger.info(f"[VideoAnalyzer] Phase 2 step {step_idx}: "
                                 f"{step['action'][:60]}")
            except Exception as e:
                logger.warning(f"[VideoAnalyzer] Phase 2 failed at {kf_ts:.1f}s: {e}")
                continue

        logger.info(f"[VideoAnalyzer] Phase 2 done: {len(steps)} steps generated")
        return steps

    def _build_detail_prompt(self, step_idx: int, timestamp: float,
                             task: str, recent: list[dict],
                             base_prompt: str) -> str:
        recent_json = json.dumps(recent, ensure_ascii=False, indent=2) if recent else "[]"
        return f"""Analyze this desktop tutorial screenshot.
{"Overall task: " + task if task else ""}

Recent Steps (most recent first, up to 3):
{recent_json}

Current Step Index: {step_idx}
Timestamp: {timestamp:.1f}s

You are seeing 2-3 frames around this key moment (before, during, after the action).

Respond with a JSON object describing what is happening:
{{
  "Observation": "What you see on screen (describe UI elements, menus, buttons, content)",
  "Think": "What the user is trying to do at this step",
  "Action": "The specific action being performed (click X, type Y, select Z)",
  "Expectation": "What should change after this action"
}}

Important rules:
- Describe actions in terms of UI elements ("Click the 'File' menu", "Select 'Export' option")
- Do NOT include pixel coordinates
- Be specific about menu paths, button names, and input values
- If this is a series of related actions, describe them as one logical step"""

    @staticmethod
    def _sanitize_step(step: dict) -> dict:
        """Remove coordinate leaks from step descriptions."""
        coord_pattern = r"\bcoordinates?\b.*?\[[^\]]*\]|\(?\d+\s*,\s*\d+\)?"
        for key in ("observation", "think", "action", "expectation"):
            val = step.get(key, "")
            if isinstance(val, str):
                val = re.sub(coord_pattern, "", val)
                val = re.sub(r"\s{2,}", " ", val).strip()
                step[key] = val
        return step

    # ── Full pipeline ─────────────────────────────────────────────────

    async def analyze_video(
        self,
        video_path: str | Path,
        task_description: str = "",
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run the full two-phase analysis pipeline.

        Args:
            video_path: Path to the MP4 file.
            task_description: What the video teaches.
            output_dir: Where to save results. Defaults to video_dir/analysis/

        Returns:
            {"steps": [...], "video_info": {...}, "screenshots_dir": str}
        """
        video_path = Path(video_path)
        info = _get_video_info(video_path)

        output_dir = Path(output_dir) if output_dir else video_path.parent / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        screenshots_dir = output_dir / "screenshots"

        logger.info(f"[VideoAnalyzer] Starting analysis: {video_path.name} "
                     f"({info['duration_s']:.1f}s, {info['width']}x{info['height']})")

        # Phase 1: Quick scan
        keyframes = await self.quick_scan(video_path, task_description)

        if not keyframes:
            logger.warning("[VideoAnalyzer] No keyframes detected, "
                           "falling back to uniform sampling")
            # Fallback: sample every 10 seconds
            t = 10.0
            while t < info["duration_s"]:
                keyframes.append(t)
                t += 10.0

        # Phase 2: Detailed analysis
        steps = await self.detailed_analysis(
            video_path, keyframes, task_description, screenshots_dir
        )

        result = {
            "steps": steps,
            "video_info": info,
            "keyframe_count": len(keyframes),
            "screenshots_dir": str(screenshots_dir),
            "task_description": task_description,
        }

        # Save analysis result
        result_path = output_dir / "analysis.json"
        # Remove non-serializable screenshot_path references for JSON
        json_safe_steps = []
        for s in steps:
            js = {k: v for k, v in s.items() if k != "context_frames"}
            json_safe_steps.append(js)
        json_safe = {**result, "steps": json_safe_steps}
        result_path.write_text(json.dumps(json_safe, indent=2, ensure_ascii=False),
                               encoding="utf-8")

        logger.info(f"[VideoAnalyzer] Analysis saved to {result_path}")
        return result


# ── Skill generation ──────────────────────────────────────────────────────

def generate_skill_md(
    skill_name: str,
    steps: list[dict[str, Any]],
    task_description: str = "",
    video_url: str = "",
    requires_bins: list[str] | None = None,
) -> str:
    """Generate SKILL.md content from analyzed steps.

    Args:
        skill_name: Skill directory name.
        steps: Output from VideoAnalyzer.analyze_video().
        task_description: What this skill teaches.
        video_url: Source video URL for attribution.
        requires_bins: Required CLI tools (e.g. ["premiere"]).

    Returns:
        SKILL.md content as string.
    """
    import datetime

    requires_yaml = ""
    if requires_bins:
        bins_str = ", ".join(f'"{b}"' for b in requires_bins)
        requires_yaml = f"\nrequires:\n  bins: [{bins_str}]"

    lines = [
        "---",
        f"description: \"{task_description or skill_name}\"",
        f"always: false{requires_yaml}",
        f"source: video-learn",
        f"video_url: \"{video_url}\"",
        f"created: \"{datetime.date.today().isoformat()}\"",
        "---",
        "",
        f"# {task_description or skill_name}",
        "",
    ]

    if video_url:
        lines.append(f"> 学习来源: {video_url}")
        lines.append("")

    lines.append("## 操作步骤")
    lines.append("")

    for step in steps:
        idx = step.get("step_idx", "?")
        action = step.get("action", "")
        observation = step.get("observation", "")
        expectation = step.get("expectation", "")
        screenshot = step.get("screenshot_path")

        lines.append(f"### Step {idx}: {action}")
        lines.append("")

        if observation:
            lines.append(f"- **界面状态**: {observation}")
        if action:
            lines.append(f"- **操作**: {action}")
        if expectation:
            lines.append(f"- **预期结果**: {expectation}")
        if screenshot:
            lines.append(f"- **截图**: `{{{{workspace}}}}/skills/{skill_name}/{Path(screenshot).name}`")

        lines.append("")

    lines.append("## 注意事项")
    lines.append("")
    lines.append("- 此技能通过视频示教自动学习生成，首次执行时建议用户监督")
    lines.append("- 具体菜单路径可能因软件版本不同而有差异")
    lines.append("")

    return "\n".join(lines)
