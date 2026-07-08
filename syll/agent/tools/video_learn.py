"""Video-to-Skill learning tool for Syll agent.

Orchestrates the full pipeline:
  web_search → download → two-phase analyze → generate SKILL.md → memory

Reuses Aloha's VideoScreenshotExtractor + TraceGenerator via VideoAnalyzer.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from loguru import logger

from syll.agent.tools.base import Tool, ToolResult


class VideoLearnTool(Tool):
    """Learn a desktop skill from online tutorial videos.

    Pipeline: search → download → analyze (two-phase) → write SKILL.md → memory.
    """

    name = "video_learn"
    description = (
        "Learn a desktop operation skill from an online tutorial video. By default "
        "it WATCHES the video in a browser (samples the screen while it plays) and "
        "summarizes the steps with the vision model — no download needed. Generates "
        "a SKILL.md you can then follow with gui_action_planned."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What to learn, e.g. 'Premiere Pro add subtitles'"
            },
            "video_url": {
                "type": "string",
                "description": "Direct video URL (skip search if provided)"
            },
            "skill_name": {
                "type": "string",
                "description": "Name for the generated skill (directory name)"
            },
            "scan_interval": {
                "type": "number",
                "description": "Seconds between frames in quick scan (default: 5.0)"
            },
            "download_only": {
                "type": "boolean",
                "description": "Only download, skip analysis (default: false)"
            },
            "method": {
                "type": "string",
                "enum": ["auto", "browser", "download"],
                "description": (
                    "auto=watch in browser, fall back to download; "
                    "browser=screen-watch only (no yt-dlp); download=yt-dlp only."
                ),
            },
            "watch_frames": {
                "type": "integer",
                "description": "browser path: number of screenshots to sample (default 12)",
            },
            "watch_interval": {
                "type": "number",
                "description": "browser path: seconds between screenshots (default 8)",
            },
        },
        "required": ["task"]
    }

    def __init__(self, model: str = "", api_key: str = "",
                 api_base: str = "", workspace: Path | None = None,
                 brave_api_key: str = ""):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.workspace = workspace or Path.home() / ".syll" / "workspace"
        self.brave_api_key = brave_api_key

    async def execute(
        self,
        task: str,
        video_url: str = "",
        skill_name: str = "",
        scan_interval: float = 5.0,
        download_only: bool = False,
        method: str = "auto",
        watch_frames: int = 12,
        watch_interval: float = 8.0,
        **kwargs: Any,
    ) -> str:
        """Learn a skill from a tutorial video.

        method: "browser" (open + screen-watch; no yt-dlp), "download" (yt-dlp +
        offline extraction), or "auto" (browser first, download fallback).
        """
        try:
            if not skill_name:
                skill_name = task.lower().replace(" ", "-")[:40]

            if not video_url:
                video_url = await self._search_video(task)
                if not video_url and method != "download":
                    return json.dumps({"error": f"No tutorial video found for: {task}"})

            if download_only:
                return await self._download_only(video_url)

            order = {"browser": ["browser"], "download": ["download"]}.get(
                method, ["browser", "download"]
            )
            last_err = ""
            for m in order:
                try:
                    if m == "browser":
                        steps, shots = await self._run_browser(
                            video_url, task, skill_name, watch_frames, watch_interval
                        )
                    else:
                        steps, shots = await self._run_download(
                            video_url, task, scan_interval
                        )
                    if steps:
                        return self._finalize_skill(
                            skill_name, steps, task, video_url, shots, source=m
                        )
                    last_err = f"{m}: produced no actionable steps"
                except Exception as e:
                    last_err = f"{m}: {e}"
                    logger.warning(f"[video_learn] {last_err}")
            return json.dumps({"error": last_err or "no usable path",
                               "video_url": video_url})
        except Exception as e:
            logger.error(f"[video_learn] Pipeline failed: {e}")
            return json.dumps({"error": str(e)})

    # ── paths ──────────────────────────────────────────────────────────

    def _analyzer(self, scan_interval: float = 5.0):
        from syll.agent.aloha.learn.video_adapter import VideoAnalyzer
        return VideoAnalyzer(
            model=self.model or "gpt-4o",
            api_key=self.api_key or None,
            api_base=self.api_base or None,
            scan_interval=scan_interval,
        )

    async def _run_browser(self, video_url, task, skill_name, n_frames, interval):
        """Primary path: open in a browser, screen-sample while it plays, analyze."""
        from syll.agent.aloha.learn.browser_watch import BrowserTutorialWatcher
        logger.info(f"[video_learn] Watching in browser: {video_url}")
        watcher = BrowserTutorialWatcher(self._analyzer(), self.workspace)
        res = await watcher.watch_and_extract(
            video_url, task, skill_name, n_frames=n_frames, interval_s=interval
        )
        if res.get("error"):
            raise RuntimeError(res["error"])
        return res.get("steps", []), res.get("screenshots_dir")

    async def _run_download(self, video_url, task, scan_interval):
        """Fallback path: yt-dlp download + offline two-phase analysis."""
        from syll.agent.aloha.learn.video_adapter import download_video
        logger.info(f"[video_learn] Downloading: {video_url}")
        video_path = download_video(video_url)  # raises if yt-dlp missing/blocked
        result = await self._analyzer(scan_interval).analyze_video(
            video_path=video_path, task_description=task
        )
        return result.get("steps", []), result.get("screenshots_dir")

    async def _download_only(self, video_url) -> str:
        from syll.agent.aloha.learn.video_adapter import download_video
        try:
            video_path = download_video(video_url)
        except RuntimeError as e:
            return json.dumps({"error": str(e), "hint": "yt-dlp may need to be installed"})
        return json.dumps({
            "status": "downloaded", "video_path": str(video_path),
            "message": "Video downloaded. Re-run with method='download' to analyze.",
        })

    def _finalize_skill(self, skill_name, steps, task, video_url, screenshots_dir,
                        source: str = "") -> str:
        """Shared tail: copy external screenshots in, write SKILL.md + memory note."""
        from syll.agent.aloha.learn.video_adapter import generate_skill_md

        skill_dir = self.workspace / "skills" / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        dst = skill_dir / "screenshots"
        src = Path(screenshots_dir) if screenshots_dir else None
        if src and src.exists() and src.resolve() != dst.resolve():
            dst.mkdir(exist_ok=True)
            for img in list(src.glob("*.jpg")) + list(src.glob("*.jpeg")):
                try:
                    shutil.copy2(img, dst / img.name)
                except Exception:
                    pass

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            generate_skill_md(skill_name=skill_name, steps=steps,
                              task_description=task, video_url=video_url),
            encoding="utf-8",
        )
        self._write_memory_note(skill_name, video_url, len(steps))
        return json.dumps({
            "status": "learned", "skill_name": skill_name,
            "skill_path": str(skill_file), "step_count": len(steps),
            "method": source,
            "steps": [{"idx": s.get("step_idx"), "action": s.get("action", "")}
                      for s in steps],
            "video_url": video_url,
        }, ensure_ascii=False, indent=2)

    def _write_memory_note(self, skill_name, video_url, n_steps) -> None:
        try:
            from datetime import datetime
            memory_dir = self.workspace / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now().strftime("%Y-%m-%d")
            memory_file = memory_dir / f"{today}.md"
            entry = (
                f"\n- [{datetime.now().strftime('%H:%M')}] "
                f"自动学习技能: **{skill_name}** (来源: {video_url or 'browser-watch'})\n"
                f"  共 {n_steps} 个操作步骤\n"
            )
            if memory_file.exists():
                memory_file.write_text(
                    memory_file.read_text(encoding="utf-8") + entry, encoding="utf-8"
                )
            else:
                memory_file.write_text(f"# {today}\n{entry}", encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed to write memory: {e}")

    async def _search_video(self, task: str) -> str:
        """Search for a tutorial video using Brave Search."""
        if not self.brave_api_key:
            logger.warning("[video_learn] No Brave API key, cannot search")
            return ""

        import httpx
        query = f"{task} 教程 bilibili"
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": 5},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self.brave_api_key,
                    },
                    timeout=10.0,
                )
                r.raise_for_status()

            results = r.json().get("web", {}).get("results", [])
            # Prefer bilibili / youtube links
            video_sites = ("bilibili.com", "youtube.com", "youtu.be")
            for item in results:
                url = item.get("url", "")
                if any(site in url for site in video_sites):
                    return url

            # Fallback to first result
            if results:
                return results[0].get("url", "")

        except Exception as e:
            logger.warning(f"[video_learn] Search failed: {e}")

        return ""
