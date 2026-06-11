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
        "Learn a desktop operation skill from online tutorial videos. "
        "Searches for videos, downloads, analyzes key frames, and generates a SKILL.md."
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
        **kwargs: Any,
    ) -> str:
        """Execute the video learning pipeline."""
        try:
            # Determine skill name
            if not skill_name:
                skill_name = task.lower().replace(" ", "-")[:40]

            # Step 1: Find video URL if not provided
            if not video_url:
                video_url = await self._search_video(task)
                if not video_url:
                    return json.dumps({"error": f"No tutorial video found for: {task}"})

            # Step 2: Download video
            logger.info(f"[video_learn] Downloading: {video_url}")
            try:
                from syll.agent.aloha.learn.video_adapter import download_video
                video_path = download_video(video_url)
            except RuntimeError as e:
                return json.dumps({"error": str(e), "hint": "yt-dlp may need to be installed"})

            if download_only:
                return json.dumps({
                    "status": "downloaded",
                    "video_path": str(video_path),
                    "message": "Video downloaded. Run video_learn again with video_url to analyze."
                })

            # Step 3: Two-phase analysis
            logger.info(f"[video_learn] Starting analysis: {video_path}")
            from syll.agent.aloha.learn.video_adapter import VideoAnalyzer

            analyzer = VideoAnalyzer(
                model=self.model or "gpt-4o",
                api_key=self.api_key or None,
                api_base=self.api_base or None,
                scan_interval=scan_interval,
            )
            result = await analyzer.analyze_video(
                video_path=video_path,
                task_description=task,
            )

            steps = result.get("steps", [])
            if not steps:
                return json.dumps({
                    "error": "Analysis produced no actionable steps",
                    "video_path": str(video_path),
                    "video_info": result.get("video_info"),
                })

            # Step 4: Generate SKILL.md
            from syll.agent.aloha.learn.video_adapter import generate_skill_md

            skill_dir = self.workspace / "skills" / skill_name
            skill_dir.mkdir(parents=True, exist_ok=True)

            # Copy screenshots into skill directory
            screenshots_src = Path(result.get("screenshots_dir", ""))
            if screenshots_src.exists():
                skill_screenshots = skill_dir / "screenshots"
                skill_screenshots.mkdir(exist_ok=True)
                for img in screenshots_src.glob("*.jpg"):
                    shutil.copy2(img, skill_screenshots / img.name)

            skill_content = generate_skill_md(
                skill_name=skill_name,
                steps=steps,
                task_description=task,
                video_url=video_url,
            )

            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(skill_content, encoding="utf-8")

            # Step 5: Write memory note
            try:
                from datetime import datetime
                memory_dir = self.workspace / "memory"
                memory_dir.mkdir(parents=True, exist_ok=True)
                today = datetime.now().strftime("%Y-%m-%d")
                memory_file = memory_dir / f"{today}.md"

                entry = (
                    f"\n- [{datetime.now().strftime('%H:%M')}] "
                    f"自动学习技能: **{skill_name}** (来源: {video_url})\n"
                    f"  共 {len(steps)} 个操作步骤\n"
                )
                if memory_file.exists():
                    memory_file.write_text(
                        memory_file.read_text(encoding="utf-8") + entry,
                        encoding="utf-8"
                    )
                else:
                    memory_file.write_text(f"# {today}\n{entry}", encoding="utf-8")
            except Exception as e:
                logger.debug(f"Failed to write memory: {e}")

            return json.dumps({
                "status": "learned",
                "skill_name": skill_name,
                "skill_path": str(skill_file),
                "step_count": len(steps),
                "steps": [
                    {"idx": s.get("step_idx"), "action": s.get("action", "")}
                    for s in steps
                ],
                "video_url": video_url,
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"[video_learn] Pipeline failed: {e}")
            return json.dumps({"error": str(e)})

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
