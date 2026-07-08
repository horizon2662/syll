"""GUI-native tutorial watching (primary path; no yt-dlp).

Open the tutorial URL in the default browser, start playback, sample the screen
while it plays, and hand the frames to ``VideoAnalyzer.analyze_frames`` for the
vision model to summarize into a procedure. Used by ``VideoLearnTool`` as the
primary "watch a tutorial" path; download (``video_adapter.download_video``) is
only a fallback for the rare actually-downloadable video.

Screen capture uses mss (fast) with a PIL.ImageGrab fallback; playback is started
best-effort (autoplay + a space keypress). All desktop interaction is guarded so
a headless/CI environment degrades to "no frames" rather than raising.
"""
from __future__ import annotations

import asyncio
import base64
import io
import time
import webbrowser
from pathlib import Path
from typing import Any

from loguru import logger


def _grab_screen_jpeg(resize: tuple[int, int] = (1280, 720)) -> tuple[str, bytes] | None:
    """Grab the primary monitor -> (data-uri jpeg, raw bytes), or None on failure."""
    img = None
    try:
        import mss
        from PIL import Image

        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    except Exception:
        try:
            from PIL import ImageGrab

            img = ImageGrab.grab()
        except Exception as e:
            logger.warning(f"[browser_watch] screen grab unavailable: {e}")
            return None
    try:
        img = img.convert("RGB").resize(resize)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        raw = buf.getvalue()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode(), raw
    except Exception as e:
        logger.warning(f"[browser_watch] encode failed: {e}")
        return None


class BrowserTutorialWatcher:
    """Open a tutorial in a browser and sample the screen while it plays."""

    def __init__(self, analyzer: Any, workspace: Path):
        self.analyzer = analyzer            # a VideoAnalyzer (uses .analyze_frames)
        self.workspace = Path(workspace)

    def open_and_play(self, url: str, settle_s: float = 6.0) -> bool:
        opened = False
        try:
            opened = bool(webbrowser.open(url, new=2))
        except Exception as e:
            logger.warning(f"[browser_watch] webbrowser.open failed: {e}")
        time.sleep(settle_s)  # let the page + player load
        try:  # best-effort: most web players start/toggle on space
            import pyautogui

            pyautogui.press("space")
        except Exception:
            pass
        return opened

    def sample_screen(
        self, n_frames: int, interval_s: float, out_dir: Path
    ) -> list[tuple[float, str, Path | None]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        frames: list[tuple[float, str, Path | None]] = []
        t0 = time.monotonic()
        for k in range(n_frames):
            shot = _grab_screen_jpeg()
            if shot is not None:
                b64, raw = shot
                fp: Path | None = out_dir / f"frame_{k:03d}.jpg"
                try:
                    fp.write_bytes(raw)
                except Exception:
                    fp = None
                frames.append((round(time.monotonic() - t0, 2), b64, fp))
            if k < n_frames - 1:
                time.sleep(interval_s)
        return frames

    def _open_and_sample(self, url, n_frames, interval_s, out_dir):
        """Blocking open+play+sample, run in an executor by watch_and_extract."""
        self.open_and_play(url)
        return self.sample_screen(n_frames, interval_s, out_dir)

    async def watch_and_extract(
        self, url: str, task: str, skill_name: str,
        n_frames: int = 12, interval_s: float = 8.0,
    ) -> dict[str, Any]:
        """open -> play -> sample screen -> analyze_frames. Returns
        {"steps": [...], "screenshots_dir": str, "frames": int} or {"error": ...}."""
        shots_dir = self.workspace / "skills" / skill_name / "screenshots"
        loop = asyncio.get_event_loop()
        frames = await loop.run_in_executor(
            None, self._open_and_sample, url, n_frames, interval_s, shots_dir
        )
        if not frames:
            return {"error": "no frames captured (no display / headless?)"}
        steps = await self.analyzer.analyze_frames(frames, task_description=task)
        return {"steps": steps, "screenshots_dir": str(shots_dir), "frames": len(frames)}
