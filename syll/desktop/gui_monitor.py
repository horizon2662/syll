"""GUI execution monitor — ghost-based overlay showing real-time progress.

Reuses the Syll ghost mascot SVG to display GUI automation progress in a
small always-on-top, click-through window.  Won't interfere with pyautogui.

Communication: UITarsTool writes progress to ``~/.syll/.gui_monitor_state.json``;
this overlay polls it every 300 ms.

Config toggle: set ``tools.gui.monitor: true/false`` in ``~/.syll/config.json``.
Default is ``true``.

Run standalone for testing::

    python -m syll.desktop.gui_monitor

Run as background serve mode (auto-show on state change)::

    python -m syll.desktop.gui_monitor --serve
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
_SVG_DIR = _HERE.parent / "web" / "static" / "ghost"
_STATE_FILE = Path.home() / ".syll" / ".gui_monitor_state.json"
_CONFIG_FILE = Path.home() / ".syll" / "config.json"

# ── Layout constants ──────────────────────────────────────────────────────

WINDOW_W = 380
WINDOW_H = 172
GHOST_SIZE = 80
GHOST_X = 14
PANEL_X = GHOST_X + GHOST_SIZE + 14
PANEL_W = WINDOW_W - PANEL_X - 14
POLL_MS = 300
RADIUS = 14
MAX_IDLE_TICKS = 200  # auto-exit after ~60 s idle in serve mode (200×300ms)


# ═══════════════════════════════════════════════════════════════════════════
# State file API  (called by UITarsTool — no Qt dependency)
# ═══════════════════════════════════════════════════════════════════════════

def write_gui_state(
    *,
    status: str,
    instruction: str = "",
    step: int = 0,
    max_steps: int = 0,
    action: str = "",
    thought: str = "",
    error: str = "",
) -> None:
    """Write current GUI execution state to the monitor state file.

    Thread-safe — safe to call from any thread (including asyncio).
    No Qt import needed.
    """
    blob = {
        "status":      status,
        "instruction": instruction,
        "step":        step,
        "max_steps":   max_steps,
        "action":      action,
        "thought":     thought,
        "error":       error,
        "ts":          datetime.now().isoformat(),
    }
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def clear_gui_state() -> None:
    """Remove the state file (called when GUI execution ends)."""
    try:
        _STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _is_enabled() -> bool:
    """Read ``tools.gui.monitor`` from config (default ``True``)."""
    try:
        if _CONFIG_FILE.exists():
            cfg = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            return bool(cfg.get("tools", {}).get("gui", {}).get("monitor", True))
    except Exception:
        pass
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Subprocess launcher  (called by UITarsTool._monitor_launch)
# ═══════════════════════════════════════════════════════════════════════════

_monitor_proc: subprocess.Popen | None = None


def launch_gui_monitor() -> None:
    """Launch the monitor overlay as an independent subprocess.

    Uses ``python -m syll.desktop.gui_monitor --serve`` so Qt runs in its
    own main thread — avoids the "QApplication in non-main thread" crash on
    Windows.

    Respects the ``tools.gui.monitor`` config toggle.
    Idempotent: if the process is already alive, no-ops.
    """
    global _monitor_proc

    if not _is_enabled():
        logger.debug("GUI monitor disabled by config (tools.gui.monitor=false)")
        return

    # Already running?
    if _monitor_proc is not None and _monitor_proc.poll() is None:
        return

    try:
        cmd = [sys.executable, "-m", "syll.desktop.gui_monitor", "--serve"]
        kw: dict = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        _monitor_proc = subprocess.Popen(cmd, **kw)  # type: ignore[arg-type]
        logger.debug(f"GUI monitor launched (pid={_monitor_proc.pid})")
    except Exception as exc:
        logger.warning(f"GUI monitor launch failed: {exc}")


def stop_gui_monitor() -> None:
    """Terminate the monitor subprocess and clean up state."""
    global _monitor_proc
    if _monitor_proc is not None and _monitor_proc.poll() is None:
        try:
            _monitor_proc.terminate()
        except Exception:
            pass
    _monitor_proc = None
    clear_gui_state()


# ═══════════════════════════════════════════════════════════════════════════
# Below: Qt GUI (only imported when running as __main__)
# ═══════════════════════════════════════════════════════════════════════════

def _run_qt_app(serve: bool = False) -> None:
    """Build and run the Qt monitor window.  Only called from __main__."""
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import (
        QColor, QFont, QPainter, QPen, QBrush, QPixmap, QRadialGradient,
    )
    from PyQt6.QtWidgets import QApplication, QWidget

    # ── Ghost state → SVG mapping (mirrors ghost.html defaults) ───────
    _GHOST_SVG = {
        "idle":     "ghost-idle-follow.svg",
        "working":  "ghost-working-thinking.svg",
        "sleeping": "ghost-sleeping.svg",
        "error":    "ghost-gui-help.svg",
    }
    _STATUS_TO_GHOST = {
        "running":   "working",
        "finished":  "idle",
        "error":     "error",
        "cancelled": "idle",
        "idle":      "idle",
    }
    # ── Colours ───────────────────────────────────────────────────────
    _C_BG      = QColor(18, 18, 24, 232)
    _C_GLOW    = QColor(80, 160, 255, 28)
    _C_BORDER  = {
        "running":  QColor(70, 150, 255, 150),
        "finished": QColor(80, 220, 120, 150),
        "error":    QColor(255, 90, 90, 150),
        "idle":     QColor(70, 70, 90, 80),
    }
    _C_TEXT    = QColor(228, 228, 238)
    _C_DIM     = QColor(135, 135, 155)
    _C_ACCENT  = QColor(80, 160, 255)
    _C_WARN    = QColor(255, 180, 60)
    _C_OK      = QColor(80, 220, 120)
    _C_ERR     = QColor(255, 90, 90)
    _C_BAR_BG  = QColor(40, 40, 56)
    _C_BAR_FG  = QColor(80, 160, 255)
    _C_DIVIDER = QColor(255, 255, 255, 18)

    class _MonitorWindow(QWidget):
        """Always-on-top, click-through overlay."""

        def __init__(self):
            super().__init__()
            self._state: dict = {}
            self._ghost_state: str = "idle"
            self._ghost_px: QPixmap = self._render_ghost("idle")
            self._idle_ticks: int = 0
            self._setup_window()
            self._start_poll()

        def _setup_window(self):
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowDoesNotAcceptFocus
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.setFixedSize(WINDOW_W, WINDOW_H)
            self.setWindowTitle("Syll GUI Monitor")
            screen = QApplication.primaryScreen().availableGeometry()
            self.move(screen.width() - WINDOW_W - 20, 20)

        def _render_ghost(self, ghost_state: str) -> QPixmap:
            svg_name = _GHOST_SVG.get(ghost_state, _GHOST_SVG["idle"])
            svg_path = _SVG_DIR / svg_name
            if svg_path.exists():
                try:
                    from PyQt6.QtSvg import QSvgRenderer
                    r = QSvgRenderer(str(svg_path))
                    if r.isValid():
                        px = QPixmap(GHOST_SIZE, GHOST_SIZE)
                        px.fill(QColor(0, 0, 0, 0))
                        p = QPainter(px)
                        p.setRenderHint(QPainter.RenderHint.Antialiasing)
                        r.render(p)
                        p.end()
                        return px
                except ImportError:
                    pass
            return self._fallback_ghost()

        @staticmethod
        def _fallback_ghost() -> QPixmap:
            px = QPixmap(GHOST_SIZE, GHOST_SIZE)
            px.fill(QColor(0, 0, 0, 0))
            p = QPainter(px)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(200, 210, 255, 200))
            p.drawEllipse(8, 4, 64, 48)
            p.drawRoundedRect(8, 32, 64, 38, 6, 6)
            p.setBrush(QColor(35, 35, 55))
            p.drawEllipse(22, 22, 11, 14)
            p.drawEllipse(46, 22, 11, 14)
            p.setBrush(QColor(255, 255, 255, 200))
            p.drawEllipse(26, 24, 5, 6)
            p.drawEllipse(50, 24, 5, 6)
            p.end()
            return px

        def _start_poll(self):
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(POLL_MS)
            self._tick()

        def _tick(self):
            try:
                if not _STATE_FILE.exists():
                    self._idle_ticks += 1
                    # Auto-exit if hidden and idle for too long (serve mode)
                    if serve and self._idle_ticks > MAX_IDLE_TICKS and not self.isVisible():
                        QApplication.quit()
                        return
                    return
                self._idle_ticks = 0
                raw = _STATE_FILE.read_text(encoding="utf-8")
                new = json.loads(raw)
            except Exception:
                return
            if new == self._state:
                return
            self._state = new

            status = new.get("status", "idle")
            gs = _STATUS_TO_GHOST.get(status, "idle")
            if gs != self._ghost_state:
                self._ghost_state = gs
                self._ghost_px = self._render_ghost(gs)

            if status == "running" and not self.isVisible():
                self.show()
            if status in ("finished", "error", "cancelled"):
                QTimer.singleShot(4000, self.hide)

            self.update()

        def paintEvent(self, _event):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            w, h = self.width(), self.height()

            status = self._state.get("status", "idle")
            border = _C_BORDER.get(status, _C_BORDER["idle"])

            # Background
            p.setPen(QPen(border, 1.5))
            p.setBrush(QBrush(_C_BG))
            p.drawRoundedRect(1, 1, w - 2, h - 2, RADIUS, RADIUS)

            # Ghost glow
            glow = QRadialGradient(
                GHOST_X + GHOST_SIZE / 2, h / 2, GHOST_SIZE * 0.72,
            )
            glow.setColorAt(0, _C_GLOW)
            glow.setColorAt(1, QColor(0, 0, 0, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(
                GHOST_X - 10,
                int(h / 2 - GHOST_SIZE * 0.72),
                GHOST_SIZE + 20,
                int(GHOST_SIZE * 1.44),
            )

            # Ghost mascot
            gy = int((h - GHOST_SIZE) / 2)
            p.drawPixmap(GHOST_X, gy, self._ghost_px)

            # Vertical divider
            div_x = GHOST_X + GHOST_SIZE + 6
            p.setPen(QPen(_C_DIVIDER, 1))
            p.drawLine(div_x, 16, div_x, h - 16)

            # Right-side progress panel
            x = PANEL_X
            y = 18

            badge_map = {
                "running":  ("⚡ GUI 自动化执行中", _C_ACCENT),
                "finished": ("✓ 执行完成",          _C_OK),
                "error":    ("✗ 执行出错",          _C_ERR),
            }
            badge_text, badge_col = badge_map.get(status, ("○ 待命中", _C_DIM))
            p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            p.setPen(QPen(badge_col))
            p.drawText(x, y + 12, badge_text)
            y += 24

            inst = self._state.get("instruction", "")
            if inst:
                p.setFont(QFont("Segoe UI", 9))
                p.setPen(QPen(_C_TEXT))
                show = inst[:30] + "..." if len(inst) > 33 else inst
                p.drawText(x, y + 11, show)
                y += 20

            step = self._state.get("step", 0)
            max_s = self._state.get("max_steps", 0)
            if max_s > 0:
                bar_h = 5
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(_C_BAR_BG))
                p.drawRoundedRect(x, y, PANEL_W, bar_h, 2, 2)
                fw = int(PANEL_W * min(step / max_s, 1.0))
                if fw > 0:
                    p.setBrush(QBrush(_C_BAR_FG))
                    p.drawRoundedRect(x, y, fw, bar_h, 2, 2)
                y += bar_h + 4
                p.setFont(QFont("Segoe UI", 8))
                p.setPen(QPen(_C_DIM))
                p.drawText(x, y + 10, f"步骤 {step} / {max_s}")
                y += 20

            action = self._state.get("action", "")
            if action:
                p.setFont(QFont("Consolas", 8))
                p.setPen(QPen(_C_WARN))
                show_a = action[:35] + "..." if len(action) > 38 else action
                p.drawText(x, y + 10, f"▸ {show_a}")
                y += 18

            err = self._state.get("error", "")
            if err:
                p.setFont(QFont("Segoe UI", 8))
                p.setPen(QPen(_C_ERR))
                show_e = err[:35] + "..." if len(err) > 38 else err
                p.drawText(x, y + 10, f"⚠ {show_e}")

            if status == "running":
                p.setFont(QFont("Segoe UI", 7))
                p.setPen(QPen(_C_WARN))
                p.drawText(x, h - 12, "⚠ 请勿移动鼠标或操作键盘")

            p.end()

    # ── Build & run ──
    app = QApplication.instance() or QApplication(sys.argv)
    win = _MonitorWindow()

    if serve:
        win.hide()  # start hidden → auto-shows when state file says "running"
    else:
        win.show()
        write_gui_state(
            status="running",
            instruction="打开 Chrome 访问 google.com",
            step=3,
            max_steps=8,
            action="click(start='(960, 540)')",
        )

    sys.exit(app.exec())


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    _serve = "--serve" in sys.argv
    _run_qt_app(serve=_serve)
