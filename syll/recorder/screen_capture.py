"""Screen recording via mss + video writer (imageio or cv2).

Primary writer: ``imageio`` with ``imageio-ffmpeg`` plugin — writes H.264
MP4 that plays in all browsers.  No compiled C extension required; works
on any Python version.

Fallback writer: ``cv2`` (opencv-python) if available — kept for backward
compatibility on systems where it is already installed.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ── Detect available backends ──────────────────────────────────────────────

_HAS_CV2 = False
_HAS_IMAGEIO = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    pass

try:
    import imageio as _imageio_mod  # noqa: F401 — presence check
    _HAS_IMAGEIO = True
except ImportError:
    pass


def _writer_backend_name() -> str:
    if _HAS_IMAGEIO:
        return "imageio"
    if _HAS_CV2:
        return "cv2"
    return "none"


# ── Screen metadata ───────────────────────────────────────────────────────

@dataclass
class ScreenInfo:
    """Screen dimensions and scaling metadata for the CONFIG line."""

    logical_width: int
    logical_height: int
    physical_width: int
    physical_height: int
    scale_factor: float
    x0: int = 0
    y0: int = 0


# ── Video writer abstraction ───────────────────────────────────────────────

class _VideoWriterBase:
    """Minimal write interface shared by both backends."""

    def write(self, frame: np.ndarray) -> None: ...
    def release(self) -> None: ...


class _ImageioWriter(_VideoWriterBase):
    """imageio-ffmpeg backed writer — writes H.264 MP4 natively."""

    def __init__(self, path: str, fps: float, size: tuple[int, int]) -> None:
        import imageio

        self._writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            output_params=["-pix_fmt", "yuv420p", "-preset", "veryfast"],
        )
        self._size = size

    def write(self, frame: np.ndarray) -> None:
        # imageio expects RGB uint8 array
        self._writer.append_data(frame)

    def release(self) -> None:
        self._writer.close()


class _Cv2Writer(_VideoWriterBase):
    """cv2.VideoWriter — tries H.264 first, falls back to mp4v."""

    _BROWSER_SAFE_FOURCCS = ["avc1", "H264", "X264"]

    def __init__(self, path: str, fps: float, size: tuple[int, int]) -> None:
        for tag in self._BROWSER_SAFE_FOURCCS:
            fourcc = cv2.VideoWriter_fourcc(*tag)
            w = cv2.VideoWriter(path, fourcc, fps, size)
            if w.isOpened():
                self._writer = w
                self._tag = tag
                return
            w.release()

        # Legacy fallback
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        self._tag = "mp4v"
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(
                "cv2.VideoWriter failed to open with any codec"
            )

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


def _create_writer(
    path: str, fps: float, size: tuple[int, int]
) -> _VideoWriterBase:
    """Create the best available video writer.

    Priority: imageio (H.264, browser-safe, pure Python) > cv2.
    """
    if _HAS_IMAGEIO:
        try:
            return _ImageioWriter(path, fps, size)
        except Exception as exc:
            logger.warning("imageio writer failed (%s), trying cv2", exc)

    if _HAS_CV2:
        try:
            return _Cv2Writer(path, fps, size)
        except Exception as exc:
            logger.warning("cv2 writer failed (%s)", exc)

    raise RuntimeError(
        "No video writer available.  Install one of:\n"
        "  pip install imageio[ffmpeg]   (recommended, pure Python)\n"
        "  pip install opencv-python     (if wheels exist for your Python)"
    )


# ── Frame reader (for /frame endpoint) ────────────────────────────────────

def read_frame_at_ms(video_path: str, timestamp_ms: int) -> np.ndarray:
    """Read a single frame at *timestamp_ms* from a video file.

    Returns a BGR uint8 numpy array (compatible with JPEG encoding).
    """
    if _HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(max(timestamp_ms, 0)))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"No frame at {timestamp_ms} ms")
            return frame
        finally:
            cap.release()

    # imageio fallback — read by frame index
    if _HAS_IMAGEIO:
        import imageio

        reader = imageio.get_reader(video_path)
        try:
            fps = reader.get_meta_data().get("fps", 15.0)
            frame_idx = int((timestamp_ms / 1000.0) * fps)
            frame_idx = max(0, min(frame_idx, len(reader) - 1))
            # imageio returns RGB; convert to BGR for JPEG encoding consistency
            rgb = reader.get_data(frame_idx)
            return rgb[:, :, ::-1].copy()
        finally:
            reader.close()

    raise RuntimeError(
        "No video reader available.  Install imageio[ffmpeg] or opencv-python."
    )


def encode_frame_jpeg(frame: np.ndarray, quality: int = 92) -> bytes:
    """Encode a BGR numpy array to JPEG bytes.

    Tries cv2 first (fast), then Pillow fallback.
    """
    if _HAS_CV2:
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if ok:
            return buf.tobytes()

    # Pillow fallback
    from PIL import Image

    # frame is BGR → convert to RGB for Pillow
    rgb = frame[:, :, ::-1] if frame.shape[2] == 3 else frame
    img = Image.fromarray(rgb)
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ── ScreenCapture class ───────────────────────────────────────────────────

class ScreenCapture:
    """Capture the screen to an MP4 video in a background thread.

    The video is written at **logical** resolution so that pixel positions
    in frames correspond 1-to-1 with pynput mouse coordinates.
    """

    def __init__(self, output_path: str, fps: int = 15, monitor_idx: int = 0):
        self.output_path = output_path
        self.fps = fps
        self.monitor_idx = monitor_idx

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()

        self.start_time: float = 0.0
        self.screen_info: ScreenInfo | None = None

    # -- public API -----------------------------------------------------------

    def start(self) -> ScreenInfo:
        """Start recording.  Blocks until the first frame has been grabbed."""
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="screen-capture"
        )
        self._thread.start()
        self._ready.wait(timeout=15)
        if self.screen_info is None:
            raise RuntimeError("Screen capture failed to initialize")
        return self.screen_info

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    # -- internals ------------------------------------------------------------

    def _loop(self) -> None:
        import mss

        with mss.mss() as sct:
            monitors = sct.monitors
            idx = min(self.monitor_idx + 1, len(monitors) - 1)
            mon = monitors[idx]
            logical_w: int = mon["width"]
            logical_h: int = mon["height"]

            # Probe physical resolution from one grab
            sample = sct.grab(mon)
            physical_w, physical_h = sample.width, sample.height
            scale = round(physical_w / logical_w, 2) if logical_w else 1.0

            self.screen_info = ScreenInfo(
                logical_width=logical_w,
                logical_height=logical_h,
                physical_width=physical_w,
                physical_height=physical_h,
                scale_factor=scale,
                x0=mon.get("left", 0),
                y0=mon.get("top", 0),
            )

            writer = _create_writer(
                self.output_path,
                float(self.fps),
                (logical_w, logical_h),
            )
            interval = 1.0 / self.fps

            logger.info(
                "ScreenCapture started: %dx%d @ %dfps, backend=%s",
                logical_w, logical_h, self.fps, _writer_backend_name(),
            )

            self.start_time = time.monotonic()
            self._ready.set()

            try:
                while not self._stop.is_set():
                    t0 = time.monotonic()
                    shot = sct.grab(mon)
                    frame = np.frombuffer(
                        shot.bgra, dtype=np.uint8
                    ).reshape(shot.height, shot.width, 4)
                    bgr = frame[:, :, :3].copy()

                    # Down-scale Retina / HiDPI to logical size
                    if bgr.shape[1] != logical_w or bgr.shape[0] != logical_h:
                        if _HAS_CV2:
                            bgr = cv2.resize(
                                bgr, (logical_w, logical_h),
                                interpolation=cv2.INTER_AREA,
                            )
                        else:
                            # numpy-based resize via Pillow (no cv2 needed)
                            from PIL import Image

                            pil_img = Image.fromarray(bgr)
                            pil_img = pil_img.resize(
                                (logical_w, logical_h), Image.LANCZOS
                            )
                            bgr = np.array(pil_img)

                    writer.write(bgr)

                    remaining = interval - (time.monotonic() - t0)
                    if remaining > 0:
                        self._stop.wait(remaining)
            finally:
                writer.release()
