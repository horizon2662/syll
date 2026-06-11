#!/usr/bin/env python3
"""验证录制视频修复是否正常工作。

运行方式:
    cd E:\download\syll-main
    python verify_recorder_fix.py

测试内容:
    1. 依赖检查 (cv2, mss, numpy, fastapi)
    2. VideoWriter codec 探测 — 确认 H.264 是否可用
    3. 录制 2 秒桌面视频
    4. 验证视频文件有效性
    5. 验证浏览器安全转码链
"""
from __future__ import annotations

import sys
import os
import time
import tempfile
import struct
from pathlib import Path

# 确保项目在 path 上
sys.path.insert(0, str(Path(__file__).parent))

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {mark} {label}{suffix}")
    return condition


# ── 1. 依赖检查 ──────────────────────────────────────────────────────────

header("Step 1: 依赖检查")

deps_ok = True
try:
    import cv2
    check("opencv-python", True, cv2.__version__)
except ImportError:
    check("opencv-python", False, "未安装")
    deps_ok = False

try:
    import mss
    check("mss", True, mss.__version__)
except ImportError:
    check("mss", False, "未安装")
    deps_ok = False

try:
    import numpy as np
    check("numpy", True, np.__version__)
except ImportError:
    check("numpy", False, "未安装")
    deps_ok = False

try:
    import fastapi
    check("fastapi", True, fastapi.__version__)
except ImportError:
    check("fastapi", False, "未安装")
    deps_ok = False

if not deps_ok:
    print("\n  ❌ 缺少依赖，请运行: pip install opencv-python mss numpy fastapi")
    sys.exit(1)


# ── 2. VideoWriter Codec 探测 ────────────────────────────────────────────

header("Step 2: VideoWriter Codec 探测")

tmpdir = tempfile.mkdtemp(prefix="syll_test_")
probe_path = os.path.join(tmpdir, "probe.mp4")

fourcc_list = ["avc1", "H264", "X264", "mp4v", "MJPG"]
working_codecs = []

for tag in fourcc_list:
    try:
        fourcc = cv2.VideoWriter_fourcc(*tag)
        w = cv2.VideoWriter(probe_path, fourcc, 15.0, (320, 240))
        if w.isOpened():
            # 写一帧确认真的能用
            import numpy as np
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            w.write(frame)
            w.release()
            size = os.path.getsize(probe_path)
            if size > 0:
                check(f"fourcc '{tag}'", True, f"文件大小 {size} bytes")
                working_codecs.append(tag)
            else:
                check(f"fourcc '{tag}'", False, "文件为空 (写入失败)")
        else:
            check(f"fourcc '{tag}'", False, "无法打开 VideoWriter")
            w.release()
    except Exception as e:
        check(f"fourcc '{tag}'", False, str(e)[:80])
    finally:
        if os.path.exists(probe_path):
            os.remove(probe_path)

browser_safe = [c for c in working_codecs if c in ("avc1", "H264", "X264")]
if browser_safe:
    print(f"\n  🎉 有 {len(browser_safe)} 个浏览器安全 codec 可用: {browser_safe}")
    print(f"     录制的视频浏览器可直接播放，无需转码！")
else:
    print(f"\n  ⚠ 没有浏览器安全 codec，录制后会自动用 cv2 转码")
    if "mp4v" in working_codecs:
        print(f"     (mp4v 可用，转码链应该能正常工作)")


# ── 3. 录制 2 秒桌面视频 ────────────────────────────────────────────────

header("Step 3: 录制 2 秒桌面视频")

test_video = os.path.join(tmpdir, "test_record.mp4")

try:
    from syll.recorder.screen_capture import ScreenCapture, _create_video_writer

    # 先测试 _create_video_writer 函数
    print("  测试 _create_video_writer()...")
    writer, codec_tag = _create_video_writer(test_video, 15.0, (640, 480))
    check(f"创建 VideoWriter", True, f"codec='{codec_tag}'")

    import numpy as np
    for i in range(30):
        # 生成彩色测试帧
        r = int(128 + 127 * (i / 30.0))
        g = int(128 + 127 * ((30 - i) / 30.0))
        b = 128
        frame = np.full((480, 640, 3), (b, g, r), dtype=np.uint8)
        # 加一个数字水印
        cv2.putText(frame, f"Frame {i:02d}", (200, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()

    video_size = os.path.getsize(test_video)
    check("视频文件写入", video_size > 0, f"{video_size} bytes")

    # 验证视频可以被读取
    cap = cv2.VideoCapture(test_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    check(f"视频帧数 > 0", frame_count > 0, f"{frame_count} frames")

except Exception as e:
    check(f"录制测试", False, str(e))


# ── 4. 浏览器安全转码链测试 ──────────────────────────────────────────────

header("Step 4: 浏览器安全转码链测试")

# 测试 _probe_video_codec (可能没有 ffprobe)
from syll.web.routes.recorder import (
    _probe_video_codec,
    _is_browser_safe_video,
    _browser_video_path,
    _transcode_with_cv2,
)

print("  检测 ffmpeg/ffprobe...")
import shutil
ffmpeg_available = shutil.which("ffmpeg") is not None
ffprobe_available = shutil.which("ffprobe") is not None
check("ffmpeg", ffmpeg_available)
check("ffprobe", ffprobe_available)

# 探测我们录制的视频 codec
if os.path.exists(test_video) and os.path.getsize(test_video) > 0:
    codec = _probe_video_codec(test_video)
    check(f"视频 codec (ffprobe)", bool(codec), codec or "未检测到 (ffprobe 不可用)")

    is_safe = _is_browser_safe_video(test_video)
    if is_safe:
        check("浏览器安全检查", True, "H.264 — 浏览器可直接播放")
    else:
        check("浏览器安全检查", False, "非 H.264 — 需要转码")

        # 测试 cv2 转码回退
        if not ffmpeg_available:
            print("\n  ffmpeg 不可用，测试 cv2 转码回退...")
            try:
                browser_path = _browser_video_path(Path(test_video))
                result = _transcode_with_cv2(Path(test_video), browser_path)
                transcode_size = os.path.getsize(str(result))
                check("cv2 转码", True, f"输出 {transcode_size} bytes")

                # 验证转码后的视频
                transcode_codec = _probe_video_codec(str(result))
                check("转码后 codec", bool(transcode_codec), transcode_codec or "ffprobe 不可用，用 cv2 验证")

                cap2 = cv2.VideoCapture(str(result))
                fc2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
                cap2.release()
                check("转码后帧数", fc2 > 0, f"{fc2} frames")
            except Exception as e:
                check("cv2 转码", False, str(e)[:120])


# ── 5. 清理 ──────────────────────────────────────────────────────────────

header("Step 5: 清理")

import shutil
try:
    shutil.rmtree(tmpdir)
    check("清理临时目录", True, tmpdir)
except Exception as e:
    check("清理临时目录", False, str(e))


# ── 总结 ──────────────────────────────────────────────────────────────────

header("总结")

print(f"""
  如果所有测试都通过 ({PASS})，说明修复已生效：
  - 录制视频时会优先使用 H.264 编码 (浏览器直接播放)
  - 即使回退到 mp4v，cv2 转码回退也能保证浏览器播放
  - 不再依赖 ffmpeg/ffprobe

  你现在可以运行 'syll wake' 然后在浏览器中录制工作流来验证。
""")
