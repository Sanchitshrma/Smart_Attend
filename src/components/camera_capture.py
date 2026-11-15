"""Live camera capture in the browser. Only still frames (JPEG, in memory) are sent to the server;
no video file is recorded or stored.

* mode="attendance": rear camera by default, a few frames per second for a few seconds while the
  teacher pans across the room.
* mode="liveness": front camera, ~12 frames per second, shows a blink prompt at a server-chosen time.
"""
from pathlib import Path

import streamlit.components.v1 as components

_camera_capture = components.declare_component(
    "smartattend_camera_capture",
    path=str(Path(__file__).with_name("camera_capture_web")),
)


def attendance_camera(key: str, seconds: int = 8, fps: int = 3):
    return _camera_capture(mode="attendance", seconds=seconds, fps=fps, label="Start live capture",
                           hint="Stand at the front, start the capture, then pan slowly across the class.",
                           key=key, default=None)


def liveness_camera(key: str, nonce: str, blinks: int, prompt_at_ms: int, after_prompt_ms: int, label: str = "Start face check"):
    return _camera_capture(mode="liveness", nonce=nonce, blinks=blinks, prompt_at_ms=prompt_at_ms,
                           after_prompt_ms=after_prompt_ms, label=label,
                           hint="Look straight at the camera in good light. When asked, blink the number of times shown.",
                           key=key, default=None)
