"""Blink-based liveness check for student sign-in and registration.

The browser records ~5 seconds of the front camera at ~12 frames per second. At a RANDOM moment
chosen by the server (0.8-2.0 s in), the student is asked to blink a RANDOM number of times (1-3).
The server then:
  1. finds exactly one face in the frames (68 dlib landmarks per frame),
  2. measures how open the eyes are in every frame with the eye aspect ratio (EAR),
  3. counts blinks AFTER the prompt appeared: the EAR drops well below the student's own open-eye
     level and recovers; the count must match exactly,
  4. checks that the face is the SAME person in every sampled frame (embedding distance),
  5. returns the face embedding taken from the same frames, so the identity used for sign-in or
     registration is bound to the frames that passed the liveness check.

A printed photo cannot blink, and a pre-recorded video does not know when the prompt will appear
or how many blinks it will ask for.
This is one layer: face matching, duplicate-face blocking, rate limits, a one-time challenge with
an expiry, and teacher review remain in place. It does not stop a determined attacker with a live
deepfake; see the project report for limits.
"""
from __future__ import annotations

import base64
import io
import logging
import math
import secrets
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger("smartattend.liveness")

MIN_FRAMES = 20                 # ~1.7 s at 12 fps
MAX_FRAMES = 120
MAX_FRAME_BYTES = 400 * 1024
MAX_FRAME_SIDE = 640
FACE_PRESENT_FRACTION = 0.7     # one face must be visible in at least 70% of frames
CLOSED_RATIO = 0.72             # eye counts as closed below 72% of the open-eye level...
OPEN_RATIO = 0.86               # ...and open again above 86% of it
MIN_OPEN_EAR = 0.16             # eyes that never look open (sunglasses, dark room) cannot be checked
SAME_PERSON_DISTANCE = 0.50     # max distance between ANY two frames of the recording
IDENTITY_FRAMES = 8              # frames spread over the recording used for the identity check
CHALLENGE_TTL_SECONDS = 90


class LivenessError(RuntimeError):
    """The frames could not be used; the message is safe to show to users."""


@dataclass
class Challenge:
    nonce: str
    blinks: int
    issued_at: float
    prompt_at_ms: int = 1200          # when the browser shows "Blink now", ms after recording starts

    def expired(self, now: float | None = None) -> bool:
        return (now or time.time()) - self.issued_at > CHALLENGE_TTL_SECONDS


def new_challenge() -> Challenge:
    """A one-time challenge: random nonce, random blink count (1-3) and random prompt time."""
    return Challenge(secrets.token_urlsafe(16), 1 + secrets.randbelow(3), time.time(), 800 + secrets.randbelow(1201))

RECORD_MS_AFTER_PROMPT = 3500         # the browser keeps recording this long after the prompt
PROMPT_EARLY_MS = 150                 # reaction time allowance before the prompt


@dataclass
class LivenessResult:
    passed: bool
    reason: str
    blinks: int = 0
    required: int = 0
    frames: int = 0
    frames_with_face: int = 0
    open_ear: float | None = None
    embedding: list[float] | None = None
    bbox: tuple[int, int, int, int] | None = None
    ears: list[float] = field(default_factory=list, repr=False)


def eye_aspect_ratio(pts) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|) for the six landmarks of one eye."""
    p = np.asarray(pts, dtype=np.float64)
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal == 0:
        return 0.0
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * horizontal))


def count_blinks(ears: list[float | None], open_level: float | None = None) -> tuple[int, float | None]:
    """Count blinks in a per-frame EAR series (None = no face in that frame).

    The open-eye level is the student's own 80th-percentile EAR, so it adapts to eye shape and
    camera distance. A blink is an open -> closed -> open sequence; frames without a face are
    skipped rather than counted as closed eyes.
    """
    vals = [e for e in ears if e is not None]
    if open_level is None:
        if len(vals) < 5:
            return 0, None
        open_level = float(np.percentile(vals, 80))
    if open_level < MIN_OPEN_EAR:
        return 0, open_level
    closed_below, open_above = open_level * CLOSED_RATIO, open_level * OPEN_RATIO
    blinks, state = 0, "unknown"
    for e in ears:
        if e is None:
            continue
        if e >= open_above:
            if state == "closed":
                blinks += 1
            state = "open"
        elif e <= closed_below and state == "open":
            state = "closed"
    return blinks, open_level


def decode_frames(frames_b64: list[str], times_ms: list[float] | None = None):
    """Decode JPEG/PNG data URLs or base64 strings from the browser, with size limits.

    Returns the images, or (images, times) when `times_ms` is given (times of skipped frames dropped)."""
    if not isinstance(frames_b64, list) or not frames_b64:
        raise LivenessError("No camera frames were received. Allow camera access and try again.")
    if len(frames_b64) > MAX_FRAMES:
        frames_b64 = frames_b64[:MAX_FRAMES]
    out, kept = [], []
    for idx, item in enumerate(frames_b64):
        if not isinstance(item, str):
            continue
        data = item.split(",", 1)[1] if item.startswith("data:") else item
        try:
            raw = base64.b64decode(data, validate=False)
        except (ValueError, TypeError):
            continue
        if not raw or len(raw) > MAX_FRAME_BYTES:
            continue
        try:
            img = Image.open(io.BytesIO(raw))
            if img.format not in ("JPEG", "PNG", "WEBP"):
                continue
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception:  # noqa: BLE001 - a corrupt frame is skipped, not fatal
            continue
        if max(img.size) > MAX_FRAME_SIDE:
            img.thumbnail((MAX_FRAME_SIDE, MAX_FRAME_SIDE))
        out.append(img)
        kept.append(idx)
    if len(out) < MIN_FRAMES:
        raise LivenessError("The camera recording was too short. Keep your face in the frame until the check finishes.")
    if times_ms is None:
        return out
    return out, [times_ms[i] for i in kept]


def _models():
    from src.pipelines.face_pipeline import load_dlib_models
    return load_dlib_models()


def check_liveness(frames: list[Image.Image], required_blinks: int, times_ms: list[float] | None = None,
                   prompt_at_ms: int | None = None) -> LivenessResult:
    """Run the blink check on decoded frames (see module docstring).

    With `times_ms` and `prompt_at_ms`, only blinks that start after the prompt are counted."""
    detector, shape_predictor, recognizer = _models()
    ears: list[float | None] = []
    shapes: list[tuple[int, object, np.ndarray] | None] = []
    rect = None
    multi = 0
    for k, img in enumerate(frames):
        a = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        # The landmark model expects the detector's own boxes, so detect in every frame; if a
        # frame is missed, reuse the previous detector box (the face barely moves in 1/12 s).
        found = detector(a, 0) or detector(a, 1)
        if len(found) > 1:
            multi += 1
        if found:
            rect = max(found, key=lambda r: r.width() * r.height())
        elif rect is None:
            ears.append(None)
            shapes.append(None)
            continue
        s = shape_predictor(a, rect)
        pts = [(s.part(i).x, s.part(i).y) for i in range(68)]
        ears.append((eye_aspect_ratio(pts[36:42]) + eye_aspect_ratio(pts[42:48])) / 2.0 if found else None)
        shapes.append((k, s, a) if found else None)

    seen = sum(e is not None for e in ears)
    res = LivenessResult(False, "", required=required_blinks, frames=len(frames), frames_with_face=seen, ears=[e for e in ears if e is not None])
    if multi > len(frames) // 10:
        res.reason = "More than one face was visible. Only you should be in the frame."
        return res
    if seen < FACE_PRESENT_FRACTION * len(frames):
        res.reason = "Your face was not visible for most of the check. Hold the phone steady with your face in the oval."
        return res
    counted = ears
    if times_ms is not None and prompt_at_ms is not None and len(times_ms) == len(ears):
        # Frames before the prompt still set the open-eye level, but their blinks do not count.
        _, res.open_ear = count_blinks(ears)
        counted = [e if t >= prompt_at_ms - PROMPT_EARLY_MS else None for e, t in zip(ears, times_ms)]
        first = next((i for i, t in enumerate(times_ms) if t >= prompt_at_ms - PROMPT_EARLY_MS), len(ears))
        if first > 0 and ears[first - 1] is not None:
            counted[first - 1] = ears[first - 1]   # the last frame before the prompt anchors the open state
    res.blinks, level = count_blinks(counted, open_level=res.open_ear)
    res.open_ear = level
    if res.open_ear is not None and res.open_ear < MIN_OPEN_EAR:
        res.reason = "Your eyes could not be seen clearly. Remove sunglasses and face the light."
        return res

    # Identity: the SAME person must be in front of the camera for the whole recording. Compare
    # faces from up to 8 frames spread over the recording, every pair (not just against the
    # average, which a half-and-half swap of two people could pass).
    with_face = [sh for sh in shapes if sh is not None]
    spread = [with_face[int(i)] for i in np.linspace(0, len(with_face) - 1, min(IDENTITY_FRAMES, len(with_face)))] if with_face else []
    embs = {k: np.array(recognizer.compute_face_descriptor(a, s, 0)) for k, s, a in spread}
    vecs = list(embs.values())
    if len(vecs) < 3:
        res.reason = "Keep your face in the oval and look at the camera."
        return res
    worst = max(float(np.linalg.norm(vecs[i] - vecs[j])) for i in range(len(vecs)) for j in range(i + 1, len(vecs)))
    if worst > SAME_PERSON_DISTANCE:
        res.reason = "The face changed during the check. Only one person should be in front of the camera."
        return res
    # The stored template comes from open-eye frames of this same recording.
    open_ks = {k for (k, _, _), e in zip([sh or (-1, None, None) for sh in shapes], ears)
               if k >= 0 and e is not None and res.open_ear and e >= res.open_ear * OPEN_RATIO}
    picks = [sh for sh in spread if sh[0] in open_ks] or spread
    mean = np.mean([embs[k] for k, _, _ in picks], axis=0)
    last = picks[len(picks) // 2][1].rect
    res.bbox = (last.left(), last.top(), last.right(), last.bottom())
    res.embedding = [float(x) for x in mean]

    if res.blinks < required_blinks:
        res.reason = (f"Blink not detected. Please blink {required_blinks} time{'s' if required_blinks > 1 else ''} "
                      "clearly when asked." if res.blinks == 0 else
                      f"Only {res.blinks} blink seen; please blink {required_blinks} times.")
        return res
    if res.blinks > required_blinks:
        res.reason = f"{res.blinks} blinks were seen but {required_blinks} were asked for. Please try again."
        return res
    res.passed, res.reason = True, "ok"
    return res


def verify_capture(payload: dict, challenge: Challenge | None) -> LivenessResult:
    """Validate the one-time challenge, decode frames and run the blink check."""
    if challenge is None or not isinstance(payload, dict):
        raise LivenessError("The check expired. Please start it again.")
    if payload.get("nonce") != challenge.nonce:
        raise LivenessError("This camera check does not belong to the current request. Please start again.")
    if challenge.expired():
        raise LivenessError("The check took too long and expired. Please start it again.")
    raw = payload.get("frames") or []
    times = payload.get("times")
    ok_times = (isinstance(raw, list) and isinstance(times, list) and len(times) == len(raw)
                and all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in times)
                and all(b >= a for a, b in zip(times, times[1:])))
    if not ok_times:
        raise LivenessError("The camera recording was incomplete. Please start the check again.")
    frames, times = decode_frames(raw[:MAX_FRAMES], times[:MAX_FRAMES])
    if times[-1] < challenge.prompt_at_ms + 1500:
        raise LivenessError("The recording stopped too soon after the prompt. Keep looking at the camera until it finishes.")
    return check_liveness(frames, challenge.blinks, times, challenge.prompt_at_ms)


def face_size_ok(bbox, min_px: int) -> bool:
    return bool(bbox) and min(bbox[2] - bbox[0], bbox[3] - bbox[1]) >= min_px and math.isfinite(bbox[0])
