"""Video attendance: a short clip panning the classroom instead of single photos.

Steps:
  1. sample frames (VIDEO_SAMPLE_FPS per second, at most VIDEO_MAX_SECONDS of video),
  2. detect faces and compute a 128-d embedding for every face in every frame,
  3. group the sightings into PEOPLE: a face joins the person whose average embedding is closest
     (within VIDEO_LINK_DISTANCE), and two faces from the same frame are never the same person,
  4. match each person's AVERAGE embedding (steadier than any single blurry frame) with the
     same open-set rule as photos (matched / ambiguous / duplicate / unknown),
  5. a person seen in fewer than VIDEO_MIN_SIGHTINGS frames is never marked automatically.
The number of people seen in at least VIDEO_MIN_SIGHTINGS frames is the video's headcount.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from src.pipelines.face_pipeline import AMBIGUOUS, MATCHED, PhotoResult, current_gallery, detect_faces, match_encodings
from src.settings import VIDEO_LINK_DISTANCE, VIDEO_MAX_SECONDS, VIDEO_MIN_SIGHTINGS, VIDEO_SAMPLE_FPS

log = logging.getLogger("smartattend.video")
MAX_VIDEO_BYTES = 80 * 1024 * 1024
MAX_FRAME_SIDE = 1280
ALLOWED_VIDEO_EXTENSIONS = ["mp4", "mov", "m4v", "webm", "avi", "mkv"]


class VideoError(RuntimeError):
    """The video could not be used; the message is safe to show to users."""


@dataclass
class Person:
    sightings: list[tuple[int, tuple[int, int, int, int] | None, np.ndarray]] = field(default_factory=list)

    @property
    def frames(self) -> set[int]:
        return {f for f, _, _ in self.sightings}

    @property
    def centroid(self) -> np.ndarray:
        return np.mean([e for _, _, e in self.sightings], axis=0)

    @property
    def best_bbox(self):
        """Largest face box among the sightings (used for the small-face check)."""
        boxes = [b for _, b, _ in self.sightings if b]
        return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if boxes else None


@dataclass
class VideoResult:
    photo: PhotoResult              # one FaceMatch per person (same shape as a photo result)
    people: list[Person]
    frames_analysed: int
    seconds: float

    @property
    def headcount(self) -> int:
        """People seen in enough frames to count as really present in the room."""
        return sum(len(p.frames) >= VIDEO_MIN_SIGHTINGS for p in self.people)


def read_frames(video_bytes: bytes, fps: float = VIDEO_SAMPLE_FPS, max_seconds: int = VIDEO_MAX_SECONDS) -> tuple[list[np.ndarray], float]:
    """Decode a video and return (RGB frames sampled at `fps`, duration in seconds)."""
    import cv2

    if not video_bytes:
        raise VideoError("The video file is empty.")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise VideoError(f"The video is larger than {MAX_VIDEO_BYTES // (1024 * 1024)} MB. Record a shorter clip.")
    fd, path = tempfile.mkstemp(suffix=".video")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(video_bytes)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise VideoError("That file could not be read as a video. Use MP4, MOV or WebM.")
        native = cap.get(cv2.CAP_PROP_FPS) or 0.0
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        native = native if 1.0 <= native <= 240.0 else 30.0
        seconds = count / native if count > 0 else 0.0
        if seconds > max_seconds + 0.5:
            raise VideoError(f"The video is {seconds:.0f} s long; keep it under {max_seconds} s (a slow pan of the room is enough).")
        step = max(int(round(native / fps)), 1)
        frames, idx = [], 0
        while len(frames) < int(max_seconds * fps) + 1:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                if max(h, w) > MAX_FRAME_SIDE:
                    k = MAX_FRAME_SIDE / max(h, w)
                    rgb = cv2.resize(rgb, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
                frames.append(np.ascontiguousarray(rgb))
            idx += 1
        cap.release()
    finally:
        os.unlink(path)
    if not frames:
        raise VideoError("No frames could be read from the video.")
    return frames, (seconds or idx / native)


def group_sightings(per_frame: list[list[tuple[tuple | None, np.ndarray]]], link: float = VIDEO_LINK_DISTANCE) -> list[Person]:
    """Group faces across frames into people (greedy, closest pairs first, one face per person per frame)."""
    people: list[Person] = []
    for f, faces in enumerate(per_frame):
        pairs = []
        for i, (_, enc) in enumerate(faces):
            for p, person in enumerate(people):
                d = float(np.linalg.norm(person.centroid - enc))
                if d <= link:
                    pairs.append((d, i, p))
        used_faces, used_people = set(), set()
        for d, i, p in sorted(pairs):
            if i in used_faces or p in used_people:
                continue
            people[p].sightings.append((f, faces[i][0], np.asarray(faces[i][1])))
            used_faces.add(i)
            used_people.add(p)
        for i, (bbox, enc) in enumerate(faces):
            if i not in used_faces:
                people.append(Person([(f, bbox, np.asarray(enc))]))
    return people


def match_people(people: list[Person], ids, matrix, allowed_ids=None, min_sightings: int = VIDEO_MIN_SIGHTINGS) -> PhotoResult:
    """Match every person's average embedding; people seen too briefly are sent to review."""
    result = match_encodings([p.centroid for p in people], ids, matrix, allowed_ids=allowed_ids,
                             bboxes=[p.best_bbox for p in people])
    _merge_split_people(result, people)
    for face, person in zip(result.faces, people):
        face.sightings = len(person.frames)
        if face.status == MATCHED and face.sightings < min_sightings:
            face.status = AMBIGUOUS
            face.flags.append("seen_once")
    return result


def _merge_split_people(result: PhotoResult, people: list[Person]) -> None:
    """One student split into two groups (e.g. after turning their head) is the same person if the
    groups never appear in the same frame: merge them instead of flagging a duplicate claim."""
    from src.pipelines.face_pipeline import DUPLICATE
    by_student: dict[int, list[int]] = {}
    for k, f in enumerate(result.faces):
        if f.student_id is not None and f.status in (MATCHED, DUPLICATE):
            by_student.setdefault(f.student_id, []).append(k)
    drop = set()
    for sid, ks in by_student.items():
        if len(ks) < 2:
            continue
        frames = [people[k].frames for k in ks]
        if any(frames[i] & frames[j] for i in range(len(ks)) for j in range(i + 1, len(ks))):
            continue  # seen together in one frame: really two different faces, keep for review
        keep = min(ks, key=lambda k: result.faces[k].distance)
        for k in ks:
            if k != keep:
                people[keep].sightings.extend(people[k].sightings)
                drop.add(k)
        kept = result.faces[keep]
        kept.status = MATCHED
        kept.flags = [x for x in kept.flags if x != "duplicate_claim"]
    for k in sorted(drop, reverse=True):
        del result.faces[k]
        del people[k]
    for i, f in enumerate(result.faces):
        f.index = i
    result.num_faces = len(result.faces)


MAX_LIVE_FRAMES = 60
MAX_LIVE_FRAME_BYTES = 2 * 1024 * 1024


def decode_live_frames(frames_b64, times_ms=None) -> tuple[list[np.ndarray], float]:
    """Decode JPEG frames captured live in the browser (nothing is written to disk)."""
    import base64
    import io

    if not isinstance(frames_b64, list) or not frames_b64:
        raise VideoError("No frames were received from the camera. Allow camera access and try again.")
    frames = []
    for item in frames_b64[:MAX_LIVE_FRAMES]:
        if not isinstance(item, str):
            continue
        data = item.split(",", 1)[1] if item.startswith("data:") else item
        try:
            raw = base64.b64decode(data, validate=False)
            if not raw or len(raw) > MAX_LIVE_FRAME_BYTES:
                continue
            img = Image.open(io.BytesIO(raw))
            if img.format not in ("JPEG", "PNG", "WEBP"):
                continue
            img = img.convert("RGB")
        except Exception:  # noqa: BLE001 - skip a corrupt frame
            continue
        if max(img.size) > MAX_FRAME_SIDE:
            img.thumbnail((MAX_FRAME_SIDE, MAX_FRAME_SIDE))
        frames.append(np.ascontiguousarray(np.asarray(img, dtype=np.uint8)))
    if not frames:
        raise VideoError("The camera frames could not be read. Please record again.")
    seconds = 0.0
    if isinstance(times_ms, list) and times_ms and all(isinstance(t, (int, float)) for t in times_ms):
        seconds = max(float(times_ms[-1]) / 1000.0, 0.0)
    return frames, seconds


def analyze_frames(frames: list[np.ndarray], seconds: float, allowed_ids: set[int] | None = None, progress=None) -> VideoResult:
    """Same pipeline as a video file, for frames captured live in the browser."""
    per_frame = []
    for k, frame in enumerate(frames):
        per_frame.append(detect_faces(Image.fromarray(frame)))
        if progress:
            progress((k + 1) / len(frames))
    people = group_sightings(per_frame)
    _, ids, matrix = current_gallery()
    return VideoResult(match_people(people, ids, matrix, allowed_ids), people, len(frames), seconds)


def analyze_video(video_bytes: bytes, allowed_ids: set[int] | None = None, progress=None) -> VideoResult:
    frames, seconds = read_frames(video_bytes)
    return analyze_frames(frames, seconds, allowed_ids, progress)

