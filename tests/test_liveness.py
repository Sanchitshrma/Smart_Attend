"""Blink liveness check used at student sign-in and registration."""
import base64
import io
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.pipelines.liveness import (MIN_FRAMES, Challenge, LivenessError, count_blinks, face_size_ok, new_challenge,
                                    verify_capture)

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(3)


def series(n=60, blinks_at=(), open_ear=0.30, closed_ear=0.12, noise=0.01):
    s = [open_ear + rng.normal(0, noise) for _ in range(n)]
    for b in blinks_at:
        s[b] = s[b + 1] = closed_ear
    return s


def test_blinks_are_counted_from_an_ear_series():
    assert count_blinks(series(blinks_at=(20,)))[0] == 1
    assert count_blinks(series(blinks_at=(10, 30, 45)))[0] == 3
    assert count_blinks(series())[0] == 0                       # a still photo never blinks


def test_noise_missing_frames_and_a_closed_start_do_not_create_blinks():
    s = series(noise=0.015)
    s[5] = s[6] = None                                          # face lost for two frames
    assert count_blinks(s)[0] == 0
    closed_start = [0.12] * 5 + series(55)                      # eyes closed when recording began
    assert count_blinks(closed_start)[0] == 0
    squint = series(blinks_at=(20,), closed_ear=0.25)           # a small twitch is not a blink
    assert count_blinks(squint)[0] == 0


def test_eyes_too_small_to_measure_count_nothing():
    assert count_blinks(series(blinks_at=(20,), open_ear=0.10, closed_ear=0.03))[0] == 0


def test_challenges_are_random_one_time_and_expire():
    a, b = new_challenge(), new_challenge()
    assert a.nonce != b.nonce and 1 <= a.blinks <= 3 and 800 <= a.prompt_at_ms <= 2000
    old = Challenge("n", 1, time.time() - 1000)
    assert old.expired() and not a.expired()


def _jpeg(img):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def test_capture_is_rejected_for_a_wrong_nonce_expiry_bad_timing_or_too_few_frames():
    ch = Challenge("abc", 1, time.time(), prompt_at_ms=1000)
    frame = _jpeg(Image.new("RGB", (64, 64), (120, 110, 100)))
    good_times = [i * 83.0 for i in range(40)]
    with pytest.raises(LivenessError):
        verify_capture({"nonce": "other", "frames": [frame] * 40, "times": good_times}, ch)
    with pytest.raises(LivenessError):
        verify_capture({"nonce": "abc", "frames": [frame] * 40, "times": good_times}, Challenge("abc", 1, time.time() - 999))
    with pytest.raises(LivenessError):
        verify_capture({"nonce": "abc", "frames": [frame] * 40, "times": good_times[::-1]}, ch)   # time runs backwards
    with pytest.raises(LivenessError):
        verify_capture({"nonce": "abc", "frames": [frame] * 40, "times": good_times[:10]}, ch)    # lengths differ
    with pytest.raises(LivenessError):
        verify_capture({"nonce": "abc", "frames": [frame] * (MIN_FRAMES - 1),
                        "times": good_times[:MIN_FRAMES - 1]}, ch)
    with pytest.raises(LivenessError):
        verify_capture({"nonce": "abc", "frames": [frame] * 20, "times": [i * 50.0 for i in range(20)]}, ch)  # ends too soon
    with pytest.raises(LivenessError):
        verify_capture(None, ch)


def test_face_size_rule():
    assert face_size_ok((0, 0, 120, 130), 80)
    assert not face_size_ok((0, 0, 60, 130), 80)
    assert not face_size_ok(None, 80)


# ------------------------------------------------------------------ real dlib landmarks
def _landmarks(img):
    from src.pipelines.face_pipeline import load_dlib_models
    det, sp, _ = load_dlib_models()
    a = np.asarray(img)
    r = det(a, 1)
    s = sp(a, r[0])
    return [(s.part(i).x, s.part(i).y) for i in range(68)]


def _close_eyes(img, lm):
    """Simulated closed eyelids: skin colour over each eye with a thin dark lash line."""
    a = np.asarray(img).copy()
    for idx in (range(36, 42), range(42, 48)):
        p = np.asarray([lm[i] for i in idx]); x0, y0 = p.min(0); x1, y1 = p.max(0)
        h = max(y1 - y0, 2); cx = (x0 + x1) // 2; pad = int(h * 1.2) + 3
        skin = a[min(y1 + pad, a.shape[0] - 1), cx].astype(int)
        a[int(y0 - pad * 0.6):int(y1 + pad * 0.4), int(x0 - 3):int(x1 + 4)] = skin
        ly = int((y0 + y1) / 2 + h * 0.2); a[ly:ly + 2, int(x0):int(x1)] = (skin * 0.35).astype(np.uint8)
    return Image.fromarray(a)


def _clip(photo, blinks_at, n=48, swap=None):
    from PIL import ImageEnhance
    r = np.random.default_rng(0)
    base = Image.open(ROOT / "class_data" / "photos" / photo).convert("RGB").resize((480, 480))
    closed = _close_eyes(base, _landmarks(base))
    other = Image.open(ROOT / "class_data" / "photos" / swap).convert("RGB").resize((480, 480)) if swap else None
    frames = []
    for k in range(n):
        src = closed if any(b <= k < b + 2 for b in blinks_at) else base
        if other is not None and k >= n // 2:
            src = other
        dx, dy = r.integers(-6, 7, 2)
        im = src.rotate(float(r.uniform(-2, 2)), translate=(int(dx), int(dy)), fillcolor=(30, 30, 30))
        frames.append(ImageEnhance.Brightness(im).enhance(float(r.uniform(0.92, 1.08))))
    return frames


@pytest.fixture(scope="module")
def dlib_ready():
    pytest.importorskip("dlib")
    if not (ROOT / "class_data" / "photos" / "student_11.jpg").exists():
        pytest.skip("sample photos not available")


def test_real_landmarks_pass_a_live_blink_and_reject_a_photo(dlib_ready):
    from src.pipelines.liveness import check_liveness
    times = [k * 83.0 for k in range(48)]
    live = check_liveness(_clip("student_11.jpg", [24]), 1, times, prompt_at_ms=1000)
    assert live.passed and live.blinks == 1 and live.embedding is not None and len(live.embedding) == 128
    photo = check_liveness(_clip("student_11.jpg", []), 1, times, prompt_at_ms=1000)
    assert not photo.passed and photo.blinks == 0


def test_real_landmarks_reject_wrong_count_early_blinks_and_face_swaps(dlib_ready):
    from src.pipelines.liveness import check_liveness
    times = [k * 83.0 for k in range(48)]
    two = _clip("student_16.jpg", [16, 32])
    assert check_liveness(two, 2, times, prompt_at_ms=1000).passed
    assert not check_liveness(two, 1, times, prompt_at_ms=1000).passed            # exact count required
    early = check_liveness(_clip("student_11.jpg", [3]), 1, times, prompt_at_ms=1500)
    assert not early.passed                                                        # blink before the prompt
    swapped = check_liveness(_clip("student_11.jpg", [12], swap="student_24.jpg"), 1, times, prompt_at_ms=500)
    assert not swapped.passed and "face changed" in swapped.reason                 # a different person mid-way


def test_face_sign_in_is_rate_limited_per_session_without_locking_out_the_classroom_network(fake_db):
    from src import settings as cfg
    from src.database import db
    for _ in range(cfg.STUDENT_FACE_MAX_FAILURES):
        db.check_student_face_attempt("session-a", "10.0.0.1")
        db.record_student_face_attempt("session-a", "10.0.0.1", False)
    with pytest.raises(db.RateLimited):
        db.check_student_face_attempt("session-a", "10.0.0.1")
    db.check_student_face_attempt("session-b", "10.0.0.1")     # a classmate on the same network still can
    stored = {r["key"] for r in fake_db.tables["auth_attempts"]}
    assert not any("10.0.0.1" in k or "session-a" in k for k in stored)   # only hashes are stored
