"""Live face check with a blink challenge, shared by student sign-in, registration and profile update.

Flow: the server creates a one-time challenge (random number of blinks, random prompt time, nonce,
90-second expiry) -> the browser records the front camera and shows the prompt -> the server checks
the nonce, the timing and the blinks, and that the same face is in every frame -> a NEW challenge is
issued for the next attempt, so a captured recording can never be submitted twice.
"""
from __future__ import annotations

import logging
import secrets

import streamlit as st

from src.components.camera_capture import liveness_camera
from src.database import db
from src.database.db import DatabaseError, RateLimited
from src.pipelines.liveness import (RECORD_MS_AFTER_PROMPT, LivenessError, LivenessResult, new_challenge,
                                    verify_capture)

log = logging.getLogger("smartattend.facecheck")

def client_keys() -> tuple[str, str | None]:
    """(this browser session, network address if Streamlit knows it). Only hashes are stored."""
    session = st.session_state.setdefault("_face_session_key", secrets.token_hex(16))
    try:
        ip = st.context.ip_address
    except Exception:  # noqa: BLE001
        ip = None
    return session, (str(ip) if ip else None)


def reset_face_check(purpose: str) -> None:
    st.session_state.pop(f"face_challenge_{purpose}", None)


def face_check(purpose: str, label: str = "Start face check") -> LivenessResult | None:
    """Render the blink check; return a PASSED result once, else None (errors are shown here)."""
    ckey = f"face_challenge_{purpose}"
    challenge = st.session_state.get(ckey)
    if challenge is None or challenge.expired():
        challenge = new_challenge()
        st.session_state[ckey] = challenge
    cap = liveness_camera(key=f"face_cam_{purpose}_{challenge.nonce}", nonce=challenge.nonce, blinks=challenge.blinks,
                          prompt_at_ms=challenge.prompt_at_ms, after_prompt_ms=RECORD_MS_AFTER_PROMPT, label=label)
    if not (isinstance(cap, dict) and cap.get("submitted") and cap.get("mode") == "liveness"):
        return None
    # One attempt per challenge: whatever happens next, the next attempt gets a fresh challenge.
    st.session_state[ckey] = new_challenge()
    keys = client_keys()
    try:
        db.check_student_face_attempt(*keys)
    except RateLimited as exc:
        st.error(str(exc))
        return None
    try:
        with st.spinner("Checking that you are really in front of the camera…"):
            res = verify_capture(cap, challenge)
    except LivenessError as exc:
        db.record_student_face_attempt(*keys, False)
        st.error(f"{exc} Press the camera button to try again.")
        return None
    except ImportError:
        st.error("Face recognition is not installed on this server.")
        return None
    except DatabaseError as exc:
        st.error(str(exc))
        return None
    except Exception:  # noqa: BLE001
        log.exception("Face check failed")
        st.error("The face check could not be completed. Please try again.")
        return None
    db.record_student_face_attempt(*keys, res.passed)
    if not res.passed:
        st.error(f"Face check failed: {res.reason} Press the camera button to try again.")
        return None
    return res
