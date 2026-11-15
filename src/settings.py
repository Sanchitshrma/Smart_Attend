"""Central place for reading configuration.

Order of precedence: environment variable -> Streamlit secrets -> default.
Nothing in this module ever logs or returns a value to the UI.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import streamlit as st
from src.config_errors import SettingsError


def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        secret = st.secrets.get(name)  # raises if no secrets file exists
    except Exception:
        secret = None
    return str(secret) if secret else default


def _float_setting(name: str, default: str, minimum: float, maximum: float) -> float:
    raw = get_setting(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise SettingsError(f"{name} must be a number between {minimum:g} and {maximum:g}.") from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise SettingsError(f"{name} must be a finite number between {minimum:g} and {maximum:g}.")
    return value


def _int_setting(name: str, default: str, minimum: int, maximum: int) -> int:
    raw = get_setting(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise SettingsError(f"{name} must be an integer between {minimum} and {maximum}.") from None
    if not minimum <= value <= maximum:
        raise SettingsError(f"{name} must be an integer between {minimum} and {maximum}.")
    return value


# Times are stored in UTC and shown in this time zone (cloud servers run in UTC).
APP_TIMEZONE = get_setting("APP_TIMEZONE", "Asia/Kolkata")

# Tunable recognition thresholds (documented in README).
FACE_MATCH_THRESHOLD = _float_setting("FACE_MATCH_THRESHOLD", "0.45", 0.0, 1.0)
# Voice: minimum raw cosine similarity for a speaker match. Raised from 0.55 because two DIFFERENT
# Resemblyzer voices typically score ~0.58 (see scripts/evaluate_voice.py).
VOICE_MATCH_THRESHOLD = _float_setting("VOICE_MATCH_THRESHOLD", "0.70", 0.0, 1.0)
# Ambiguity margin: if the 2nd-best candidate is also within the threshold and closer than
# this to the best, the face is sent to teacher review instead of auto-assigned.
# PROVISIONAL default: not yet tuned on real labelled data (see evaluation/README).
FACE_MATCH_MARGIN = _float_setting("FACE_MATCH_MARGIN", "0.05", 0.0, 2.0)
# dlib re-sampling jitter for embeddings (0 = fastest; higher = slightly more stable, slower).
FACE_NUM_JITTERS = _int_setting("FACE_NUM_JITTERS", "0", 0, 100)
# Faces whose bounding box is smaller than this (pixels) are flagged as low quality.
FACE_MIN_SIZE_PX = _int_setting("FACE_MIN_SIZE_PX", "40", 1, 2000)
# Voice: 2nd-best speaker within this similarity of the best (and above threshold) => teacher review.
# PROVISIONAL, not tuned on labelled recordings.
VOICE_MATCH_MARGIN = _float_setting("VOICE_MATCH_MARGIN", "0.05", 0.0, 2.0)
# Voice check: the answer is compared with every stored voice profile, and the called student must
# stand out from that crowd by VOICE_MIN_ZSCORE standard deviations and beat every other student by
# VOICE_Z_MARGIN (used when at least VOICE_MIN_COHORT voice profiles exist). Tuned on synthetic
# recordings (scripts/evaluate_voice.py); re-tune on real classroom audio.
VOICE_MIN_ZSCORE = _float_setting("VOICE_MIN_ZSCORE", "2.5", 0.0, 10.0)
VOICE_Z_MARGIN = _float_setting("VOICE_Z_MARGIN", "0.5", 0.0, 5.0)
VOICE_MIN_COHORT = _int_setting("VOICE_MIN_COHORT", "8", 4, 100000)

# ---- learning over time (extra face templates from teacher-confirmed attendance)
# After the teacher saves a session, confident face matches they kept are stored as extra templates.
FACE_LEARNING = _int_setting("FACE_LEARNING", "1", 0, 1)                 # 1 = on, 0 = off
FACE_MAX_TEMPLATES = _int_setting("FACE_MAX_TEMPLATES", "10", 1, 100)    # learned templates kept per student
# A new sample must differ from every stored template by at least this (otherwise it adds nothing)...
FACE_LEARN_MIN_NOVELTY = _float_setting("FACE_LEARN_MIN_NOVELTY", "0.12", 0.0, 1.0)
# ...and must stay this close to the student's registration face, so errors cannot drift over time.
FACE_LEARN_MAX_FROM_ENROLMENT = _float_setting("FACE_LEARN_MAX_FROM_ENROLMENT", "0.50", 0.0, 1.0)

# ---- video attendance
VIDEO_SAMPLE_FPS = _float_setting("VIDEO_SAMPLE_FPS", "2", 0.5, 10.0)    # frames analysed per second of video
VIDEO_MAX_SECONDS = _int_setting("VIDEO_MAX_SECONDS", "20", 3, 120)
VIDEO_MIN_SIGHTINGS = _int_setting("VIDEO_MIN_SIGHTINGS", "2", 1, 20)      # frames a face must appear in to be auto-marked
# Two sightings closer than this (face distance) are treated as the same person across frames.
VIDEO_LINK_DISTANCE = _float_setting("VIDEO_LINK_DISTANCE", "0.40", 0.0, 1.0)

# ---- voice check for students the camera missed (1:1 verification)
VOICE_ANSWER_MIN_SPEECH_SECONDS = _float_setting("VOICE_ANSWER_MIN_SPEECH_SECONDS", "1.5", 0.5, 10.0)

# ---- account security
# One-time secret that lets the FIRST administrator be created. Remove it from your secrets after setup.
ADMIN_SETUP_TOKEN = get_setting("ADMIN_SETUP_TOKEN")
LOGIN_MAX_FAILURES = _int_setting("LOGIN_MAX_FAILURES", "5", 1, 100)
LOGIN_WINDOW_MINUTES = _int_setting("LOGIN_WINDOW_MINUTES", "15", 1, 1440)
STUDENT_REGISTER_PER_HOUR = _int_setting("STUDENT_REGISTER_PER_HOUR", "30", 1, 10000)
# Student face sign-in: failed face checks allowed per device/network in the window, and the total
# number of checks (each one runs face models, so this also limits abuse of server time).
STUDENT_FACE_MAX_FAILURES = _int_setting("STUDENT_FACE_MAX_FAILURES", "8", 1, 1000)
STUDENT_FACE_MAX_ATTEMPTS = _int_setting("STUDENT_FACE_MAX_ATTEMPTS", "30", 1, 10000)
STUDENT_FACE_WINDOW_MINUTES = _int_setting("STUDENT_FACE_WINDOW_MINUTES", "15", 1, 1440)
SESSION_IDLE_TIMEOUT_MINUTES = _int_setting("SESSION_IDLE_TIMEOUT_MINUTES", "30", 1, 1440)
INVITE_MAX_DAYS = 14
