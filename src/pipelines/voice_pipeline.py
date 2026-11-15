"""Voice identification pipeline (Resemblyzer speaker embeddings + cosine similarity).

Bulk attendance decides all speaking turns of one recording together:
  1. split the recording into speaker turns (silence-based),
  2. embed each turn (256-d Resemblyzer d-vector),
  3. score each turn against EVERY stored voice profile (the cohort) and normalise the
     scores (z-score), because raw cosine similarity between different Resemblyzer voices
     is high (typically ~0.58) and a fixed raw threshold cannot separate students,
  4. send a turn to review if a student from another class sounds as close,
  5. apply "each student answers once": a turn that was ambiguous between A and B becomes
     a match for B when another turn is already a confident match for A.
Ambiguous and unknown turns are never marked present automatically.

Heavy ML imports are lazy so the rest of the app (and unit tests) work without
PyTorch installed. This module never touches Streamlit; failures raise
`VoiceError` and the UI decides how to show them.
"""
from __future__ import annotations

import io
import logging
import math

import numpy as np
import streamlit as st

from dataclasses import dataclass, field

from src.services import biometrics
from src.settings import (VOICE_MATCH_MARGIN, VOICE_MATCH_THRESHOLD, VOICE_MIN_COHORT,
                          VOICE_MIN_ZSCORE, VOICE_Z_MARGIN)
from src.validation import MAX_AUDIO_SECONDS

log = logging.getLogger("smartattend.voice")
SAMPLE_RATE = 16000
MAX_SECONDS = MAX_AUDIO_SECONDS
MIN_SEGMENT_SECONDS = 0.6
# A profile needs this much detected SPEECH (silence is trimmed first). The earlier rule counted the
# whole recording, so 6 s with 1 s of speech passed while 4 s of continuous speech failed.
VOICE_PROFILE_MIN_SPEECH_SECONDS = 3.0
TURN_GAP_SECONDS = 0.7


class VoiceError(RuntimeError):
    """Voice processing failed; message is safe to show to users."""


@st.cache_resource(show_spinner="Loading voice model…")
def load_voice_encoder():
    from resemblyzer import VoiceEncoder

    return VoiceEncoder()


def _load_audio(audio_bytes: bytes) -> np.ndarray:
    import librosa
    import soundfile as sf

    try:
        # Browser recordings are WAV. Inspect the header before decoding so an
        # overlong file cannot allocate an unbounded waveform first.
        try:
            duration = sf.info(io.BytesIO(audio_bytes)).duration
        except Exception:  # other formats may be readable by librosa's fallback decoder
            duration = None
        if duration is not None and duration > MAX_SECONDS:
            raise VoiceError(f"Recording is too long (max {MAX_SECONDS // 60} minutes).")
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
    except VoiceError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Could not decode audio")
        raise VoiceError("Could not read the audio recording. Please record again.") from exc
    if audio.size == 0:
        raise VoiceError("The recording is empty.")
    return audio


def get_voice_embedding(audio_bytes: bytes) -> list[float]:
    from resemblyzer import preprocess_wav

    audio = _load_audio(audio_bytes)
    quality = assess_audio(audio)
    if quality.fatal:
        raise VoiceError(" ".join(quality.issues))
    try:
        wav = preprocess_wav(audio)
        if len(wav) < SAMPLE_RATE * VOICE_PROFILE_MIN_SPEECH_SECONDS:
            raise VoiceError(
                f"Not enough clear speech was detected ({len(wav) / SAMPLE_RATE:.1f} s). Read the whole prompt "
                f"in your normal voice: at least {VOICE_PROFILE_MIN_SPEECH_SECONDS:.0f} seconds of speaking is needed."
            )
        return load_voice_encoder().embed_utterance(wav).tolist()
    except VoiceError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Voice embedding failed")
        raise VoiceError("Could not analyse the voice recording. Try a longer, clearer sample.") from exc


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


# ------------------------------------------------------------------ audio quality
MIN_SECONDS = 0.5


@dataclass
class AudioQuality:
    duration: float
    peak: float
    clipped_fraction: float
    speech_fraction: float
    snr_db: float
    issues: list[str] = field(default_factory=list)
    fatal: bool = False


def assess_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> AudioQuality:
    """Cheap, explainable checks on a mono float waveform in [-1, 1]. No ML involved."""
    audio = np.asarray(audio, dtype=np.float64).ravel()
    dur = len(audio) / sr if sr else 0.0
    q = AudioQuality(dur, float(np.max(np.abs(audio))) if audio.size else 0.0, 0.0, 0.0, 0.0)
    if audio.size == 0 or not np.isfinite(audio).all():
        q.issues.append("The recording is empty or corrupt."); q.fatal = True
        return q
    if dur < MIN_SECONDS:
        q.issues.append(f"Recording is too short ({dur:.1f}s). Record a clear spoken response."); q.fatal = True
    if dur > MAX_SECONDS:
        q.issues.append("Recording is too long (max 10 minutes)."); q.fatal = True
    if q.peak < 0.01:
        q.issues.append("The recording is silent: check the microphone and its permissions."); q.fatal = True
        return q
    q.clipped_fraction = float(np.mean(np.abs(audio) >= 0.99))
    if q.clipped_fraction > 0.01:
        q.issues.append("Audio is clipped (too loud). Move back from the microphone.")
    frame = int(0.03 * sr)
    n = len(audio) // frame
    if n >= 5:
        rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-9
        db = 20 * np.log10(rms)
        q.speech_fraction = float(np.mean(db > db.max() - 30))
        q.snr_db = float(np.percentile(db, 90) - np.percentile(db, 10))
        if q.speech_fraction < 0.05:
            q.issues.append("Almost no speech detected."); q.fatal = True
        if q.snr_db < 10:
            q.issues.append("Background noise is high relative to speech; results may be unreliable.")
    return q


# ------------------------------------------------------------------ matching
@dataclass
class SpeakerMatch:
    status: str                      # matched | ambiguous | unknown
    student_id: int | None = None
    score: float = 0.0
    second_score: float | None = None
    candidates: list[int] = field(default_factory=list)
    zscore: float | None = None      # how far the best match stands out from all stored voices
    flags: list[str] = field(default_factory=list)  # e.g. other_class_closer, resolved_answer_once


def rank_speakers(embedding, candidates: dict[int, list[float]], threshold: float = VOICE_MATCH_THRESHOLD,
                  margin: float = VOICE_MATCH_MARGIN) -> SpeakerMatch:
    """Never returns the nearest speaker blindly: weak => unknown, too close to call => ambiguous."""
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Voice match threshold must be finite and between 0 and 1.")
    if not math.isfinite(margin) or not 0.0 <= margin <= 2.0:
        raise ValueError("Voice match margin must be finite and between 0 and 2.")
    if embedding is None or not candidates:
        return SpeakerMatch("unknown")
    scored = sorted(((cosine_similarity(embedding, e), sid) for sid, e in candidates.items() if e is not None and len(e)), reverse=True)
    if not scored:
        return SpeakerMatch("unknown")
    best, sid = scored[0]
    second = scored[1][0] if len(scored) > 1 else None
    if best < threshold:
        return SpeakerMatch("unknown", None, max(best, 0.0), second)
    close = [s for sc, s in scored if sc >= threshold and best - sc < margin]
    if len(close) > 1:
        return SpeakerMatch("ambiguous", None, best, second, close)
    return SpeakerMatch("matched", sid, best, second, [sid])


def identify_speaker(new_embedding, candidates_dict: dict[int, list[float]], threshold: float = VOICE_MATCH_THRESHOLD):
    """Back-compatible helper: (student_id | None, best_score). Ambiguous counts as None."""
    m = rank_speakers(new_embedding, candidates_dict, threshold)
    return m.student_id, m.score


def _unit_rows(vectors) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float64)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def match_turns(embeddings, candidates: dict[int, list[float]], cohort: dict[int, list[float]] | None = None,
                threshold: float = VOICE_MATCH_THRESHOLD, margin: float = VOICE_MATCH_MARGIN,
                min_z: float = VOICE_MIN_ZSCORE, z_margin: float = VOICE_Z_MARGIN,
                min_cohort: int = VOICE_MIN_COHORT) -> list[SpeakerMatch]:
    """Decide every speaking turn of one recording together (see module docstring).

    `candidates`: voice profiles of students enrolled in the subject (only they can be matched).
    `cohort`: voice profiles of ALL students (used to normalise scores and to spot a closer
    voice from another class). With fewer than `min_cohort` usable profiles, z-scores are not
    meaningful, so the raw threshold + margin rule (`rank_speakers`) is used for each turn.
    """
    for name, value, hi in (("threshold", threshold, 1.0), ("margin", margin, 2.0), ("min_z", min_z, 10.0), ("z_margin", z_margin, 5.0)):
        if not math.isfinite(value) or not 0.0 <= value <= hi:
            raise ValueError(f"Voice {name} must be finite and between 0 and {hi:g}.")
    clean = {int(k): v for k, v in candidates.items() if biometrics.is_valid_embedding(v, biometrics.VOICE_DIM)}
    pool = {int(k): v for k, v in (cohort or {}).items() if biometrics.is_valid_embedding(v, biometrics.VOICE_DIM)}
    pool.update(clean)
    turns = [np.asarray(e, dtype=np.float64) for e in embeddings if e is not None]
    if not clean or not turns:
        return [SpeakerMatch("unknown") for _ in turns]

    if len(pool) < min_cohort:  # too few voices to normalise against: per-turn raw rule
        results = [rank_speakers(e, clean, threshold, margin) for e in turns]
        _answer_once(results, lambda k, allowed: rank_speakers(turns[k], {s: clean[s] for s in allowed}, threshold, margin), set(clean))
        return results

    ids = list(pool)
    mat = _unit_rows([pool[i] for i in ids])
    enrolled = set(clean)
    scored = []
    for e in turns:
        sims = mat @ (e / (np.linalg.norm(e) or 1.0))
        order = np.argsort(-sims)
        rest = sims[order[3:]]  # leave out the closest few so a genuine match does not inflate the crowd
        z = (sims - rest.mean()) / (rest.std() + 1e-9)
        scored.append((sims, z, order))

    def decide(k: int, allowed: set[int]) -> SpeakerMatch:
        sims, z, order = scored[k]
        cand = [j for j in order if ids[j] in allowed]
        if not cand:
            return SpeakerMatch("unknown")
        b = cand[0]
        best, best_z = float(sims[b]), float(z[b])
        second = float(sims[cand[1]]) if len(cand) > 1 else None
        if best < threshold or best_z < min_z:
            return SpeakerMatch("unknown", None, max(best, 0.0), second, zscore=best_z)
        top = order[0]
        if ids[top] not in enrolled and z[top] >= best_z - z_margin:
            return SpeakerMatch("ambiguous", None, best, second, [ids[b]], best_z, ["other_class_closer"])
        close = [ids[j] for j in cand if sims[j] >= threshold and best_z - z[j] < z_margin]
        if len(close) > 1:
            return SpeakerMatch("ambiguous", None, best, second, close, best_z)
        return SpeakerMatch("matched", ids[b], best, second, [ids[b]], best_z)

    results = [decide(k, enrolled) for k in range(len(turns))]
    _answer_once(results, decide, enrolled)
    return results


def _answer_once(results: list[SpeakerMatch], decide, enrolled: set[int]) -> None:
    """Each student answers once: re-decide ambiguous turns without students already matched elsewhere."""
    changed = True
    while changed:
        changed = False
        taken = {r.student_id for r in results if r.status == "matched"}
        for k, r in enumerate(results):
            if r.status == "ambiguous" and "other_class_closer" not in r.flags:
                again = decide(k, enrolled - taken)
                if again.status == "matched":
                    again.flags.append("resolved_answer_once")
                    results[k] = again
                    taken.add(again.student_id)
                    changed = True


@dataclass
class BulkVoiceResult:
    matched: dict[int, float] = field(default_factory=dict)
    ambiguous: list[list[int]] = field(default_factory=list)  # candidate groups needing review
    other_class: list[int] = field(default_factory=list)      # enrolled students a closer outside voice contested
    best_scores: list[float] = field(default_factory=list)
    segments: int = 0
    quality: AudioQuality | None = None


def process_bulk_audio_detailed(audio_bytes: bytes, candidates: dict[int, list[float]],
                                threshold: float = VOICE_MATCH_THRESHOLD, margin: float = VOICE_MATCH_MARGIN,
                                cohort: dict[int, list[float]] | None = None, **match_kw) -> BulkVoiceResult:
    """Split a classroom recording on silence and identify each speaker segment."""
    import librosa
    from resemblyzer import preprocess_wav

    audio = _load_audio(audio_bytes)
    quality = assess_audio(audio)
    if quality.fatal:
        raise VoiceError(" ".join(quality.issues))
    encoder = load_voice_encoder()
    out = BulkVoiceResult(quality=quality)
    spans = librosa.effects.split(audio, top_db=30).tolist()
    # Keep the words in one spoken turn together. Without this, short word-sized
    # fragments produce unstable speaker embeddings and are often rejected.
    turns = []
    max_gap = int(TURN_GAP_SECONDS * SAMPLE_RATE)
    for start, end in spans:
        if turns and start - turns[-1][1] <= max_gap:
            turns[-1][1] = end
        else:
            turns.append([start, end])

    embeddings = []
    for start, end in turns:
        if (end - start) < SAMPLE_RATE * MIN_SEGMENT_SECONDS:
            continue
        try:
            embeddings.append(encoder.embed_utterance(preprocess_wav(audio[start:end])))
        except Exception:  # noqa: BLE001 - skip a bad segment, keep the rest
            log.warning("Skipping an unreadable audio segment", exc_info=True)
    out.segments = len(embeddings)
    for m in match_turns(embeddings, candidates, cohort, threshold, margin, **match_kw):
        out.best_scores.append(m.score)
        if m.status == "matched" and m.score > out.matched.get(m.student_id, -1.0):
            out.matched[m.student_id] = m.score
        elif m.status == "ambiguous":
            out.ambiguous.append(m.candidates)
            if "other_class_closer" in m.flags:
                out.other_class.extend(m.candidates)
    return out


def process_bulk_audio(audio_bytes: bytes, candidates: dict[int, list[float]], threshold: float = VOICE_MATCH_THRESHOLD) -> dict[int, float]:
    return process_bulk_audio_detailed(audio_bytes, candidates, threshold).matched
