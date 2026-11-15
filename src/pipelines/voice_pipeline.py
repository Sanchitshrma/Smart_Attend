"""Voice pipeline: voice profiles and the voice check for students the camera missed.

The camera marks most of the class. For the few students it could not recognise, the teacher calls
only those names, and each answer is checked 1:1 ("is this really Rohan?") with `verify_speaker`:
  * the answer's Resemblyzer d-vector (256-d) is compared with EVERY stored voice profile,
  * scores are normalised against that crowd (z-score), because raw cosine similarity between
    different voices is high (typically ~0.58),
  * if another student's voice is clearly closer, the answer is rejected and that student is
    named (a friend answering for someone else); unclear answers are left to the teacher.

Heavy ML imports are lazy so the rest of the app (and unit tests) work without PyTorch installed.
This module never touches Streamlit; failures raise `VoiceError` and the UI decides how to show them.
"""
from __future__ import annotations

import io
import logging
import math

import numpy as np
import streamlit as st

from dataclasses import dataclass, field

from src.services import biometrics
from src.settings import (VOICE_ANSWER_MIN_SPEECH_SECONDS, VOICE_MATCH_MARGIN, VOICE_MATCH_THRESHOLD,
                          VOICE_MIN_COHORT, VOICE_MIN_ZSCORE, VOICE_Z_MARGIN)
from src.validation import MAX_AUDIO_SECONDS

log = logging.getLogger("smartattend.voice")
SAMPLE_RATE = 16000
MAX_SECONDS = MAX_AUDIO_SECONDS
# A profile needs this much detected SPEECH (silence is trimmed first). The earlier rule counted the
# whole recording, so 6 s with 1 s of speech passed while 4 s of continuous speech failed.
VOICE_PROFILE_MIN_SPEECH_SECONDS = 3.0


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
    if len(audio) > GROUP_MAX_SECONDS * SAMPLE_RATE:
        raise VoiceError(f"The recording is too long (max {GROUP_MAX_SECONDS // 60} minutes). Record smaller groups.")
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


def get_answer_embedding(audio_bytes: bytes) -> list[float]:
    """Embed one student's spoken answer during a voice check (shorter than a profile recording)."""
    from resemblyzer import preprocess_wav

    audio = _load_audio(audio_bytes)
    if len(audio) > GROUP_MAX_SECONDS * SAMPLE_RATE:
        raise VoiceError(f"The recording is too long (max {GROUP_MAX_SECONDS // 60} minutes). Record smaller groups.")
    quality = assess_audio(audio)
    if quality.fatal:
        raise VoiceError(" ".join(quality.issues))
    try:
        wav = preprocess_wav(audio)
        if len(wav) < SAMPLE_RATE * VOICE_ANSWER_MIN_SPEECH_SECONDS:
            raise VoiceError("The answer was too short. Ask the student to say the full sentence, "
                             "e.g. “Present, ma’am. I am here in the class today.”")
        return load_voice_encoder().embed_utterance(wav).tolist()
    except VoiceError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Voice embedding failed")
        raise VoiceError("Could not analyse the answer. Record it again in a quieter moment.") from exc


@dataclass
class VoiceCheck:
    """Result of checking that a spoken answer really comes from the student whose name was called."""
    status: str                      # verified | rejected | unsure | no_profile
    score: float = 0.0               # cosine similarity with the called student's profile
    zscore: float | None = None      # how far that score stands out from all stored voices
    sounds_like: int | None = None   # another student whose voice is clearly closer (likely proxy)


def verify_speaker(embedding, claimed_id: int, profiles: dict[int, list[float]],
                   threshold: float = VOICE_MATCH_THRESHOLD, min_z: float = VOICE_MIN_ZSCORE,
                   z_margin: float = VOICE_Z_MARGIN, margin: float = VOICE_MATCH_MARGIN,
                   min_cohort: int = VOICE_MIN_COHORT) -> VoiceCheck:
    """1:1 check: is this answer from `claimed_id`? `profiles` holds every stored voice (the cohort).

    Verifying one claimed identity is a much easier question than picking a speaker out of a whole
    class. If another student's voice is clearly closer, the answer is rejected and that student is
    reported in `sounds_like` (a friend answering for someone else).
    """
    clean = {int(k): v for k, v in profiles.items() if biometrics.is_valid_embedding(v, biometrics.VOICE_DIM)}
    if int(claimed_id) not in clean:
        return VoiceCheck("no_profile")
    e = np.asarray(embedding, dtype=np.float64)
    ids = list(clean)
    sims = _unit_rows([clean[i] for i in ids]) @ (e / (np.linalg.norm(e) or 1.0))
    c = ids.index(int(claimed_id))
    score = float(sims[c])
    others = [k for k in range(len(ids)) if k != c]
    rival = max(others, key=lambda k: sims[k]) if others else None

    if len(ids) < min_cohort:  # too few voices to normalise against: raw rule
        if rival is not None and sims[rival] > score + margin:
            return VoiceCheck("rejected", score, None, ids[rival])
        if score >= threshold:
            return VoiceCheck("verified", score)
        return VoiceCheck("rejected" if score < threshold - 0.1 else "unsure", score)

    order = np.argsort(-sims)
    rest = sims[order[3:]]
    z = (sims - rest.mean()) / (rest.std() + 1e-9)
    zc = float(z[c])
    if rival is not None and z[rival] - zc >= z_margin:
        return VoiceCheck("rejected", score, zc, ids[rival])
    if score >= threshold and zc >= min_z and (rival is None or zc - z[rival] >= z_margin):
        return VoiceCheck("verified", score, zc)
    if score < threshold - 0.1 or zc < min_z - 1.5:
        return VoiceCheck("rejected", score, zc)
    return VoiceCheck("unsure", score, zc)


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


# ============================================================ one recording for several students
# The teacher calls the names of the students the camera missed, one after another, and records
# all the answers in ONE recording. The recording is split into speakers (speaker diarization) and
# each speaker is checked only against the called students (a small group, so this is far easier
# than identifying a whole class). A voice that belongs to someone else is reported as a proxy.

VAD_FRAME_MS = 30
VAD_FILL_SECONDS = 0.2           # pauses shorter than this stay inside one speech segment
MIN_SPEECH_SECONDS = 0.5
CHANGE_WINDOW_SECONDS = 1.2      # window on each side when looking for a change of speaker
CHANGE_STEP_SECONDS = 0.25
CHANGE_SIMILARITY = 0.55         # left/right windows less similar than this => new speaker
SAME_SPEAKER_SIMILARITY = 0.85   # segments at least this similar are merged into one speaker
MIN_TURN_SECONDS = 0.8
GROUP_MAX_SECONDS = 180
TEACHER_MAX_TURN_SECONDS = 1.5   # reading a name is short; a student's answer is longer
GROUP_MIN_SPEAKER_SECONDS = 1.0     # shorter speakers (e.g. the teacher reading a name) are not judged


@dataclass
class Speaker:
    segments: list[tuple[int, int]]
    embedding: np.ndarray = field(repr=False)

    @property
    def seconds(self) -> float:
        return sum(e - s for s, e in self.segments) / SAMPLE_RATE


def speech_segments(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list[tuple[int, int]]:
    """Voice activity detection (WebRTC VAD), with short pauses filled and tiny bursts dropped."""
    try:
        import webrtcvad
    except ImportError:  # energy fallback
        import librosa
        spans = librosa.effects.split(audio, top_db=30).tolist()
    else:
        vad = webrtcvad.Vad(2)
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        n = int(sr * VAD_FRAME_MS / 1000)
        spans = []
        for i in range(len(audio) // n):
            if vad.is_speech(pcm[2 * i * n:2 * (i + 1) * n], sr):
                s, e = i * n, (i + 1) * n
                if spans and s - spans[-1][1] <= 0:
                    spans[-1][1] = e
                else:
                    spans.append([s, e])
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= VAD_FILL_SECONDS * sr:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= MIN_SPEECH_SECONDS * sr]


def _embed(encoder, audio: np.ndarray) -> np.ndarray:
    v = encoder.embed_utterance(audio)
    return v / (np.linalg.norm(v) or 1.0)


def split_on_speaker_change(audio: np.ndarray, segments, encoder) -> list[tuple[int, int]]:
    """Split segments where the voice changes, even if two students left almost no pause."""
    w, step, min_len = (int(x * SAMPLE_RATE) for x in (CHANGE_WINDOW_SECONDS, CHANGE_STEP_SECONDS, MIN_TURN_SECONDS))
    out = []
    todo = list(segments)
    while todo:
        s, e = todo.pop(0)
        if e - s < 2 * max(w, min_len):
            out.append((s, e))
            continue
        best_t, best_sim = None, 1.0
        for t in range(s + max(w, min_len), e - max(w, min_len) + 1, step):
            sim = float(_embed(encoder, audio[t - w:t]) @ _embed(encoder, audio[t:t + w]))
            if sim < best_sim:
                best_t, best_sim = t, sim
        if best_t is not None and best_sim < CHANGE_SIMILARITY:
            todo[:0] = [(s, best_t), (best_t, e)]   # look for more changes inside each half
        else:
            out.append((s, e))
    return sorted(out)


def cluster_speakers(audio: np.ndarray, segments, encoder) -> list[Speaker]:
    """Merge segments spoken by the same person (agglomerative, average linkage on cosine)."""
    groups = [[seg] for seg in segments]
    embs = [_embed(encoder, audio[s:e]) for s, e in segments]
    while len(groups) > 1:
        best = (-1.0, None, None)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                sim = float(embs[i] @ embs[j])
                if sim > best[0]:
                    best = (sim, i, j)
        sim, i, j = best
        if sim < SAME_SPEAKER_SIMILARITY:
            break
        groups[i] += groups.pop(j)
        embs.pop(j)
        embs[i] = _embed(encoder, np.concatenate([audio[s:e] for s, e in groups[i]]))
    return [Speaker(sorted(g), emb) for g, emb in zip(groups, embs)]


def diarize(audio: np.ndarray, encoder=None) -> list[Speaker]:
    """Who spoke when: voice activity -> split on speaker changes -> group by speaker."""
    from resemblyzer.audio import normalize_volume

    encoder = encoder or load_voice_encoder()
    norm = np.clip(normalize_volume(audio.astype(np.float32), -30, increase_only=True), -1, 1)
    segments = split_on_speaker_change(norm, speech_segments(norm), encoder)
    return cluster_speakers(norm, segments, encoder)


@dataclass
class GroupCheck:
    """Outcome of one recording for a group of called students."""
    speakers: int = 0
    verified: dict[int, float] = field(default_factory=dict)    # student_id -> similarity
    unsure: dict[int, str] = field(default_factory=dict)        # student_id -> note for the teacher
    proxies: list[int] = field(default_factory=list)            # students whose voice answered although not called
    unknown_voices: int = 0                                      # voices matching nobody
    quality: AudioQuality | None = None


def teacher_index(speakers: list[Speaker]) -> int | None:
    """Find the teacher's voice: it reads every name, so it comes back many times in short bursts,
    while each student answers once. Prefer the speaker with the most short turns (at least 2);
    otherwise the voice heard first, if it comes back again."""
    if len(speakers) < 2:
        return None
    short = [i for i, sp in enumerate(speakers)
             if len(sp.segments) >= 2 and sp.seconds / len(sp.segments) <= TEACHER_MAX_TURN_SECONDS]
    if short:
        return max(short, key=lambda i: (len(speakers[i].segments), -speakers[i].segments[0][0]))
    k = min(range(len(speakers)), key=lambda i: speakers[i].segments[0][0] if speakers[i].segments else 1 << 62)
    return k if len(speakers[k].segments) >= 2 else None


def verify_group(speakers: list[Speaker], called: list[int], profiles: dict[int, list[float]],
                 min_seconds: float = GROUP_MIN_SPEAKER_SECONDS) -> GroupCheck:
    """Match each speaker to at most one called student; report voices that belong to others.

    Each (speaker, called student) pair uses the same 1:1 rule as `verify_speaker`, so a voice is
    accepted only if it is close to that student AND clearly closer to them than to anyone else in
    `profiles` (pass every stored voice so the comparison crowd is large). Very short speakers, such
    as the teacher reading a name, are skipped because a second of speech is not enough to judge, and
    so is the teacher's own voice (see `teacher_index`).
    """
    called = list(dict.fromkeys(int(c) for c in called))
    out = GroupCheck(speakers=len(speakers))
    if not speakers or not called:
        return out
    claims: list[tuple[float, int, int]] = []   # (score, speaker index, student)
    teacher = teacher_index(speakers)
    for k, sp in enumerate(speakers):
        if sp.seconds < min_seconds or k == teacher:
            continue
        checks = {sid: verify_speaker(sp.embedding, sid, profiles) for sid in called}
        ok = [(c.score, sid) for sid, c in checks.items() if c.status == "verified"]
        if ok:
            score, sid = max(ok)
            claims.append((float(score), k, sid))
            continue
        scored = [(c.score or 0.0, sid, c) for sid, c in checks.items() if c.status != "no_profile"]
        if not scored:
            continue
        score, sid, best = max(scored, key=lambda x: x[0])
        if best.sounds_like is not None and best.sounds_like not in called:
            if best.sounds_like not in out.proxies:
                out.proxies.append(best.sounds_like)   # sounds like someone who was not called
        elif best.status == "unsure":
            out.unsure.setdefault(sid, f"A voice was close to this student ({score:.2f}) but not clear enough")
        else:
            out.unknown_voices += 1
    for score, k, sid in sorted(claims, reverse=True):   # each student is verified by one voice only
        if sid in out.verified:
            continue
        out.verified[sid] = score
        out.unsure.pop(sid, None)
    return out


def check_group_recording(audio_bytes: bytes, called: list[int], profiles: dict[int, list[float]]) -> GroupCheck:
    """Decode one recording, run quality checks, diarize, and verify the called students."""
    audio = _load_audio(audio_bytes)
    if len(audio) > GROUP_MAX_SECONDS * SAMPLE_RATE:
        raise VoiceError(f"The recording is too long (max {GROUP_MAX_SECONDS // 60} minutes). Record smaller groups.")
    quality = assess_audio(audio)
    if quality.fatal:
        raise VoiceError(" ".join(quality.issues))
    try:
        speakers = diarize(audio)
    except VoiceError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Diarization failed")
        raise VoiceError("Could not analyse the recording. Please record again in a quieter moment.") from exc
    if not speakers:
        raise VoiceError("No clear speech was detected. Ask each student to answer clearly, one after another.")
    result = verify_group(speakers, called, profiles)
    result.quality = quality
    return result
