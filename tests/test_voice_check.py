"""Voice check for students the camera missed: 1:1 verification that catches proxy answers.
Pure numpy except the last test, which runs the real Resemblyzer model when it is installed."""
from pathlib import Path

import numpy as np
import pytest

from src.pipelines.voice_pipeline import verify_speaker

rng = np.random.default_rng(5)
ROOT = Path(__file__).resolve().parents[1]


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def crowd(n):
    """Voices sharing a common component, like real Resemblyzer embeddings (~0.6 between people)."""
    base = unit(rng.normal(size=256))
    return [unit(base * 1.2 + unit(rng.normal(size=256))) for _ in range(n)]


def near(v, noise=0.15):
    return unit(np.asarray(v) + noise * unit(rng.normal(size=256)))


def test_the_called_student_answering_is_verified():
    voices = crowd(20)
    profiles = {i: v.tolist() for i, v in enumerate(voices)}
    assert float(voices[0] @ voices[1]) > 0.55          # different people still sound "similar" in raw terms
    r = verify_speaker(near(voices[4]), 4, profiles)
    assert r.status == "verified" and r.zscore >= 2.5 and r.sounds_like is None


def test_a_friend_answering_for_an_absent_student_is_rejected_and_named():
    voices = crowd(20)
    profiles = {i: v.tolist() for i, v in enumerate(voices)}
    r = verify_speaker(near(voices[7]), 4, profiles)    # teacher called 4, student 7 answered
    assert r.status == "rejected" and r.sounds_like == 7


def test_a_stranger_or_missing_profile_is_never_verified():
    voices = crowd(20)
    profiles = {i: v.tolist() for i, v in enumerate(voices)}
    stranger = crowd(1)[0]
    assert verify_speaker(stranger, 4, profiles).status in ("rejected", "unsure")
    assert verify_speaker(near(voices[4]), 99, profiles).status == "no_profile"
    profiles[5] = [float("nan")] * 256
    assert verify_speaker(near(voices[4]), 5, profiles).status == "no_profile"


def test_small_voice_gallery_uses_the_raw_rule():
    voices = [unit(rng.normal(size=256)) for _ in range(3)]
    profiles = {i: v.tolist() for i, v in enumerate(voices)}
    assert verify_speaker(near(voices[1], 0.05), 1, profiles).status == "verified"
    proxy = verify_speaker(near(voices[2], 0.05), 1, profiles)
    assert proxy.status == "rejected" and proxy.sounds_like == 2


def test_real_model_profiles_answers_and_proxy_detection():
    pytest.importorskip("resemblyzer")
    librosa = pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    import io
    from src.pipelines.voice_pipeline import VoiceError, get_answer_embedding, get_voice_embedding

    def wav(a):
        b = io.BytesIO(); sf.write(b, a, 16000, format="WAV"); return b.getvalue()

    clips = [librosa.load(ROOT / "class_data" / "voices" / f"student_{k:02d}.mp3", sr=16000)[0] for k in (1, 2, 3)]
    with pytest.raises(VoiceError, match="Not enough clear speech"):
        get_voice_embedding(wav(clips[0]))                      # one ~3.5 s clip has ~2.6 s of speech
    with pytest.raises(VoiceError, match="too short"):
        get_answer_embedding(wav(clips[0][:16000]))              # a one-second mumble
    gap = np.zeros(8000)
    profiles = {k: get_voice_embedding(wav(np.concatenate([c, gap, c * 0.8]))) for k, c in enumerate(clips)}
    answer = get_answer_embedding(wav(clips[1] * 0.6))
    assert verify_speaker(answer, 1, profiles).status == "verified"
    proxy = verify_speaker(answer, 0, profiles)                  # student 1 answers for student 0
    assert proxy.status == "rejected" and proxy.sounds_like == 1
