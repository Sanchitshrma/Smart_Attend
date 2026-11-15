"""One recording for several students: split the recording into speakers, then verify each called
student. Pure numpy except the last test, which runs the real Resemblyzer model when installed."""
from pathlib import Path

import numpy as np
import pytest

from src.pipelines.voice_pipeline import SAMPLE_RATE, Speaker, verify_group

rng = np.random.default_rng(9)
ROOT = Path(__file__).resolve().parents[1]


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def crowd(n):
    base = unit(rng.normal(size=256))
    return [unit(base * 1.2 + unit(rng.normal(size=256))) for _ in range(n)]


def near(v, noise=0.15):
    return unit(np.asarray(v) + noise * unit(rng.normal(size=256)))


def speaker(emb, seconds=2.0):
    return Speaker([(0, int(seconds * SAMPLE_RATE))], np.asarray(emb))


def test_called_students_are_verified_and_a_proxy_voice_is_named():
    voices = crowd(20)
    profiles = {i: v.tolist() for i, v in enumerate(voices)}
    teacher = crowd(1)[0]
    speakers = [speaker(teacher, 0.7), speaker(near(voices[1])), speaker(teacher, 0.6),
                speaker(near(voices[2])), speaker(near(voices[9]))]          # 9 was not called
    r = verify_group(speakers, [1, 2, 3, 4], profiles)
    assert set(r.verified) == {1, 2}
    assert r.proxies == [9]
    assert 3 not in r.verified and 4 not in r.verified
    assert r.speakers == 5


def test_a_student_is_verified_by_one_voice_only_and_strangers_are_not_matched():
    voices = crowd(20)
    profiles = {i: v.tolist() for i, v in enumerate(voices)}
    r = verify_group([speaker(near(voices[3])), speaker(near(voices[3])), speaker(crowd(1)[0])], [3, 5], profiles)
    assert list(r.verified) == [3]
    assert 5 not in r.verified


def test_short_speakers_and_missing_profiles_are_skipped():
    voices = crowd(12)
    profiles = {i: v.tolist() for i, v in enumerate(voices)}
    assert verify_group([speaker(near(voices[2]), 0.5)], [2], profiles).verified == {}
    assert verify_group([speaker(near(voices[2]))], [99], profiles).verified == {}
    assert verify_group([], [2], profiles).verified == {}
    assert verify_group([speaker(near(voices[2]))], [], profiles).verified == {}


def test_real_model_splits_one_recording_into_students():
    pytest.importorskip("resemblyzer")
    pytest.importorskip("webrtcvad")
    librosa = pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    import io
    from src.pipelines.voice_pipeline import VoiceError, check_group_recording, get_voice_embedding

    def wav(a):
        b = io.BytesIO(); sf.write(b, a, 16000, format="WAV"); return b.getvalue()

    ks = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    clips = {k: librosa.load(ROOT / "class_data" / "voices" / f"student_{k:02d}.mp3", sr=16000)[0] for k in ks}
    gap = np.zeros(8000)
    profiles = {k: get_voice_embedding(wav(np.concatenate([c, gap, c * 0.8]))) for k, c in clips.items()}
    pause = lambda s: np.zeros(int(s * 16000))  # noqa: E731
    # The teacher called 1, 2 and 3. Students 1 and 2 answer; student 4 answers instead of 3.
    rec = np.concatenate([pause(0.5), clips[1] * 0.7, pause(0.6), clips[2] * 0.5, pause(0.4), clips[4] * 0.8, pause(0.5)])
    r = check_group_recording(wav(rec), [1, 2, 3], profiles)
    assert set(r.verified) == {1, 2}
    assert r.proxies == [4]
    assert r.speakers >= 3
    with pytest.raises(VoiceError):
        check_group_recording(wav(np.zeros(16000 * 3)), [1], profiles)


def test_the_teachers_voice_reading_names_is_never_judged():
    voices = crowd(20)
    profiles = {i: v.tolist() for i, v in enumerate(voices)}
    sr = SAMPLE_RATE
    # The teacher happens to sound like student 6 and reads names at 0 s, 4 s and 8 s.
    teacher = Speaker([(0, sr), (4 * sr, 5 * sr), (8 * sr, 9 * sr)], near(voices[6]))
    answer = Speaker([(2 * sr, 4 * sr)], near(voices[1]))
    r = verify_group([teacher, answer], [1, 2], profiles)
    assert r.proxies == [] and set(r.verified) == {1}


def test_the_teacher_is_found_even_if_the_first_name_was_heard_separately():
    voices = crowd(20)
    profiles = {i: v.tolist() for i, v in enumerate(voices)}
    sr = SAMPLE_RATE
    first_call = Speaker([(0, int(0.8 * sr))], near(voices[6]))                  # too short to judge anyway
    teacher = Speaker([(5 * sr, 6 * sr), (10 * sr, 11 * sr), (15 * sr, 16 * sr)], near(voices[6]))
    answer = Speaker([(2 * sr, 4 * sr)], near(voices[1]))
    r = verify_group([first_call, answer, teacher], [1, 2], profiles)
    assert r.proxies == [] and set(r.verified) == {1}
