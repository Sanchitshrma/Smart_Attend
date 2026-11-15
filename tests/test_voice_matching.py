"""Voice matching across a whole recording: normalised scores, other-class check, answer-once,
and the review table never ticking an ambiguous student. Pure numpy (no PyTorch needed),
except the last test, which runs the real Resemblyzer model when it is installed."""
from pathlib import Path

import numpy as np
import pytest

from src.pipelines.voice_pipeline import BulkVoiceResult, match_turns
from src.services.attendance import voice_review_inputs

rng = np.random.default_rng(5)
ROOT = Path(__file__).resolve().parents[1]


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def crowd(n, base=None, spread=1.0):
    """Unit vectors that share a common component, like real Resemblyzer embeddings (all
    voices score ~0.6 against each other, so raw similarity alone separates students badly)."""
    base = unit(rng.normal(size=256)) if base is None else base
    return base, [unit(base * 1.2 + spread * unit(rng.normal(size=256))) for _ in range(n)]


def near(v, noise=0.15):
    return unit(np.asarray(v) + noise * unit(rng.normal(size=256)))


def test_a_clear_voice_is_matched_and_a_stranger_is_not():
    _, voices = crowd(20)
    gallery = {i: v.tolist() for i, v in enumerate(voices)}
    enrolled = {i: gallery[i] for i in range(8)}
    base_sim = float(np.mean([voices[0] @ voices[j] for j in range(1, 20)]))
    assert base_sim > 0.55  # different people already exceed the old 0.55 threshold
    res = match_turns([near(voices[3])], enrolled, gallery)
    assert (res[0].status, res[0].student_id) == ("matched", 3) and res[0].zscore >= 2.5
    stranger = crowd(1, base=None)[1][0]  # nobody in the database
    assert match_turns([stranger], enrolled, gallery)[0].status == "unknown"


def test_a_closer_voice_from_another_class_goes_to_review():
    _, voices = crowd(20)
    gallery = {i: v.tolist() for i, v in enumerate(voices)}
    gallery[99] = near(voices[2], 0.05).tolist()          # outsider who sounds like student 2
    enrolled = {i: gallery[i] for i in range(8)}
    res = match_turns([near(gallery[99], 0.02)], enrolled, gallery)[0]
    assert res.status == "ambiguous" and "other_class_closer" in res.flags and res.candidates == [2]


def test_each_student_answers_once_resolves_a_lookalike():
    _, voices = crowd(20)
    gallery = {i: v.tolist() for i, v in enumerate(voices)}
    gallery[1] = near(voices[0], 0.30).tolist()           # students 0 and 1 sound alike
    enrolled = {i: gallery[i] for i in range(8)}
    clear_zero = np.asarray(gallery[0])                   # student 0 answers clearly
    between = unit(np.asarray(gallery[0]) + np.asarray(gallery[1]))  # could be either
    res = match_turns([clear_zero, between], enrolled, gallery)
    assert (res[0].status, res[0].student_id) == ("matched", 0)
    assert (res[1].status, res[1].student_id) == ("matched", 1) and "resolved_answer_once" in res[1].flags


def test_small_cohort_falls_back_to_the_raw_rule_and_bad_profiles_are_ignored():
    _, voices = crowd(3, spread=3.0)
    enrolled = {0: voices[0].tolist(), 1: voices[1].tolist(), 2: voices[2].tolist(), 3: [float("nan")] * 256, 4: [1.0, 2.0]}
    res = match_turns([near(voices[1], 0.05), unit(rng.normal(size=256))], enrolled, cohort=None)
    assert (res[0].status, res[0].student_id) == ("matched", 1) and res[0].zscore is None
    assert res[1].status == "unknown"
    assert match_turns([], enrolled) == [] and match_turns([voices[0]], {})[0].status == "unknown"
    with pytest.raises(ValueError):
        match_turns([voices[0]], enrolled, min_z=float("nan"))


def test_review_table_never_ticks_an_ambiguous_student():
    enrolled = [{"student_id": i, "name": f"S{i}", "voice_embedding": [0.1] * 256} for i in range(1, 5)]
    enrolled.append({"student_id": 5, "name": "S5", "voice_embedding": None})
    result = BulkVoiceResult(matched={1: 0.83}, ambiguous=[[2, 3], [1, 4]], other_class=[4])
    present, scores, notes = voice_review_inputs(enrolled, result)
    assert set(present) == {1} and scores == {1: 0.83}          # 2, 3, 4 are NOT ticked
    assert notes[2] == notes[3] == "Possible voice match; please verify"
    assert "another class" in notes[4] and "No voice profile" in notes[5] and 1 not in notes


def test_real_model_profile_needs_three_seconds_of_speech_and_matches_its_owner():
    pytest.importorskip("resemblyzer")
    librosa = pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    import io
    from src.pipelines.voice_pipeline import VoiceError, get_voice_embedding, rank_speakers

    def wav(a):
        b = io.BytesIO(); sf.write(b, a, 16000, format="WAV"); return b.getvalue()

    clips = [librosa.load(ROOT / "class_data" / "voices" / f"student_{k:02d}.mp3", sr=16000)[0] for k in (1, 2, 3)]
    with pytest.raises(VoiceError, match="Not enough clear speech"):
        get_voice_embedding(wav(clips[0]))                       # one ~3.5 s clip has ~2.6 s of speech
    gap = np.zeros(8000)
    profiles = {k: get_voice_embedding(wav(np.concatenate([c, gap, c * 0.8]))) for k, c in enumerate(clips)}
    assert all(len(p) == 256 for p in profiles.values())
    probe = get_voice_embedding(wav(np.concatenate([clips[1] * 0.6, gap, clips[1]])))
    assert rank_speakers(probe, profiles).student_id == 1
