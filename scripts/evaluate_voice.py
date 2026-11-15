"""Evaluate bulk voice attendance on simulated classroom recordings.

    python scripts/evaluate_voice.py                 # 40 simulated classes, seed 99
    python scripts/evaluate_voice.py --classes 20 --seed 7 --out voice_report.json

How a class is simulated (from the synthetic clips in class_data/voices, one clip per voice):
  * duplicate synthetic voices (cosine >= 0.95) are dropped, 6 voices are held out as people who
    are not in the database at all, the rest form the gallery of stored voice profiles;
  * a profile is enrolled from two "takes" of the student's clip (slightly slower / faster);
  * each class has 15 enrolled students; 8 of them answer, plus one student from another class
    and one unknown person, in random order, each as a new "take" (speed, pitch, level and room
    echo changed), separated by 0.35-1.2 s pauses, with background noise at 15-25 dB SNR.

It reports the CURRENT matcher (match_turns with the configured settings) against the ORIGINAL
rule (raw cosine >= 0.55, margin 0.05, ambiguous candidates ticked present). These are synthetic
text-to-speech voices: use the numbers to compare rules, not as real-world accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.voice_pipeline import SAMPLE_RATE as SR, load_voice_encoder, match_turns, rank_speakers  # noqa: E402


def _take(a, rng, rate=1.0, semis=0.0, gain=1.0, rt=0.0):
    import librosa
    b = librosa.effects.time_stretch(a, rate=rate) if rate != 1.0 else a.copy()
    if semis:
        b = librosa.effects.pitch_shift(b, sr=SR, n_steps=semis)
    if rt:
        n = int(rt * SR); ir = np.exp(-np.linspace(0, 6, n)) * rng.normal(size=n) * 0.3; ir[0] = 1.0
        peak = np.max(np.abs(b)); b = np.convolve(b, ir)[: len(b)]; b = b / (np.max(np.abs(b)) + 1e-9) * peak
    return b * gain


def _noise(n, level, rng):
    w = np.cumsum(rng.normal(size=n)); w -= np.convolve(w, np.ones(400) / 400, mode="same")
    return w / (np.max(np.abs(w)) + 1e-9) * level


def _turns(audio):
    """Same segmentation as process_bulk_audio_detailed."""
    import librosa
    from src.pipelines.voice_pipeline import MIN_SEGMENT_SECONDS, TURN_GAP_SECONDS
    out = []
    for s, e in librosa.effects.split(audio, top_db=30).tolist():
        if out and s - out[-1][1] <= int(TURN_GAP_SECONDS * SR):
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e - s >= MIN_SEGMENT_SECONDS * SR]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out")
    args = ap.parse_args()

    import librosa
    from resemblyzer import preprocess_wav

    rng = np.random.default_rng(args.seed)
    enc = load_voice_encoder()
    rows = [r for r in csv.DictReader(open(ROOT / "class_data" / "students_with_voice.csv", encoding="utf-8-sig"))
            if r.get("voice_audio") and (ROOT / "class_data" / "voices" / r["voice_audio"]).exists()]
    clips = [librosa.load(ROOT / "class_data" / "voices" / r["voice_audio"], sr=SR)[0] for r in rows]
    raw = np.array([enc.embed_utterance(preprocess_wav(c)) for c in clips])
    voices = []
    for i in range(len(clips)):  # drop duplicate synthetic voices
        if all(raw[i] @ raw[j] < 0.95 for j in voices):
            voices.append(i)
    holdout = [int(x) for x in rng.choice(voices, 6, replace=False)]
    gallery = [v for v in voices if v not in holdout]
    prof = {v: enc.embed_utterance(preprocess_wav(np.concatenate(
        [_take(clips[v], rng, 0.97), np.zeros(SR // 2), _take(clips[v], rng, 1.03, gain=0.8)]))) for v in gallery}

    totals = {k: dict(correct=0, wrong=0, review_turns=0, missed=0) for k in ("original", "current")}
    present_total = absent_total = 0
    for _ in range(args.classes):
        enrolled = [int(x) for x in rng.choice(gallery, 15, replace=False)]
        present = enrolled[:8]
        outsider = int(rng.choice([g for g in gallery if g not in enrolled])); unknown = int(rng.choice(holdout))
        speakers = present + [outsider, unknown]; rng.shuffle(speakers)
        parts = [np.zeros(int(0.6 * SR))]; probes = []
        for v in speakers:
            p = _take(clips[v], rng, float(rng.uniform(0.92, 1.08)), float(rng.uniform(-0.6, 0.6)),
                      float(rng.uniform(0.25, 1.0)), float(rng.uniform(0.15, 0.35)))
            probes.append(p); parts += [p, np.zeros(int(rng.uniform(0.35, 1.2) * SR))]
        audio = np.concatenate(parts); rms = np.sqrt(np.mean(np.concatenate(probes) ** 2))
        audio = np.clip(audio + _noise(len(audio), rms / 10 ** (rng.uniform(15, 25) / 20) * 1.5, rng), -1, 1)
        emb = [enc.embed_utterance(preprocess_wav(audio[s:e])) for s, e in _turns(audio)]
        cand = {v: prof[v].tolist() for v in enrolled}
        cohort = {v: prof[v].tolist() for v in gallery}
        pres = set(present); present_total += len(pres); absent_total += len(enrolled) - len(pres)

        old = [rank_speakers(e, cand, 0.55, 0.05) for e in emb]
        old_ticked = {m.student_id for m in old if m.status == "matched"} | {s for m in old if m.status == "ambiguous" for s in m.candidates}
        new = match_turns(emb, cand, cohort)
        new_ticked = {m.student_id for m in new if m.status == "matched"}
        for key, res, ticked in (("original", old, old_ticked), ("current", new, new_ticked)):
            t = totals[key]
            t["correct"] += len(ticked & pres); t["wrong"] += len(ticked - pres); t["missed"] += len(pres - ticked)
            t["review_turns"] += sum(m.status == "ambiguous" for m in res)

    report = {"classes": args.classes, "seed": args.seed, "present_students": present_total,
              "absent_enrolled_students": absent_total, "outsider_turns": 2 * args.classes,
              "ticked_present_if_teacher_confirms_without_changes": totals,
              "note": "Synthetic text-to-speech voices; compare rules, do not read as real-world accuracy."}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
