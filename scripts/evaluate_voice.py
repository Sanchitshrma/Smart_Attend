"""Evaluate the voice check (1:1 verification of students the camera missed) on simulated answers.

    python scripts/evaluate_voice.py                 # seed 99
    python scripts/evaluate_voice.py --rounds 3 --seed 7 --out voice_report.json

Simulation (from the synthetic clips in class_data/voices, one clip per voice):
  * duplicate synthetic voices (cosine >= 0.95) are dropped, 6 voices are held out as people who
    are not in the database at all, the rest form the gallery of stored voice profiles;
  * a profile is enrolled from two "takes" of the student's clip (slightly slower / faster);
  * each answer is a new take (speed, pitch, level, room echo, background noise 15-25 dB SNR).
Three kinds of claim are checked with verify_speaker:
  genuine  - the called student answers,
  proxy    - another student answers for the called (absent) student,
  unknown  - someone not in the database answers.
These are synthetic text-to-speech voices: the numbers show how the rule behaves, not real accuracy.
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

from src.pipelines.voice_pipeline import SAMPLE_RATE as SR, load_voice_encoder, verify_speaker  # noqa: E402


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4, help="answers simulated per gallery voice")
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
    profiles = {v: enc.embed_utterance(preprocess_wav(np.concatenate(
        [_take(clips[v], rng, 0.97), np.zeros(SR // 2), _take(clips[v], rng, 1.03, gain=0.8)]))).tolist() for v in gallery}

    def answer(v):
        a = _take(clips[v], rng, float(rng.uniform(0.92, 1.08)), float(rng.uniform(-0.6, 0.6)),
                  float(rng.uniform(0.25, 1.0)), float(rng.uniform(0.15, 0.35)))
        rms = np.sqrt(np.mean(a ** 2))
        a = np.clip(a + _noise(len(a), rms / 10 ** (rng.uniform(15, 25) / 20) * 1.5, rng), -1, 1)
        return enc.embed_utterance(preprocess_wav(a))

    counts = {k: {"verified": 0, "unsure": 0, "rejected": 0, "named_real_speaker": 0, "n": 0} for k in ("genuine", "proxy", "unknown")}
    for _ in range(args.rounds):
        for v in gallery:
            e = answer(v)
            others = [g for g in gallery if g != v]
            for kind, claim in (("genuine", v), ("proxy", int(rng.choice(others)))):
                r = verify_speaker(e, claim, profiles)
                c = counts[kind]; c["n"] += 1; c[r.status] += 1
                c["named_real_speaker"] += int(kind == "proxy" and r.sounds_like == v)
        for u in holdout:
            r = verify_speaker(answer(u), int(rng.choice(gallery)), profiles)
            c = counts["unknown"]; c["n"] += 1; c[r.status] += 1

    report = {"seed": args.seed, "gallery_voices": len(gallery), "held_out_voices": len(holdout), "claims": counts,
              "note": "Synthetic text-to-speech voices; shows how the rule behaves, not real-world accuracy."}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
