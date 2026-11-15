"""Evaluate the ONE-recording voice check (speaker separation + verification) on simulated classes.

    python scripts/evaluate_voice_group.py               # 60 simulated recordings, seed 99
    python scripts/evaluate_voice_group.py --trials 40 --seed 5

Each simulated recording: 15 students enrolled, the teacher calls 5 names the camera missed.
3 of them answer, 1 is absent and a classmate (already marked present) answers instead (proxy),
1 is absent and nobody answers. Before every answer the teacher reads the name (a voice not in the
database). Answers are 1.6-2.6 s new "takes" of the student's clip (speed, pitch, level, room echo),
pauses between speakers are only 0.1-0.9 s, and background noise is added (15-25 dB SNR).

Reported: called students verified, absent students wrongly verified, proxies named, and the same
1:1 rule run on PERFECTLY cut answers (an upper bound: how much the speaker separation loses).
These are synthetic text-to-speech voices: the numbers show how the method behaves, not real accuracy.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_voice import _noise, _take  # noqa: E402
from src.pipelines import voice_pipeline as vp  # noqa: E402

SR = vp.SAMPLE_RATE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    import librosa
    from resemblyzer import preprocess_wav
    from resemblyzer.audio import trim_long_silences

    rng = np.random.default_rng(args.seed)
    enc = vp.load_voice_encoder()
    data = ROOT / "class_data"
    rows = [r for r in csv.DictReader(open(data / "students_with_voice.csv")) if (data / "voices" / r["voice_audio"]).exists()]
    clips = [librosa.load(data / "voices" / r["voice_audio"], sr=SR)[0] for r in rows]
    embs = np.array([enc.embed_utterance(preprocess_wav(c)) for c in clips])
    keep = []
    for i in range(len(clips)):           # drop duplicate synthetic voices
        if all(float(embs[i] @ embs[j]) < 0.95 for j in keep):
            keep.append(i)
    profiles = {}
    for i in keep:
        a = clips[i]
        e = np.concatenate([_take(a, rng, 0.97), np.zeros(SR // 2), _take(a, rng, 1.03, gain=0.8)])
        profiles[i] = enc.embed_utterance(preprocess_wav(e)).tolist()
    print(f"{len(keep)} distinct voices; {args.trials} simulated recordings")

    def crop(a, lo, hi):
        n = int(rng.uniform(lo, hi) * SR); st = int(rng.integers(0, max(1, len(a) - n)))
        return a[st:st + n]

    t = dict(ok=0, n=0, false=0, proxy=0, oracle=0)
    for _ in range(args.trials):
        pool = [int(x) for x in rng.choice(keep, 16, replace=False)]
        enrolled, teacher = pool[:15], pool[15]
        called = enrolled[:5]; answering = called[:3]
        proxy_voice = enrolled[int(rng.integers(5, 15))]
        order = answering + [proxy_voice]; rng.shuffle(order)
        parts, bounds, pos = [np.zeros(SR // 2)], [], SR // 2
        for who in order:
            call = crop(trim_long_silences(_take(clips[teacher], rng, rt=0.2)), 0.6, 0.9)
            ans = crop(trim_long_silences(_take(clips[who], rng, float(rng.uniform(0.92, 1.08)), float(rng.uniform(-0.6, 0.6)),
                                                float(rng.uniform(0.3, 1.0)), float(rng.uniform(0.15, 0.35)))), 1.6, 2.6)
            g1, g2 = int(rng.uniform(0.2, 0.8) * SR), int(rng.uniform(0.1, 0.9) * SR)
            parts += [call, np.zeros(g1), ans, np.zeros(g2)]
            pos += len(call) + g1; bounds.append((who, pos, pos + len(ans))); pos += len(ans) + g2
        audio = np.concatenate(parts)
        rms = np.sqrt(np.mean(audio[audio != 0] ** 2))
        audio = np.clip(audio + _noise(len(audio), rms / (10 ** (rng.uniform(15, 25) / 20)) * 1.5, rng), -1, 1).astype(np.float32)

        r = vp.verify_group(vp.diarize(audio, enc), called, profiles)
        t["n"] += len(answering)
        t["ok"] += sum(i in r.verified for i in answering)
        t["false"] += sum(i in r.verified for i in called[3:])
        t["proxy"] += proxy_voice in r.proxies
        for who, a, b in bounds:          # same rule on perfectly cut answers
            if who in answering:
                t["oracle"] += vp.verify_speaker(vp._embed(enc, audio[a:b]), who, profiles).status == "verified"

    n = args.trials
    print(f"called students who answered, verified : {t['ok']}/{t['n']}  (perfect cuts: {t['oracle']}/{t['n']})")
    print(f"absent students wrongly verified        : {t['false']}/{2 * n}")
    print(f"proxy voices named                      : {t['proxy']}/{n}")


if __name__ == "__main__":
    main()
