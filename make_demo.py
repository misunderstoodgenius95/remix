#!/usr/bin/env python3
"""Genera una canzone demo originale (pop lento, La minore) per provare il remixer."""
import argparse

import numpy as np

from house_remixer import SR, Kit, bass_note, midi_to_hz, pad_chord, save_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="demo_song.wav")
    ap.add_argument("--bpm", type=float, default=100.0)
    ap.add_argument("--bars", type=int, default=40)
    a = ap.parse_args()

    rng = np.random.default_rng(1)
    kit = Kit(rng)
    bar = 4 * 60 / a.bpm * SR
    N = int(a.bars * bar + 3 * SR)
    mix = np.zeros((N, 2))
    # progressione: Am - F - C - G
    prog = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [55, 60, 64]), (55, [55, 59, 62])]
    melody = [69, 72, 71, 67, 69, 64, 65, 67]

    def put(sig, at, g=1.0):
        if sig.ndim == 1:
            sig = np.stack([sig, sig], 1)
        n = min(len(sig), N - at)
        mix[at:at + n] += sig[:n] * g

    for b in range(a.bars):
        root, chord = prog[b % 4]
        t0 = int(b * bar)
        put(pad_chord(chord, bar / SR), t0, 0.5)
        if b >= 4:  # batteria pop: cassa 1 e 3, rullante 2 e 4
            for beat, s in ((0, kit.kick), (1, kit.snare), (2, kit.kick), (3, kit.snare)):
                put(s, t0 + int(beat * bar / 4), 0.6)
            for e in range(8):
                put(kit.hat_c, t0 + int(e * bar / 8), 0.12)
            put(bass_note(midi_to_hz(root - 24), bar / SR * 0.48), t0, 0.35)
            put(bass_note(midi_to_hz(root - 24), bar / SR * 0.48), t0 + int(bar / 2), 0.35)
        if 8 <= b % 16 < 16:  # "ritornello" con melodia
            for i, m in enumerate(melody[(b % 2) * 4:(b % 2) * 4 + 4]):
                t = np.arange(int(bar / 4)) / SR
                f = midi_to_hz(m)
                tone = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.exp(-t * 3)
                put(tone, t0 + int(i * bar / 4), 0.25)
    mix /= np.abs(mix).max() * 1.12
    save_audio(a.output, mix)
    print(f"Demo salvata in {a.output} ({a.bpm:g} BPM, {a.bars} battute)")


if __name__ == "__main__":
    main()
