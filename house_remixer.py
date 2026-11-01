#!/usr/bin/env python3
"""
House Remixer - trasforma una canzone in un remix house.

Pipeline:
  1. carica l'audio (WAV nativo, qualsiasi altro formato tramite ffmpeg)
  2. rileva BPM, griglia dei beat, primo downbeat e tonalita'
  3. porta il brano al BPM house scelto (phase vocoder: la tonalita' non cambia)
  4. analizza la nota di basso di ogni battuta per far seguire al nuovo basso
     l'armonia originale
  5. arrangia un remix (intro, build, drop, break, build, drop, outro) con
     batteria, basso, stab/pad sintetizzati, sidechain, filtri, riser, riverbero
  6. esporta il mix (WAV/MP3) ed eventualmente gli stem separati

Dipendenze: solo numpy (+ ffmpeg opzionale per MP3/FLAC/M4A...).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SR = 44100
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
# Profili di Krumhansl-Kessler per la stima della tonalita'
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def log(msg: str) -> None:
    print(msg, flush=True)


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


# ---------------------------------------------------------------------------
# I/O audio
# ---------------------------------------------------------------------------

def resample_linear(x: np.ndarray, n_out: int) -> np.ndarray:
    """Ricampiona (n, ch) a n_out campioni con interpolazione lineare."""
    t_in = np.arange(len(x))
    t_out = np.linspace(0, len(x) - 1, n_out)
    return np.stack([np.interp(t_out, t_in, x[:, c]) for c in range(x.shape[1])], axis=1)


def _read_wav(path: str, sr: int) -> np.ndarray:
    with wave.open(path, "rb") as w:
        ch, sw, fr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        data = w.readframes(n)
    if sw == 1:
        x = (np.frombuffer(data, np.uint8).astype(np.float64) - 128) / 128.0
    elif sw == 2:
        x = np.frombuffer(data, "<i2") / 32768.0
    elif sw == 3:
        b = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 2 ** 23, v - 2 ** 24, v) / float(2 ** 23)
    elif sw == 4:
        x = np.frombuffer(data, "<i4") / 2147483648.0
    else:
        raise wave.Error(f"sample width {sw} non supportata")
    x = x.reshape(-1, ch)
    if ch == 1:
        x = np.repeat(x, 2, axis=1)
    elif ch > 2:
        x = x[:, :2]
    if fr != sr:
        x = resample_linear(x, int(round(len(x) * sr / fr)))
    return x.astype(np.float64)


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Restituisce un array float64 (n, 2) a `sr` Hz."""
    if not os.path.isfile(path):
        raise SystemExit(f"File non trovato: {path}")
    if path.lower().endswith(".wav"):
        try:
            return _read_wav(path, sr)
        except (wave.Error, EOFError):
            pass  # es. WAV float32: lo passiamo a ffmpeg
    if shutil.which("ffmpeg") is None:
        raise SystemExit("Per leggere questo formato serve ffmpeg (oppure converti in WAV PCM).")
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise SystemExit("ffmpeg non riesce a leggere il file:\n" + res.stderr.decode(errors="ignore"))
    return np.frombuffer(res.stdout, dtype="<f4").reshape(-1, 2).astype(np.float64)


def _write_wav16(path: str, x: np.ndarray, sr: int) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def save_audio(path: str, x: np.ndarray, sr: int = SR) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith(".wav"):
        _write_wav16(path, x, sr)
        return
    if shutil.which("ffmpeg") is None:
        raise SystemExit("Per esportare in questo formato serve ffmpeg: usa un'uscita .wav")
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "mix.wav")
        _write_wav16(tmp, x, sr)
        extra = ["-b:a", "320k"] if path.lower().endswith(".mp3") else []
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, *extra, path], check=True)


# ---------------------------------------------------------------------------
# DSP di base (tutto vettorizzato con numpy)
# ---------------------------------------------------------------------------

def time_stretch(x: np.ndarray, rate: float, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Phase vocoder stereo-coerente. rate > 1 = piu' veloce. Il pitch non cambia."""
    if abs(rate - 1.0) < 1e-4:
        return x.copy()
    win = np.hanning(n_fft)
    pad = n_fft
    xp = np.pad(x, ((pad, pad + n_fft), (0, 0)))
    n_frames = (len(xp) - n_fft) // hop
    steps = np.arange(0, n_frames - 1, rate)
    omega = 2 * np.pi * hop * np.arange(n_fft // 2 + 1) / n_fft
    y = np.zeros((len(steps) * hop + n_fft, 2))
    wsum = np.zeros(len(y))
    cache: dict[int, np.ndarray] = {}

    def frame(k: int) -> np.ndarray:
        if k not in cache:
            if len(cache) > 4:
                cache.clear()
            seg = xp[k * hop:k * hop + n_fft].T * win
            cache[k] = np.fft.rfft(seg, axis=1)  # (2, bins)
        return cache[k]

    phase = None
    for i, s in enumerate(steps):
        k = int(s)
        a = s - k
        f0, f1 = frame(k), frame(k + 1)
        mid0, mid1 = f0.sum(0), f1.sum(0)
        if phase is None:
            phase = np.angle(mid0)
        mag = (1 - a) * np.abs(f0) + a * np.abs(f1)
        # conserva la differenza di fase L/R rispetto al canale mid -> immagine stereo stabile
        rel = np.angle(f0) - np.angle(mid0)[None, :]
        Y = mag * np.exp(1j * (phase[None, :] + rel))
        out = np.fft.irfft(Y, n=n_fft, axis=1) * win
        y[i * hop:i * hop + n_fft] += out.T
        wsum[i * hop:i * hop + n_fft] += win ** 2
        dp = np.angle(mid1) - np.angle(mid0) - omega
        dp -= 2 * np.pi * np.round(dp / (2 * np.pi))
        phase = phase + omega + dp
    y /= np.maximum(wsum, 1e-3)[:, None]
    start = int(round(pad / rate))
    length = int(round(len(x) / rate))
    return y[start:start + length]


def pitch_shift(x: np.ndarray, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-3:
        return x
    p = 2.0 ** (semitones / 12.0)
    return resample_linear(time_stretch(x, 1.0 / p), len(x))


def stft_filter(x: np.ndarray, gain_fn, sr: int = SR, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Filtro tempo-variante: gain_fn(freqs, progress 0..1) -> guadagno per bin."""
    if len(x) == 0:
        return x
    win = np.hanning(n_fft)
    pad = n_fft
    xp = np.pad(x, ((pad, pad + n_fft), (0, 0)))
    n_frames = (len(xp) - n_fft) // hop + 1
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    y = np.zeros_like(xp)
    wsum = np.zeros(len(xp))
    for i in range(n_frames):
        a = i * hop
        prog = float(np.clip((a - pad + n_fft / 2) / max(len(x), 1), 0, 1))
        seg = xp[a:a + n_fft].T * win
        out = np.fft.irfft(np.fft.rfft(seg, axis=1) * gain_fn(freqs, prog), n=n_fft, axis=1) * win
        y[a:a + n_fft] += out.T
        wsum[a:a + n_fft] += win ** 2
    y /= np.maximum(wsum, 1e-6)[:, None]
    return y[pad:pad + len(x)]


def _lp(f, fc, order=2):
    return 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))


def _hp(f, fc, order=2):
    return 1.0 / np.sqrt(1.0 + (fc / np.maximum(f, 1.0)) ** (2 * order))


def sweep(f0: float, f1: float, prog: float) -> float:
    """Interpolazione logaritmica della frequenza di taglio."""
    return f0 * (f1 / f0) ** prog


def lowpass_sweep(x, f0, f1, res=0.6, sr=SR):
    def g(f, p):
        fc = sweep(f0, f1, p)
        bump = res * np.exp(-(np.log2(np.maximum(f, 1) / fc) ** 2) / (2 * 0.12 ** 2))
        return _lp(f, fc, 3) + bump
    return stft_filter(x, g, sr)


def highpass_sweep(x, f0, f1, sr=SR):
    return stft_filter(x, lambda f, p: _hp(f, sweep(f0, f1, p), 3), sr)


def fft_band(sig: np.ndarray, sr: int, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Filtro statico via FFT su un segnale breve (1D o 2D lungo l'asse 0)."""
    n = len(sig)
    X = np.fft.rfft(sig, axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    g = np.ones_like(f)
    if lo:
        g *= _hp(f, lo, 2)
    if hi:
        g *= _lp(f, hi, 2)
    if sig.ndim == 2:
        g = g[:, None]
    return np.fft.irfft(X * g, n=n, axis=0)


def fft_convolve(x: np.ndarray, ir: np.ndarray, block: int = 65536) -> np.ndarray:
    """Convoluzione overlap-add (x: (n,2), ir: (m,2)). Output lungo n + m - 1."""
    m = len(ir)
    nfft = 1 << int(np.ceil(np.log2(block + m - 1)))
    H = np.fft.rfft(ir, n=nfft, axis=0)
    y = np.zeros((len(x) + m - 1, 2))
    for s in range(0, len(x), block):
        seg = x[s:s + block]
        if not np.any(seg):
            continue
        out = np.fft.irfft(np.fft.rfft(seg, n=nfft, axis=0) * H, n=nfft, axis=0)
        e = min(len(y), s + len(seg) + m - 1)
        y[s:e] += out[:e - s]
    return y


def make_reverb_ir(rng, seconds=2.2, sr=SR, damp=6000):
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * t / seconds)[:, None]
    ir = fft_band(ir, sr, lo=200, hi=damp)
    ir[: int(0.012 * sr)] = 0  # pre-delay
    return ir / np.sqrt(np.sum(ir ** 2) / 2)


def ping_pong_echo(x: np.ndarray, delay: int, feedback=0.45, taps=5) -> np.ndarray:
    y = x.copy()
    g = 1.0
    for i in range(1, taps + 1):
        g *= feedback
        d = delay * i
        if d >= len(x):
            break
        src = x[:-d] * g
        if i % 2:  # alterna i canali
            src = src[:, ::-1]
        y[d:] += src
    return y


def fade(x: np.ndarray, ms_in=5.0, ms_out=5.0, sr=SR) -> np.ndarray:
    x = x.copy()
    a, b = min(len(x), int(sr * ms_in / 1000)), min(len(x), int(sr * ms_out / 1000))
    if a:
        x[:a] *= np.linspace(0, 1, a)[:, None]
    if b:
        x[-b:] *= np.linspace(1, 0, b)[:, None]
    return x


# ---------------------------------------------------------------------------
# Analisi: BPM, beat, tonalita', note di basso per battuta
# ---------------------------------------------------------------------------

def _analysis_signal(x: np.ndarray, sr: int, target: int = 11025):
    mono = x.mean(axis=1)
    f = max(1, sr // target)
    n = len(mono) // f * f
    return mono[:n].reshape(-1, f).mean(axis=1), sr / f


def magnitude_spectrogram(sig: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    win = np.hanning(n_fft).astype(np.float32)
    sig = np.pad(sig.astype(np.float32), (n_fft // 2, n_fft))
    frames = sliding_window_view(sig, n_fft)[::hop]
    out = np.empty((len(frames), n_fft // 2 + 1), np.float32)
    for i in range(0, len(frames), 256):
        out[i:i + 256] = np.abs(np.fft.rfft(frames[i:i + 256] * win, axis=1))
    return out


def _comb(env: np.ndarray, fps: float, bpm: float, nbins: int):
    period = fps * 60.0 / bpm
    ph = (np.arange(len(env)) % period) / period
    h = np.bincount((ph * nbins).astype(int) % nbins, weights=env, minlength=nbins)
    h = (np.roll(h, 1) + 2 * h + np.roll(h, -1)) / 4
    j = int(np.argmax(h))
    return h[j], (j + 0.5) / nbins * period / fps


def detect_beats(x: np.ndarray, sr: int, lo=60.0, hi=190.0, bpm_override=None):
    """Restituisce (bpm, secondi del primo downbeat)."""
    sig, sra = _analysis_signal(x, sr)
    hop = 128
    M = magnitude_spectrogram(sig, 1024, hop)
    fps = sra / hop
    L = np.log1p(10 * M)
    d = np.maximum(0, np.diff(L, axis=0))
    freqs = np.fft.rfftfreq(1024, 1 / sra)
    flux = np.concatenate([[0], d.sum(1)])
    flux_low = np.concatenate([[0], d[:, freqs < 150].sum(1)])
    k = max(1, int(fps * 0.5))
    env = np.maximum(0, flux - np.convolve(flux, np.ones(k) / k, "same"))

    if bpm_override:
        bpm = float(bpm_override)
    else:
        grid = np.arange(lo, hi + 1e-9, 0.2)
        scores = [_comb(env, fps, b, 32)[0] for b in grid]
        coarse = grid[int(np.argmax(scores))]
        # brani lenti: se il doppio tempo e' quasi altrettanto forte, usalo
        # (stretch meno estremo verso i ~124 BPM della house)
        if coarse < 80:
            dbl = _comb(env, fps, coarse * 2, 32)[0]
            if dbl >= 0.85 * max(scores):
                coarse *= 2
        fine = np.arange(coarse - 0.3, coarse + 0.3 + 1e-9, 0.01)
        fs = np.array([_comb(env, fps, b, 64)[0] for b in fine])
        bpm = float(fine[int(np.argmax(fs))])
        # molti brani hanno BPM interi: aggancia se praticamente equivalente
        r = round(bpm)
        if abs(r - bpm) < 0.15 and _comb(env, fps, r, 64)[0] >= 0.97 * fs.max():
            bpm = float(r)
    _, phase = _comb(env, fps, bpm, 64)
    beat = 60.0 / bpm
    # la finestra di Hann "vede" l'attacco ~n_fft/4 campioni in anticipo: compensa
    phase = (phase + 1024 / 4 / sra) % beat

    # downbeat: quale dei 4 beat della battuta ha piu' energia (basse + totale)
    n_beats = int((len(env) / fps - phase) / beat)
    best_j, best_s = 0, -1.0
    tot = flux / (flux.max() + 1e-9) + 2 * flux_low / (flux_low.max() + 1e-9)
    for j in range(4):
        idx = np.round((phase + (np.arange(j, n_beats, 4)) * beat) * fps).astype(int)
        idx = idx[idx < len(tot)]
        s = float(tot[idx].mean()) if len(idx) else 0
        if s > best_s * 1.02:  # a parita', preferisci il primo beat
            best_j, best_s = j, s
    return bpm, phase + best_j * beat


def harmonic_analysis(x: np.ndarray, sr: int, offset: float, bar_len: float, n_bars: int):
    """Tonalita' globale + pitch class della nota di basso di ogni battuta."""
    sig, sra = _analysis_signal(x, sr)
    n_fft, hop = 8192, 2048
    M = magnitude_spectrogram(sig, n_fft, hop) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sra)

    def chroma(fmin, fmax):
        m = (freqs >= fmin) & (freqs <= fmax)
        pcs = np.round(12 * np.log2(freqs[m] / 440.0) + 69).astype(int) % 12
        spec = M[:, m]
        return np.stack([spec[:, pcs == p].sum(1) for p in range(12)], axis=1)

    full = chroma(80, 2000).sum(0)
    full = full / (full.max() + 1e-12)
    best = (-2.0, 0, "minor")
    for root in range(12):
        for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            c = np.corrcoef(np.roll(prof, root), full)[0, 1]
            if c > best[0]:
                best = (c, root, mode)
    _, key_root, mode = best

    low = chroma(65, 260)
    t = np.arange(len(M)) * hop / sra
    bar_idx = np.floor((t - offset) / bar_len).astype(int)
    ok = (bar_idx >= 0) & (bar_idx < n_bars)
    per_bar = np.zeros((n_bars, 12))
    np.add.at(per_bar, bar_idx[ok], low[ok])
    scale = MINOR_SCALE if mode == "minor" else MAJOR_SCALE
    pcs = [(key_root + s) % 12 for s in scale]
    roots = []
    energy = per_bar.sum(1)
    thr = 0.05 * (np.median(energy[energy > 0]) if np.any(energy > 0) else 1)
    prev = key_root
    for b in range(n_bars):
        if energy[b] < thr:
            roots.append(prev)
            continue
        sc = per_bar[b, pcs] * np.where(np.array(pcs) == key_root, 1.15, 1.0)
        prev = pcs[int(np.argmax(sc))]
        roots.append(prev)
    return key_root, mode, roots


# ---------------------------------------------------------------------------
# Sintesi strumenti
# ---------------------------------------------------------------------------

class Kit:
    """Batteria sintetizzata (originale, nessun campione esterno)."""

    def __init__(self, rng, sr=SR):
        self.sr, self.rng = sr, rng
        self.kick = self._kick()
        self.clap = self._clap()
        self.hat_c = self._hat(0.07, 85)
        self.hat_o = self._hat(0.35, 13)
        self.shaker = self._shaker()
        self.snare = self._snare()
        self.ride = self._ride()

    def _t(self, sec):
        return np.arange(int(sec * self.sr)) / self.sr

    def _noise(self, n):
        return self.rng.standard_normal(n)

    def _kick(self):
        t = self._t(0.45)
        f = 48 + 125 * np.exp(-t * 38)
        body = np.sin(2 * np.pi * np.cumsum(f) / self.sr) * np.exp(-t * 6.5)
        click = fft_band(self._noise(len(t)), self.sr, 2000, 9000) * np.exp(-t * 400) * 0.25
        k = np.tanh(2.0 * (body + click))
        return k / np.abs(k).max()

    def _clap(self):
        t = self._t(0.35)
        n = fft_band(self._noise(len(t)), self.sr, 900, 5500)
        env = np.zeros_like(t)
        for d in (0.0, 0.011, 0.022):
            env += np.where(t >= d, np.exp(-(t - d) * 180), 0)
        env += np.where(t >= 0.03, 0.55 * np.exp(-(t - 0.03) * 16), 0)
        c = n * env
        return c / np.abs(c).max()

    def _hat(self, sec, decay):
        t = self._t(sec)
        h = fft_band(self._noise(len(t)), self.sr, 7000, 16000) * np.exp(-t * decay)
        return h / np.abs(h).max()

    def _shaker(self):
        t = self._t(0.09)
        env = np.minimum(t / 0.008, 1) * np.exp(-t * 45)
        s = fft_band(self._noise(len(t)), self.sr, 5000, 12000) * env
        return s / np.abs(s).max()

    def _snare(self):
        t = self._t(0.25)
        s = fft_band(self._noise(len(t)), self.sr, 1000, 8000) * np.exp(-t * 18)
        s += 0.6 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25)
        return s / np.abs(s).max()

    def _ride(self):
        t = self._t(0.9)
        r = sum(np.sin(2 * np.pi * f * t) for f in (3430, 4650, 5980, 7210)) * 0.15
        r = (r + fft_band(self._noise(len(t)), self.sr, 5000, 14000)) * np.exp(-t * 5)
        return r / np.abs(r).max()

    def impact(self):
        t = self._t(3.0)
        crash = fft_band(self._noise(len(t)), self.sr, 3000, 15000) * np.exp(-t * 2.2) * 0.6
        f = 35 + 30 * np.exp(-t * 3)
        boom = np.sin(2 * np.pi * np.cumsum(f) / self.sr) * np.exp(-t * 2.5)
        return crash + boom


def bass_note(freq: float, dur: float, sr=SR, cutoff=900.0, env_amt=1800.0,
              decay=9.0, sub=0.6, max_h=40) -> np.ndarray:
    """Basso sottrattivo simulato con sintesi additiva band-limited."""
    t = np.arange(int(dur * sr)) / sr
    if len(t) == 0:
        return t
    K = int(min(max_h, (sr / 2 - 500) // freq))
    k = np.arange(1, K + 1)
    fenv = cutoff + env_amt * np.exp(-t * 25)  # inviluppo del filtro
    w = (1.0 / k)[None, :] / np.sqrt(1 + ((k[None, :] * freq) / fenv[:, None]) ** 4)
    saw = np.sum(np.sin(2 * np.pi * freq * t[:, None] * k[None, :]) * w, axis=1)
    s = 0.6 * saw + sub * np.sin(2 * np.pi * freq * t)
    amp = np.minimum(t / 0.004, 1) * (0.55 + 0.45 * np.exp(-t * decay))
    r = min(len(t), int(0.015 * sr))
    amp[-r:] *= np.linspace(1, 0, r)
    return np.tanh(1.5 * s * amp)


def chord_tone_set(key_root, mode, root_pc):
    scale = MINOR_SCALE if mode == "minor" else MAJOR_SCALE
    pcs = [(key_root + s) % 12 for s in scale]
    i = pcs.index(root_pc) if root_pc in pcs else 0
    return [pcs[i], pcs[(i + 2) % 7], pcs[(i + 4) % 7]]


def voice(pcs, center=64):
    """Sceglie l'ottava di ogni nota piu' vicina a `center` (MIDI)."""
    out = []
    for pc in pcs:
        m = pc + 12 * round((center - pc) / 12)
        out.append(m)
    return sorted(out)


def organ_stab(midis, dur, sr=SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    out = np.zeros((len(t), 2))
    for m in midis:
        f = midi_to_hz(m)
        for h, a in ((1, 1.0), (2, 0.6), (3, 0.35), (4, 0.2), (6, 0.1)):
            if f * h < sr / 2:
                out[:, 0] += a * np.sin(2 * np.pi * f * h * t * 0.9985)
                out[:, 1] += a * np.sin(2 * np.pi * f * h * t * 1.0015)
    env = np.minimum(t / 0.003, 1) * np.exp(-t * 9)
    return out * env[:, None] / (len(midis) * 2.0)


def pad_chord(midis, dur, sr=SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    out = np.zeros((len(t), 2))
    for m in midis:
        f = midi_to_hz(m)
        for ch, det in ((0, 0.996), (1, 1.004)):
            for h in range(1, 9):
                out[:, ch] += np.sin(2 * np.pi * f * det * h * t) / h / (1 + (h * f / 1500) ** 2)
    a, r = min(0.4, dur / 3), min(0.5, dur / 3)
    env = np.minimum(t / a, 1) * np.clip((dur - t) / r, 0, 1)
    return out * env[:, None] / (len(midis) * 2.0)


def riser(rng, seconds, sr=SR) -> np.ndarray:
    n = int(seconds * sr)
    x = rng.standard_normal((n, 2)) * 0.5

    def g(f, p):
        fc = sweep(300, 9000, p)
        return np.exp(-(np.log2(np.maximum(f, 1) / fc) ** 2) / (2 * 0.8 ** 2))
    y = stft_filter(x, g, sr)
    return y * (np.linspace(0, 1, n) ** 2)[:, None]


# ---------------------------------------------------------------------------
# Preset e arrangiamento
# ---------------------------------------------------------------------------

PRESETS = {
    "classic": dict(bpm=124, swing=0.0, sidechain=0.55, bass="offbeat", music="stab",
                    hats="offbeat", ride=False, reverb=0.22, cutoff=900),
    "deep": dict(bpm=120, swing=0.12, sidechain=0.40, bass="deep", music="pad",
                 hats="shaker", ride=False, reverb=0.32, cutoff=450),
    "tech": dict(bpm=126, swing=0.0, sidechain=0.65, bass="rolling", music=None,
                 hats="sixteenths", ride=True, reverb=0.18, cutoff=650),
}

# (step in sedicesimi, durata in sedicesimi, ottava, velocity)
BASS_PATTERNS = {
    "offbeat": [(2, 2, 0, 1.0), (6, 2, 0, 0.9), (10, 2, 0, 1.0), (14, 2, 1, 0.85)],
    "deep": [(0, 3, 0, 1.0), (3, 1, 0, 0.6), (6, 2, 0, 0.85), (10, 3, 0, 0.9), (14, 1, 1, 0.6)],
    "rolling": [(s, 1, 0, 1.0 if s % 4 == 2 else 0.7) for s in range(16) if s % 4 != 0],
}
STAB_PATTERN = [(3, 2, 0.9), (6, 2, 0.75), (10, 2, 0.9), (13, 2, 0.7)]


@dataclass
class Section:
    name: str
    bars: int
    src: int | None  # battuta di partenza nel brano originale (None = niente brano)
    layers: set = field(default_factory=set)
    src_fx: tuple | None = None  # ("lp"|"hp", f_inizio, f_fine)
    src_gain: float = 1.0


def build_structure(kind: str, n_src: int, phrase_energy: list[float]) -> list[Section]:
    phrases = list(range(0, max(n_src, 1), 8))
    order = sorted(range(len(phrases)), key=lambda i: -phrase_energy[i])
    hook = phrases[order[0]]
    hook2 = phrases[order[1]] if len(order) > 1 else hook + 16
    # il break usa la frase piu' tranquilla (escludendo la prima, spesso intro)
    calm_cands = [i for i in range(1, len(phrases))] or [0]
    calm = phrases[min(calm_cands, key=lambda i: phrase_energy[i])]
    drop = {"impact", "kick", "clap", "hats", "bass", "music", "src", "lowcut"}
    if kind == "short":
        return [
            Section("Intro", 8, 0, {"kick", "hats", "src"}, ("lp", 250, 2500), 0.6),
            Section("Build", 8, hook, {"kick_build", "roll", "riser", "src"}, ("hp", 80, 1500)),
            Section("Drop", 16, hook, set(drop)),
            Section("Break", 8, calm, {"pad", "src", "src_verb", "hats_light"}, ("hp", 400, 250)),
            Section("Drop 2", 16, hook2, set(drop)),
            Section("Outro", 8, hook2 + 16, {"kick", "clap", "hats", "src", "fadeout"}, ("lp", 6000, 250), 0.8),
        ]
    return [
        Section("Intro A", 8, None, {"kick", "hats_light"}),
        Section("Intro B", 8, 0, {"kick", "clap", "hats", "src"}, ("lp", 250, 2500), 0.6),
        Section("Build 1", 8, hook, {"kick_build", "roll", "riser", "src"}, ("hp", 80, 1500)),
        Section("Drop 1", 16, hook, set(drop)),
        Section("Break", 8, calm, {"pad", "echo_stab", "src", "src_verb", "hats_light"}, ("hp", 400, 250)),
        Section("Build 2", 8, hook, {"kick_build", "roll", "riser", "src"}, ("hp", 100, 2000)),
        Section("Drop 2", 16, hook2, set(drop)),
        Section("Outro", 8, hook2 + 16, {"kick", "clap", "hats", "bass_half", "src", "fadeout"},
                ("lp", 6000, 250), 0.8),
    ]


class Remixer:
    BUSES = ("drums", "bass", "music", "source", "fx")

    def __init__(self, song: np.ndarray, sr: int, bpm: float, offset: float,
                 key_root: int, mode: str, bar_roots: list[int], preset: dict,
                 rng, lowcut=True, gains=None):
        self.sr, self.bpm, self.p, self.rng = sr, bpm, preset, rng
        self.song, self.offset = song, offset
        self.key_root, self.mode, self.bar_roots = key_root, mode, bar_roots
        self.beat = 60.0 / bpm
        self.bar_s = 4 * self.beat * sr  # battuta in campioni (float)
        self.n_src = len(bar_roots)
        self.kit = Kit(rng, sr)
        self.lowcut = lowcut
        self.gains = dict(drums=1.0, bass=1.0, music=1.0, source=1.0, fx=1.0)
        self.gains.update(gains or {})
        self.kick_times: list[int] = []

    # --- utilita' di tempo e posizionamento -------------------------------
    def pos(self, bar: float, step: float = 0.0) -> int:
        swing = self.p["swing"] if int(step) % 2 == 1 else 0.0
        return int(round((bar * 16 + step + swing) * self.bar_s / 16))

    def place(self, bus, sample, at, gain=1.0, pan=0.0):
        if at >= len(bus) or gain == 0:
            return
        if sample.ndim == 1:
            th = (pan + 1) * np.pi / 4
            sample = np.stack([sample * np.cos(th), sample * np.sin(th)], axis=1) * np.sqrt(2)
        n = min(len(sample), len(bus) - at)
        bus[at:at + n] += sample[:n] * gain

    def vel(self, v):
        return v * (1 + self.rng.uniform(-0.08, 0.08))

    def src_bars(self, start: int, n: int) -> np.ndarray:
        """Estrae n battute del brano (a partire da `start`, in loop se finisce)."""
        parts = []
        for b in range(start, start + n):
            bb = b % max(self.n_src, 1)
            a = int(round(self.offset * self.sr + bb * self.bar_s))
            e = int(round(self.offset * self.sr + (bb + 1) * self.bar_s))
            seg = self.song[a:e]
            if bb == 0 or b % max(self.n_src, 1) == 0:
                seg = fade(seg, 3, 0, self.sr)
            parts.append(seg)
        out = np.concatenate(parts) if parts else np.zeros((0, 2))
        return fade(out, 4, 4, self.sr)

    def root_at(self, src_bar):
        if src_bar is None or self.n_src == 0:
            return self.key_root
        return self.bar_roots[src_bar % self.n_src]

    # --- render ---------------------------------------------------------------
    def render(self, sections: list[Section]):
        total_bars = sum(s.bars for s in sections)
        N = int(total_bars * self.bar_s + 4 * self.sr)
        B = {k: np.zeros((N, 2)) for k in self.BUSES}
        send = np.zeros((N, 2))
        sc_free = np.zeros(N, bool)  # zone dove la sidechain non si applica
        bar0 = 0
        for sec in sections:
            log(f"  - {sec.name:<8} battute {bar0 + 1:>3}-{bar0 + sec.bars:<3} "
                f"[{', '.join(sorted(sec.layers))}]")
            self._render_section(sec, bar0, B, send)
            bar0 += sec.bars

        # reverb (send) e sidechain
        ir = make_reverb_ir(self.rng, 2.4 if self.p["music"] == "pad" else 1.8, self.sr)
        B["fx"] += fft_convolve(send, ir)[:N] * self.p["reverb"]
        sc = self._sidechain_curve(N)
        depth = self.p["sidechain"]
        for name, d in (("bass", 1.0), ("music", 0.9), ("source", 0.6), ("fx", 0.7)):
            B[name] *= (1 - depth * d * sc)[:, None]
        for k in B:
            B[k] *= self.gains.get(k, 1.0)
        return B

    def _sidechain_curve(self, N):
        L = int(self.beat * self.sr)
        t = np.arange(L) / self.sr
        dip = np.exp(-t / 0.11) * np.minimum(t / 0.003, 1)
        c = np.zeros(N)
        for k in self.kick_times:
            n = min(L, N - k)
            if n > 0:
                c[k:k + n] = np.maximum(c[k:k + n], dip[:n])
        return c

    def _render_section(self, sec: Section, bar0: int, B, send):
        p, kit, sr = self.p, self.kit, self.sr
        Ly = sec.layers
        start = self.pos(bar0)
        end = self.pos(bar0 + sec.bars)

        # ---- brano originale -------------------------------------------------
        if "src" in Ly and sec.src is not None and self.n_src > 0:
            seg = self.src_bars(sec.src, sec.bars)[: end - start]
            if sec.src_fx:
                kind, f0, f1 = sec.src_fx
                seg = lowpass_sweep(seg, f0, f1, sr=sr) if kind == "lp" else highpass_sweep(seg, f0, f1, sr)
            elif "lowcut" in Ly and self.lowcut:
                seg = stft_filter(seg, lambda f, _p: _hp(f, 140, 2), sr)
            if "fadeout" in Ly:
                seg = seg * np.linspace(1, 0, len(seg))[:, None] ** 1.5
            self.place(B["source"], seg, start, 0.8 * sec.src_gain)
            if "src_verb" in Ly:
                self.place(send, seg, start, 0.8)

        # ---- batteria ---------------------------------------------------------
        for b in range(sec.bars):
            bar = bar0 + b
            last = b == sec.bars - 1
            fo = 1.0 - (b / sec.bars) * 0.8 if "fadeout" in Ly else 1.0
            kick_on = "kick" in Ly or ("kick_build" in Ly and not last)
            if kick_on:
                for s in (0, 4, 8, 12):
                    at = self.pos(bar, s)
                    self.place(B["drums"], kit.kick, at, 0.95 * fo)
                    self.kick_times.append(at)
            if "clap" in Ly:
                for s in (4, 12):
                    self.place(B["drums"], kit.clap, self.pos(bar, s), self.vel(0.5) * fo)
                    self.place(send, kit.clap, self.pos(bar, s), 0.15)
            if "hats" in Ly:
                self._hats(B["drums"], bar, fo)
            elif "hats_light" in Ly:
                for s in (2, 6, 10, 14):
                    self.place(B["drums"], kit.hat_c, self.pos(bar, s), self.vel(0.22), pan=0.2)
            if "roll" in Ly:
                frac = b / sec.bars
                div = 4 if frac < 0.5 else 2 if frac < 0.75 else 1 if not last else 0.5
                s = 0.0
                while s < 16:
                    v = 0.25 + 0.6 * (b * 16 + s) / (sec.bars * 16)
                    self.place(B["drums"], kit.snare, self.pos(bar, s), v)
                    s += div

        # ---- effetti -----------------------------------------------------
        if "riser" in Ly:
            r = riser(self.rng, (end - start) / sr, sr)
            self.place(B["fx"], r, start, 0.3)
            self.place(send, r, start, 0.3)
        if "impact" in Ly:
            imp = kit.impact()
            self.place(B["fx"], imp, start, 0.35)
            self.place(send, imp, start, 0.4)

        # ---- basso --------------------------------------------------------
        if "bass" in Ly or "bass_half" in Ly:
            nb = sec.bars if "bass" in Ly else sec.bars // 2
            pat = BASS_PATTERNS[p["bass"]]
            step_s = self.bar_s / 16 / sr
            for b in range(nb):
                src_bar = None if sec.src is None else sec.src + b
                pc = self.root_at(src_bar)
                base = 36 + (pc % 12)  # C2..B2
                if base > 43:
                    base -= 12  # resta nel registro basso (E1..G2)
                for st, ln, octv, v in pat:
                    note = bass_note(midi_to_hz(base + 12 * octv), ln * step_s * 0.95, sr,
                                     cutoff=p["cutoff"], max_h=12 if p["bass"] == "deep" else 40)
                    self.place(B["bass"], note, self.pos(bar0 + b, st), self.vel(0.5 * v))

        # ---- stab / pad ---------------------------------------------------------
        music = p["music"]
        if "music" in Ly and music or "pad" in Ly or "echo_stab" in Ly:
            stab_bus = np.zeros((end - start + 2 * sr, 2))
            for b in range(sec.bars):
                src_bar = None if sec.src is None else sec.src + b
                chord = chord_tone_set(self.key_root, self.mode, self.root_at(src_bar))
                bar = bar0 + b
                if "pad" in Ly or ("music" in Ly and music == "pad"):
                    g = 0.35 if "pad" in Ly else 0.18
                    pc = pad_chord(voice(chord, 60), self.bar_s / sr * 1.02, sr)
                    self.place(B["music"], pc, self.pos(bar), g)
                    self.place(send, pc, self.pos(bar), g)
                if ("music" in Ly and music == "stab") or "echo_stab" in Ly:
                    pat = STAB_PATTERN if "music" in Ly else STAB_PATTERN[:1]
                    for st, ln, v in pat:
                        s = organ_stab(voice(chord, 66), ln * self.bar_s / 16 / sr, sr)
                        self.place(stab_bus, s, self.pos(bar, st) - start, self.vel(0.55 * v))
            if np.any(stab_bus):
                if "echo_stab" in Ly:
                    stab_bus = ping_pong_echo(stab_bus, int(0.75 * self.beat * sr))
                self.place(B["music"], stab_bus, start)
                self.place(send, stab_bus, start, 0.3)

    def _hats(self, bus, bar, fo):
        kit, style = self.kit, self.p["hats"]
        for s in range(16):
            at = self.pos(bar, s)
            if style == "offbeat":
                if s % 4 == 2:
                    self.place(bus, kit.hat_o, at, self.vel(0.32) * fo, pan=-0.1)
                elif s % 2 == 1:
                    self.place(bus, kit.hat_c, at, self.vel(0.14) * fo, pan=0.25)
            elif style == "shaker":
                v = (0.22, 0.12, 0.3, 0.12)[s % 4]
                self.place(bus, kit.shaker, at, self.vel(v) * fo, pan=0.3)
                if s % 4 == 2:
                    self.place(bus, kit.hat_o, at, self.vel(0.18) * fo, pan=-0.2)
            else:  # sixteenths
                v = (0.2, 0.12, 0.3, 0.12)[s % 4]
                self.place(bus, kit.hat_c, at, self.vel(v) * fo, pan=0.2)
                if s % 4 == 2:
                    self.place(bus, kit.hat_o, at, self.vel(0.25) * fo, pan=-0.15)
            if self.p["ride"] and s % 2 == 0:
                self.place(bus, kit.ride, at, self.vel(0.08) * fo, pan=-0.35)


def master(mix: np.ndarray, ceiling_db=-1.0) -> np.ndarray:
    """Normalizzazione + soft-clip + limitazione del picco."""
    ref = np.percentile(np.abs(mix), 99.9) + 1e-9
    y = np.tanh(1.3 * mix / ref) / np.tanh(1.3)
    y = y / (np.abs(y).max() + 1e-9) * 10 ** (ceiling_db / 20)
    return y


def phrase_energies(song, sr, offset, bar_s, n_src):
    out = []
    for b in range(0, max(n_src, 1), 8):
        a = int(offset * sr + b * bar_s)
        e = int(offset * sr + min(b + 8, n_src) * bar_s)
        seg = song[a:e]
        out.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Remixa una canzone in stile house (numpy + ffmpeg opzionale).")
    ap.add_argument("input", help="file audio (wav, mp3, flac, m4a...)")
    ap.add_argument("-o", "--output", help="file di uscita (.wav o .mp3)")
    ap.add_argument("-p", "--preset", choices=PRESETS, default="classic",
                    help="stile: classic (organ stab, basso in levare), deep (pad, shaker), "
                         "tech (basso rolling, ride)")
    ap.add_argument("--bpm", type=float, help="BPM del remix (default del preset)")
    ap.add_argument("--source-bpm", type=float, help="forza il BPM del brano originale")
    ap.add_argument("--shift-beats", type=int, default=0,
                    help="sposta il downbeat di N beat se la battuta risulta sfasata")
    ap.add_argument("--semitones", type=float, default=0.0, help="trasposizione in semitoni")
    ap.add_argument("--structure", choices=("full", "short"), default="full")
    ap.add_argument("--no-lowcut", action="store_true",
                    help="non tagliare le basse del brano originale nei drop")
    ap.add_argument("--source-gain", type=float, default=1.0)
    ap.add_argument("--drums-gain", type=float, default=1.0)
    ap.add_argument("--bass-gain", type=float, default=1.0)
    ap.add_argument("--stems", metavar="DIR", help="esporta anche gli stem in questa cartella")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    preset = dict(PRESETS[a.preset])
    target_bpm = a.bpm or preset["bpm"]
    rng = np.random.default_rng(a.seed)
    out = a.output or os.path.splitext(os.path.basename(a.input))[0] + f"_house_{a.preset}.wav"

    log(f"Carico {a.input} ...")
    x = load_audio(a.input)
    log(f"  durata {len(x) / SR:.1f}s")

    log("Analisi del ritmo ...")
    src_bpm, downbeat = detect_beats(x, SR, bpm_override=a.source_bpm)
    downbeat += a.shift_beats * 60.0 / src_bpm
    downbeat %= 4 * 60.0 / src_bpm
    log(f"  BPM originale ~ {src_bpm:.2f}, primo downbeat a {downbeat:.3f}s")

    rate = target_bpm / src_bpm
    if abs(rate - 1) > 1e-4:
        log(f"Time-stretch {src_bpm:.2f} -> {target_bpm:.2f} BPM (x{rate:.3f}) ...")
    song = time_stretch(x, rate)
    if a.semitones:
        log(f"Trasposizione di {a.semitones:+g} semitoni ...")
        song = pitch_shift(song, a.semitones)
    offset = downbeat / rate
    bar_s = 4 * 60.0 / target_bpm * SR
    n_src = int((len(song) - offset * SR) // bar_s)
    if n_src < 1:
        raise SystemExit("Il brano e' troppo corto (serve almeno una battuta).")

    log("Analisi armonica ...")
    key_root, mode, roots = harmonic_analysis(song, SR, offset, bar_s / SR, n_src)
    log(f"  tonalita' stimata: {NOTE_NAMES[key_root]} {'minore' if mode == 'minor' else 'maggiore'}, "
        f"{n_src} battute utilizzabili")

    sections = build_structure(a.structure, n_src, phrase_energies(song, SR, offset, bar_s, n_src))
    log(f"Arrangiamento ({a.preset}, {target_bpm:g} BPM):")
    rmx = Remixer(song, SR, target_bpm, offset, key_root, mode, roots, preset, rng,
                  lowcut=not a.no_lowcut,
                  gains=dict(source=a.source_gain, drums=a.drums_gain, bass=a.bass_gain))
    buses = rmx.render(sections)

    log("Mastering ed esportazione ...")
    mix = sum(buses.values())
    peak_ref = np.percentile(np.abs(mix), 99.9) + 1e-9
    final = master(mix)
    final = final[: int(np.max(np.nonzero(np.abs(final).max(1) > 1e-4)[0], initial=0)) + SR // 2]
    save_audio(out, final)
    if a.stems:
        for k, v in buses.items():
            stem = np.clip(v / peak_ref * 0.89, -1, 1)[: len(final)]
            save_audio(os.path.join(a.stems, f"{k}.wav"), stem)
        log(f"  stem salvati in {a.stems}/")
    log(f"Fatto! -> {out}  ({len(final) / SR:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
