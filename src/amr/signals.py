"""Baseband signal simulation: random symbols -> RRC pulse shaping -> random phase -> AWGN.

SNR convention: signal power is normalised to 1 per complex sample; AWGN has total variance 1/SNR across the
whole sampled bandwidth (so it is "wideband" SNR, not in-band SNR).
"""
from __future__ import annotations

import numpy as np

MODULATIONS = ["BPSK", "QPSK", "16QAM"]
SPS = 8          # samples per symbol
ROLLOFF = 0.35
SPAN = 8         # RRC filter span in symbols


def rrc_taps(beta: float = ROLLOFF, sps: int = SPS, span: int = SPAN) -> np.ndarray:
    """Root-raised-cosine filter (unit energy)."""
    n = np.arange(-span * sps // 2, span * sps // 2 + 1)
    t = n / sps
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif np.isclose(abs(ti), 1 / (4 * beta)):
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                          + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) \
                   / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def random_symbols(mod: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-average-power constellation symbols."""
    if mod == "BPSK":
        return rng.choice([-1.0, 1.0], n).astype(complex)
    if mod == "QPSK":
        return (rng.choice([-1, 1], n) + 1j * rng.choice([-1, 1], n)) / np.sqrt(2)
    if mod == "16QAM":
        lv = np.array([-3, -1, 1, 3])
        return (rng.choice(lv, n) + 1j * rng.choice(lv, n)) / np.sqrt(10)
    raise ValueError(f"unknown modulation {mod}")


def generate(mod: str, n_symbols: int, snr_db: float, rng: np.random.Generator):
    """Return (waveform at SPS samples/symbol, symbol-rate matched-filter output).

    The symbol-rate output assumes ideal timing recovery (documented assumption); carrier phase is random
    and NOT corrected, so classifiers must be rotation-invariant.
    """
    h = rrc_taps()
    sym = random_symbols(mod, n_symbols, rng)
    up = np.zeros(n_symbols * SPS, dtype=complex)
    up[::SPS] = sym
    tx = np.convolve(up, h)                           # pulse shaping
    tx *= np.exp(1j * rng.uniform(0, 2 * np.pi))      # unknown carrier phase
    p_sig = np.mean(np.abs(tx) ** 2)
    tx /= np.sqrt(p_sig)
    noise_var = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_var / 2) * (rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape))
    rx = tx + noise
    mf = np.convolve(rx, h)                           # matched filter
    delay = len(h) - 1                                # TX + MF filter delays
    sym_rate = mf[delay: delay + n_symbols * SPS: SPS]
    sym_rate = sym_rate / np.sqrt(np.mean(np.abs(sym_rate) ** 2))  # power normalisation (receiver AGC)
    return rx, sym_rate
