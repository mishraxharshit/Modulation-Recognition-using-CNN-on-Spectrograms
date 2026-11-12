"""Input representations: spectrogram image, constellation histogram, symbol-rate I/Q, and cumulants."""
from __future__ import annotations

import numpy as np
from scipy import signal

SPEC_NPERSEG, SPEC_NOVERLAP = 64, 32
CONST_BINS, CONST_RANGE = 32, 2.0


def spectrogram_image(rx: np.ndarray) -> np.ndarray:
    """Log-power STFT, frequency axis centred, standardised to zero mean / unit variance."""
    _, _, S = signal.spectrogram(rx, nperseg=SPEC_NPERSEG, noverlap=SPEC_NOVERLAP,
                                 return_onesided=False, mode="psd", detrend=False)
    S = np.fft.fftshift(S, axes=0)
    L = np.log10(S + 1e-12)
    return ((L - L.mean()) / (L.std() + 1e-8)).astype(np.float32)


def constellation_image(sym: np.ndarray) -> np.ndarray:
    H, _, _ = np.histogram2d(sym.real, sym.imag, bins=CONST_BINS, range=[[-CONST_RANGE, CONST_RANGE]] * 2)
    return (H / (H.max() + 1e-8)).astype(np.float32)


def iq_sequence(sym: np.ndarray) -> np.ndarray:
    return np.stack([sym.real, sym.imag]).astype(np.float32)


def cumulant_features(sym: np.ndarray) -> np.ndarray:
    """Classic higher-order-statistics features (normalised): |C20|, |C40|, C42."""
    c20 = np.mean(sym ** 2)
    c21 = np.mean(np.abs(sym) ** 2)
    c40 = np.mean(sym ** 4) - 3 * c20 ** 2
    c42 = np.mean(np.abs(sym) ** 4) - np.abs(c20) ** 2 - 2 * c21 ** 2
    return np.array([np.abs(c20) / c21, np.abs(c40) / c21 ** 2, np.real(c42) / c21 ** 2], dtype=np.float32)
