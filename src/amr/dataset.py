"""Builds the dataset once, with every representation derived from the SAME noisy realisation."""
from __future__ import annotations

import numpy as np

from .features import constellation_image, cumulant_features, iq_sequence, spectrogram_image
from .signals import MODULATIONS, generate

SNRS_DB = [-10, -5, 0, 5, 10, 15, 20]
N_SYMBOLS = 512


def build(n_per_class_snr: int, seed: int):
    rng = np.random.default_rng(seed)
    spec, const, iq, cum, y, snr = [], [], [], [], [], []
    for ci, mod in enumerate(MODULATIONS):
        for s in SNRS_DB:
            for _ in range(n_per_class_snr):
                rx, sym = generate(mod, N_SYMBOLS, s, rng)
                spec.append(spectrogram_image(rx)); const.append(constellation_image(sym))
                iq.append(iq_sequence(sym)); cum.append(cumulant_features(sym))
                y.append(ci); snr.append(s)
    out = dict(spec=np.stack(spec)[:, None], const=np.stack(const)[:, None], iq=np.stack(iq),
               cum=np.stack(cum), y=np.array(y), snr=np.array(snr))
    perm = rng.permutation(len(out["y"]))
    return {k: v[perm] for k, v in out.items()}
