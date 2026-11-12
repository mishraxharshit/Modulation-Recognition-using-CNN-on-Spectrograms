import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from amr import dataset, features  # noqa: E402
from amr.signals import MODULATIONS, SPS, generate, random_symbols, rrc_taps  # noqa: E402


def test_rrc_unit_energy_and_symmetry():
    h = rrc_taps()
    assert np.sum(h ** 2) == pytest.approx(1.0)
    assert np.allclose(h, h[::-1])


@pytest.mark.parametrize("mod", MODULATIONS)
def test_symbols_have_unit_average_power(mod):
    s = random_symbols(mod, 200_000, np.random.default_rng(0))
    assert np.mean(np.abs(s) ** 2) == pytest.approx(1.0, abs=0.02)


def test_bpsk_is_real_and_constellation_sizes():
    rng = np.random.default_rng(0)
    assert np.allclose(random_symbols("BPSK", 100, rng).imag, 0)
    assert len(np.unique(np.round(random_symbols("QPSK", 5000, rng), 6))) == 4
    assert len(np.unique(np.round(random_symbols("16QAM", 20000, rng), 6))) == 16


@pytest.mark.parametrize("snr", [0, 10, 20])
def test_generate_noise_level_matches_snr(snr):
    rng = np.random.default_rng(1)
    clean_ref = generate("QPSK", 2000, 80, np.random.default_rng(1))[0]   # ~noiseless, same RNG stream
    noisy = generate("QPSK", 2000, snr, np.random.default_rng(1))[0]
    meas = 10 * np.log10(np.mean(np.abs(clean_ref) ** 2) / np.mean(np.abs(noisy - clean_ref) ** 2))
    assert meas == pytest.approx(snr, abs=0.7)


def test_symbol_rate_output_length_and_power():
    rx, sym = generate("16QAM", 300, 20, np.random.default_rng(2))
    assert len(sym) == 300
    assert np.mean(np.abs(sym) ** 2) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("mod,c40,c42", [("BPSK", 2.0, -2.0), ("QPSK", 1.0, -1.0), ("16QAM", 0.68, -0.68)])
def test_cumulants_match_theory(mod, c40, c42):
    s = random_symbols(mod, 400_000, np.random.default_rng(3))
    f = features.cumulant_features(s)
    assert f[1] == pytest.approx(c40, abs=0.03)       # |C40|
    assert f[2] == pytest.approx(c42, abs=0.03)       # C42


def test_dataset_shapes_and_balance():
    d = dataset.build(2, seed=0)
    n = 2 * len(MODULATIONS) * len(dataset.SNRS_DB)
    assert d["y"].shape == (n,) and d["spec"].shape[0] == n and d["iq"].shape == (n, 2, dataset.N_SYMBOLS)
    assert d["const"].shape == (n, 1, features.CONST_BINS, features.CONST_BINS)
    assert np.bincount(d["y"]).tolist() == [n // 3] * 3
