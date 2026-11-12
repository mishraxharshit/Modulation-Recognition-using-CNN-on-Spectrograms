"""Reproduce all results:  python run_experiments.py   (see README for a fast smoke-run flags)"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent / "src"))
from amr import dataset  # noqa: E402
from amr.models import Cnn1d, Cnn2d  # noqa: E402
from amr.signals import MODULATIONS  # noqa: E402
from amr.train import fit_predict  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
NAMES = {"cumulants_rf": "Cumulants + Random Forest", "spectrogram_cnn": "CNN on spectrogram",
         "constellation_cnn": "CNN on constellation image", "iq_cnn1d": "1-D CNN on symbol-rate I/Q"}


def acc_by_snr(y, pred, snr):
    return {int(s): float((pred[snr == s] == y[snr == s]).mean()) for s in dataset.SNRS_DB}


def main(a):
    t0 = time.time()
    tr = dataset.build(a.n_train, seed=101)   # three independent datasets (different RNG seeds)
    va = dataset.build(a.n_val, seed=202)
    te = dataset.build(a.n_test, seed=303)
    print(f"data built in {time.time() - t0:.0f}s: train={len(tr['y'])} val={len(va['y'])} test={len(te['y'])}", flush=True)

    preds = {k: [] for k in NAMES}
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(tr["cum"], tr["y"])
    preds["cumulants_rf"].append(rf.predict(te["cum"]))

    setups = {"spectrogram_cnn": ("spec", Cnn2d), "constellation_cnn": ("const", Cnn2d), "iq_cnn1d": ("iq", Cnn1d)}
    for name, (key, cls) in setups.items():
        for seed in range(a.seeds):
            t1 = time.time()
            p, hist, best_val = fit_predict(cls(), tr[key], tr["y"], va[key], va["y"], te[key],
                                            epochs=a.epochs, seed=seed)
            preds[name].append(p)
            print(f"{name} seed {seed}: val_acc={best_val:.3f} test_acc={np.mean(p == te['y']):.3f} ({time.time() - t1:.0f}s)", flush=True)

    y, snr = te["y"], te["snr"]
    results = {"config": vars(a), "n_test_per_class_per_snr": a.n_test, "methods": {}}
    for name, plist in preds.items():
        per_seed = [acc_by_snr(y, p, snr) for p in plist]
        mean_curve = {s: float(np.mean([r[s] for r in per_seed])) for s in dataset.SNRS_DB}
        std_curve = {s: float(np.std([r[s] for r in per_seed])) for s in dataset.SNRS_DB}
        overall = [float((p == y).mean()) for p in plist]
        high = [float((p[snr >= 5] == y[snr >= 5]).mean()) for p in plist]
        results["methods"][name] = {"acc_vs_snr_mean": mean_curve, "acc_vs_snr_std": std_curve,
                                    "overall_acc_mean": float(np.mean(overall)), "overall_acc_std": float(np.std(overall)),
                                    "acc_snr_ge_5dB_mean": float(np.mean(high)), "acc_snr_ge_5dB_std": float(np.std(high)),
                                    "n_runs": len(plist)}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in NAMES:
        m = results["methods"][name]
        xs = dataset.SNRS_DB
        ax.errorbar(xs, [m["acc_vs_snr_mean"][s] * 100 for s in xs], yerr=[m["acc_vs_snr_std"][s] * 100 for s in xs],
                    marker="o", capsize=3, label=NAMES[name])
    ax.axhline(100 / 3, color="grey", ls=":", label="chance (33.3%)")
    ax.set(xlabel="SNR [dB]", ylabel="test accuracy [%]", title="Modulation recognition: BPSK / QPSK / 16-QAM", ylim=(0, 102))
    ax.grid(alpha=0.3); ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_snr.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(17, 4.3))
    mask = snr >= 5
    for ax, name in zip(axs, NAMES):
        cm = confusion_matrix(y[mask], preds[name][0][mask], labels=range(3), normalize="true")
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=MODULATIONS, yticklabels=MODULATIONS,
               title=NAMES[name], xlabel="predicted", ylabel="true")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
    fig.suptitle("Confusion matrices, test samples with SNR >= 5 dB (first run)"); fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=150); plt.close(fig)

    # example inputs
    from amr import features
    from amr.signals import generate
    rng = np.random.default_rng(7)
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    for j, mod in enumerate(MODULATIONS):
        rx, sym = generate(mod, dataset.N_SYMBOLS, 15, rng)
        axs[0, j].imshow(features.spectrogram_image(rx), aspect="auto", origin="lower", cmap="viridis")
        axs[0, j].set_title(f"{mod} spectrogram (15 dB)")
        axs[1, j].imshow(features.constellation_image(sym).T, origin="lower", cmap="magma", extent=[-2, 2, -2, 2])
        axs[1, j].set_title(f"{mod} constellation (15 dB)")
    fig.tight_layout(); fig.savefig(OUT / "example_inputs.png", dpi=130); plt.close(fig)

    print(json.dumps({k: {x: round(v[x], 4) for x in ("overall_acc_mean", "acc_snr_ge_5dB_mean")}
                      for k, v in results["methods"].items()}, indent=2))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200); ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=100); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(a)
