from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def fit_predict(model: nn.Module, Xtr, ytr, Xva, yva, Xte, epochs: int = 12, lr: float = 2e-3,
                batch: int = 64, seed: int = 0):
    """Train with Adam, keep the epoch with the best validation accuracy, return test predictions."""
    torch.manual_seed(seed)
    Xtr, ytr, Xva, yva, Xte = (torch.as_tensor(a) for a in (Xtr, ytr, Xva, yva, Xte))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        va = _accuracy(model, Xva, yva)
        hist.append((ep, tot / len(Xtr), va))
        if va > best:
            best, best_state = va, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return _predict(model, Xte), hist, best


@torch.no_grad()
def _predict(model, X, batch: int = 256):
    model.eval()
    return torch.cat([model(X[i:i + batch]).argmax(1) for i in range(0, len(X), batch)]).numpy()


def _accuracy(model, X, y):
    return float((_predict(model, X) == y.numpy()).mean())
