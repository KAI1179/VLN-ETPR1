"""Bounded CPU-only LOSO fits for the D8 goal-proximity probe."""
from __future__ import annotations

import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def _fit_fold(x, y, scene, held, label):
    test = scene == held
    train = ~test
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, solver="lbfgs", max_iter=5000, multi_class="multinomial"),
    )
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(x[train].astype(np.float64), y[train])
        predicted = estimator.predict(x[test].astype(np.float64))
    messages = []
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            message = str(warning.message)
            messages.append(message)
            print(f"WARNING {label} scene={held}: {message}", flush=True)
        else:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    n_iter = estimator.named_steps["logisticregression"].n_iter_.tolist()
    converged = not messages and max(n_iter) < 5000
    fold = {
        "scene": str(held),
        "n": int(test.sum()),
        "n_iter": n_iter,
        "converged": converged,
        "warnings": messages,
        "accuracy": float(accuracy_score(y[test], predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], predicted)),
        "spearman": (
            float(spearmanr(y[test], predicted).correlation)
            if np.unique(predicted).size > 1 and np.unique(y[test]).size > 1
            else None
        ),
        "confusion": confusion_matrix(y[test], predicted, labels=np.arange(5)).tolist(),
    }
    print(f"{label} scene={held} n_iter={n_iter} converged={converged}", flush=True)
    return np.flatnonzero(test), predicted, int(np.bincount(y[train], minlength=5).argmax()), fold


def readout(x, y, scene, workers=4, label=""):
    """Fit each held-out scene once, returning pooled predictions and fold records."""
    if not 1 <= workers <= 4:
        raise ValueError("D8 fits require between 1 and 4 worker processes")
    held_scenes = np.unique(scene)
    workers = min(workers, len(held_scenes))
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable"):
            mem[key] = int(value.split()[0]) * 1024
    rss = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmRSS:")
    )
    # Parent queue copies plus float64 subsets/scaler/solver working arrays per worker.
    estimate = workers * (x.size * 8 * 8 + y.nbytes + scene.nbytes + 256 * 1024**2)
    if estimate > mem["MemAvailable"] or rss + estimate > mem["MemTotal"] // 2:
        raise MemoryError(
            f"D8 fit memory estimate exceeds budget: parent_rss={rss}, "
            f"workers_estimate={estimate}, available={mem['MemAvailable']}, "
            f"half_ram={mem['MemTotal']//2}"
        )
    pred = np.empty_like(y)
    majority = np.empty_like(y)
    folds = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [executor.submit(_fit_fold, x, y, scene, held, label) for held in held_scenes]
        for future in futures:
            indices, predicted, common, fold = future.result()
            pred[indices] = predicted
            majority[indices] = common
            folds.append(fold)
    return pred, majority, folds
