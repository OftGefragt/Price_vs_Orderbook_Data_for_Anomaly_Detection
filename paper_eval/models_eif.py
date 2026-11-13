"""GA-configured Extended Isolation Forest with optional one-class SVM refinement.

Uses the repository EIF implementation when importable. Falls back to
sklearn IsolationForest with the same tree and subsample settings if the
local eif package cannot be loaded.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


REPO_ROOT = Path(__file__).resolve().parents[1]
EIF_DIR = (
    REPO_ROOT
    / "Level 2 Analysis"
    / "GA-optimized-EIF-SVM-for-anomaly-detection-main"
    / "eif"
)


def _try_load_eif():
    if str(EIF_DIR) not in sys.path:
        sys.path.insert(0, str(EIF_DIR.parent))
    try:
        from eif.eif_class import iForest  # type: ignore

        return iForest
    except Exception:
        return None


class EIFSVMDetector:
    def __init__(
        self,
        ntrees: int = 235,
        sample_size: int = 1691,
        extension_level: int = 4,
        contamination: float = 0.062,
        random_state: int = 42,
    ):
        self.ntrees = ntrees
        self.sample_size = sample_size
        self.extension_level = extension_level
        self.contamination = contamination
        self.random_state = random_state
        self._forest = None
        self._svm = OneClassSVM(kernel="rbf", gamma="scale", nu=max(contamination, 0.01))
        self._backend = "unknown"
        self.train_seconds = 0.0

    def fit(self, x_train: np.ndarray) -> "EIFSVMDetector":
        t0 = time.perf_counter()
        n = len(x_train)
        sample_size = min(self.sample_size, n)
        iForest = _try_load_eif()
        if iForest is not None:
            # Extension level must be < feature dimension.
            ext = min(self.extension_level, x_train.shape[1] - 1)
            self._forest = iForest(
                x_train,
                ntrees=self.ntrees,
                sample_size=sample_size,
                ExtensionLevel=ext,
            )
            eif_scores = self._forest.compute_paths(X_in=x_train)
            self._backend = "eif"
        else:
            self._forest = IsolationForest(
                n_estimators=self.ntrees,
                max_samples=sample_size,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._forest.fit(x_train)
            # Higher score => more anomalous (invert sklearn score_samples).
            eif_scores = -self._forest.score_samples(x_train)
            self._backend = "sklearn_iforest"
        self._svm.fit(eif_scores.reshape(-1, 1))
        self.train_seconds = time.perf_counter() - t0
        return self

    def score(self, x_test: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        if self._backend == "eif":
            scores = self._forest.compute_paths(X_in=x_test)
        else:
            scores = -self._forest.score_samples(x_test)
        # Keep continuous EIF/IF scores as the primary ranking signal.
        # SVM is retained for compatibility with the original pipeline design.
        _ = self._svm.predict(scores.reshape(-1, 1))
        infer_seconds = time.perf_counter() - t0
        return np.asarray(scores, dtype=np.float64), infer_seconds
