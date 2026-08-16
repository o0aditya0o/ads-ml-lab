"""Feature hashing -- the encoding every large-scale ads system actually uses.

An honest caveat specific to this dataset, worth knowing before Week 1 draws the wrong
conclusion: Criteo's ``cat1..cat9`` are stored as huge integer hashes, but their *distinct
counts* are modest -- 9, 70, 1829, 21, 51, 30, 57196, 11, 30, or about 59k values in
total. You could one-hot that on a laptop. Hashing is not strictly necessary here.

It is still the right thing to practise, for the reason it exists in production: the
vocabulary is not known in advance and changes daily, and a stateless encoder needs no
vocabulary at all. Treat Week 1's ``n_bits`` sweep as an experiment in *what collisions
cost you*, run on a dataset small enough that you can also compute the no-collision
answer and compare. That comparison is not available at real scale, which is exactly why
it is worth doing once here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


class HashingEncoder:
    """Hash categorical columns into a shared sparse space of ``2**n_bits`` buckets.

    Each column is salted with its own name, so ``cat1=5`` and ``cat2=5`` land in
    different buckets -- without the salt, unrelated features collide systematically and
    you lose accuracy for no reason.

    Parameters
    ----------
    n_bits
        log2 of the output width. At 2**18 (262k buckets) the attribution dataset's ~59k
        distinct values collide at roughly 2%; drop to 2**12 and the encoder is
        destroying most of the signal. Sweep it -- the accuracy-vs-width curve, with
        :meth:`collision_report` beside it, is one of the more instructive plots in
        Week 1.
    """

    def __init__(self, columns: list[str], n_bits: int = 18) -> None:
        self.columns = list(columns)
        self.n_bits = int(n_bits)
        self.n_features = 1 << self.n_bits

    def _bucket(self, col: str, values: np.ndarray) -> np.ndarray:
        salt = np.uint64(abs(hash(col)) & 0xFFFFFFFF)
        v = values.astype("uint64", copy=False)
        # splitmix64-style avalanche: cheap, vectorised, and mixes the low bits that
        # Criteo's ids do not vary much in.
        x = (v + salt) * np.uint64(0x9E3779B97F4A7C15)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        return (x % np.uint64(self.n_features)).astype(np.int64)

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        """One row per input row, ``len(self.columns)`` non-zeros in each."""
        n = len(df)
        cols = np.empty(n * len(self.columns), dtype=np.int64)
        for i, c in enumerate(self.columns):
            cols[i * n:(i + 1) * n] = self._bucket(c, df[c].to_numpy())
        rows = np.tile(np.arange(n, dtype=np.int64), len(self.columns))
        data = np.ones(n * len(self.columns), dtype=np.float32)
        m = sparse.csr_matrix((data, (rows, cols)), shape=(n, self.n_features))
        m.sum_duplicates()  # a row can hash two features to one bucket; keep it a count
        return m

    fit_transform = transform  # stateless by construction -- that is the whole appeal

    def fit(self, df: pd.DataFrame, y=None) -> "HashingEncoder":
        return self

    def collision_report(self, df: pd.DataFrame) -> dict[str, float]:
        """How much information the hash is destroying at this width.

        Run it before you blame the model: if ``occupancy`` is near 1.0 every bucket
        holds several distinct values and no linear model can separate them.
        """
        distinct = sum(df[c].nunique() for c in self.columns)
        used = len(np.unique(np.concatenate(
            [self._bucket(c, df[c].unique()) for c in self.columns])))
        return {
            "distinct_values": float(distinct),
            "buckets": float(self.n_features),
            "load_factor": distinct / self.n_features,
            "occupancy": used / self.n_features,
            "est_collision_rate": 1.0 - used / max(distinct, 1),
        }
