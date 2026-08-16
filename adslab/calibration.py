"""Post-hoc calibrators -- the Week 3 exercise, with the interface fixed in advance.

All three follow the sklearn shape (``fit`` on a held-out set, ``transform`` on new
predictions) so they can be dropped into any week's pipeline without rewiring.

The rule that makes them honest: **fit the calibrator on validation predictions, never
on training predictions**. A model is overconfident on data it memorised, so a
calibrator fitted there learns to correct a distortion that does not exist at serving
time, and makes test calibration worse.
"""
from __future__ import annotations

import numpy as np

from .metrics import TodoError


class Calibrator:
    """Base class. Subclasses implement ``fit`` and ``transform``."""

    def fit(self, y_prob, y_true) -> "Calibrator":
        raise NotImplementedError

    def transform(self, y_prob) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, y_prob, y_true) -> np.ndarray:
        return self.fit(y_prob, y_true).transform(y_prob)


class PlattScaler(Calibrator):
    """Logistic regression on the *logit* of the prediction: ``sigmoid(a * logit(p) + b)``.

    Week 3. Two parameters, so it is stable on small validation sets, but it can only
    apply a smooth monotone squeeze -- it cannot fix a kink.

    Fit on ``logit(p)``, not on ``p``. Fitting on the raw probability is a common and
    quiet bug: it works, it improves the number a little, and it is not Platt scaling.
    """

    def __init__(self) -> None:
        self.a: float | None = None
        self.b: float | None = None

    def fit(self, y_prob, y_true) -> "PlattScaler":
        raise TodoError(3, "PlattScaler.fit")

    def transform(self, y_prob) -> np.ndarray:
        raise TodoError(3, "PlattScaler.transform")


class IsotonicCalibrator(Calibrator):
    """Non-parametric monotone fit -- a step function, free to bend anywhere.

    Week 3. Strictly more flexible than Platt and strictly more prone to overfitting a
    small validation set. It also produces *ties*: many inputs map to one output, which
    destroys fine-grained ranking. Check AUC before and after; if isotonic calibration
    costs you AUC, that is why.
    """

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self.out_of_bounds = out_of_bounds
        self._iso = None

    def fit(self, y_prob, y_true) -> "IsotonicCalibrator":
        raise TodoError(3, "IsotonicCalibrator.fit")

    def transform(self, y_prob) -> np.ndarray:
        raise TodoError(3, "IsotonicCalibrator.transform")


class SamplingRateCorrector(Calibrator):
    """Undo a *known* negative-downsampling rate in closed form.

    Week 3, and the one calibrator here that needs no validation data at all -- which is
    precisely why it is the right tool when you know the sampling rate and the wrong one
    when you are guessing it.

    If negatives were kept with probability ``w`` and positives all kept, a model trained
    on the sample predicts ``q``, and the unbiased estimate on the true population is

        p = q / (q + (1 - q) / w)

    Week 3's stress test uses the mirror-image case (positives dropped, negatives kept),
    which is also the shape of privacy-driven conversion loss in Week 5. Derive that
    version -- do not assume the formula above is symmetric, because it is not.
    """

    def __init__(self, negative_keep_rate: float) -> None:
        if not 0 < negative_keep_rate <= 1:
            raise ValueError("negative_keep_rate must be in (0, 1]")
        self.w = float(negative_keep_rate)

    def fit(self, y_prob=None, y_true=None) -> "SamplingRateCorrector":
        return self  # nothing to learn: the rate is known by construction

    def transform(self, y_prob) -> np.ndarray:
        raise TodoError(3, "SamplingRateCorrector.transform")
