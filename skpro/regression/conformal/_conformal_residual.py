"""Online conformal regressor based on residual quantiles (Jackknife+)."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["patelchaitany"]
__all__ = ["ConformalResidualRegressor"]

from collections import deque
from copy import deepcopy

import numpy as np
import pandas as pd

from skpro.regression.adapters._coerce import coerce_to_skpro_regressor
from skpro.regression.base import BaseProbaRegressor


class ConformalResidualRegressor(BaseProbaRegressor):
    """Online conformal regressor using residual quantiles (Jackknife+).

    Wraps any skpro regressor and produces prediction intervals from
    online residual tracking. Residuals are computed before the inner
    model learns each observation, preserving the out-of-sample guarantee
    from the Jackknife+ method.

    Works with any inner regressor — online estimators (River adapters,
    ``OndilOnlineGamlss``) get true incremental learning, while batch
    estimators work with no-op ``update`` calls.

    On ``predict_interval``, intervals are ``point_prediction ± quantile(residuals)``
    at the requested coverage level.

    On ``predict_proba``, an ``Empirical`` distribution is built from
    ``point_prediction + each_stored_residual``, following the
    ``BootstrapRegressor`` pattern.

    Parameters
    ----------
    estimator : skpro regressor or River estimator
        Inner regressor to wrap. If a River estimator is passed, it is
        auto-coerced via ``coerce_to_skpro_regressor``.
    window_size : int or None, default=None
        Number of most recent residuals to keep. If ``None``, all residuals
        are stored (growing memory). Recommended for non-stationary data
        where the model's error profile changes over time.

    Attributes
    ----------
    estimator_ : skpro regressor
        Fitted (deep-copied) inner regressor.
    residuals_ : collections.deque
        Stored residuals from fit and update calls.

    See Also
    --------
    MapieSplitConformalRegressor :
        Split conformal prediction using MAPIE (batch, not online).
    BootstrapRegressor :
        Empirical distribution from bootstrap point predictions.

    References
    ----------
    .. [1] Barber, R.F., Candes, E.J., Ramdas, A. and Tibshirani, R.J., 2021.
       Predictive inference with the jackknife+. The Annals of Statistics,
       49(1), pp.486-507.

    Examples
    --------
    >>> from skpro.regression.conformal import ConformalResidualRegressor
    >>> from skpro.regression.residual import ResidualDouble
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> import pandas as pd
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X = X.iloc[:50]
    >>> y = pd.DataFrame(y.iloc[:50])
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=42
    ... )
    >>> reg = ConformalResidualRegressor(
    ...     ResidualDouble(LinearRegression()), window_size=30
    ... )
    >>> reg.fit(X_train, y_train)
    ConformalResidualRegressor(...)
    >>> intervals = reg.predict_interval(X_test, coverage=[0.9])
    """

    _tags = {
        "authors": ["patelchaitany"],
        "maintainers": ["patelchaitany", "fkiraly"],
        "capability:update": True,
        "capability:pred_int": True,
        "capability:multioutput": False,
        "capability:missing": False,
    }

    def __init__(self, estimator, window_size=None):
        self.estimator = estimator
        self.window_size = window_size

        if window_size is not None and window_size < 1:
            raise ValueError(
                f"window_size must be a positive integer or None, "
                f"got {window_size}."
            )

        super().__init__()

    def _fit(self, X, y, C=None):
        estimator = coerce_to_skpro_regressor(self.estimator)
        self.estimator_ = deepcopy(estimator)
        self.residuals_ = deque(maxlen=self.window_size)
        self._y_cols = y.columns.tolist()

        # fit inner on the first row so it can predict
        self.estimator_.fit(X.iloc[[0]], y.iloc[[0]])

        # row-by-row from the second row onwards
        if len(X) > 1:
            self._collect_residuals(X.iloc[1:], y.iloc[1:])
        return self

    def _update(self, X, y, C=None):
        self._collect_residuals(X, y)
        return self

    def _collect_residuals(self, X, y):
        """Row-by-row predict-then-learn loop to collect residuals.

        For each row: predict before the model sees it (out-of-sample),
        record the residual, then update the model.
        """
        y_col = y.iloc[:, 0]

        for i in range(len(X)):
            Xi = X.iloc[[i]]
            yi = y.iloc[[i]]

            y_pred_i = self.estimator_.predict(Xi)
            residual = y_col.iloc[i] - y_pred_i.iloc[0, 0]
            self.residuals_.append(residual)

            self.estimator_.update(Xi, yi)

    def _predict(self, X):
        return self.estimator_.predict(X)

    def _predict_interval(self, X, coverage):
        y_pred = self.estimator_.predict(X)
        residuals = np.array(self.residuals_)

        if len(residuals) == 0:
            raise ValueError(
                "No residuals have been collected yet. "
                "Call fit or update with training data first."
            )

        columns = pd.MultiIndex.from_product(
            [self._y_cols, coverage, ["lower", "upper"]]
        )
        result = np.empty((len(X), len(self._y_cols) * len(coverage) * 2))

        col_idx = 0
        for col in self._y_cols:
            point = y_pred[col].values
            for c in coverage:
                alpha = (1 - c) / 2
                lower_q = np.quantile(residuals, alpha)
                upper_q = np.quantile(residuals, 1 - alpha)
                result[:, col_idx] = point + lower_q
                result[:, col_idx + 1] = point + upper_q
                col_idx += 2

        return pd.DataFrame(result, index=X.index, columns=columns)

    def _predict_proba(self, X):
        from skpro.distributions.empirical import Empirical

        y_pred = self.estimator_.predict(X)
        residuals = np.array(self.residuals_)

        if len(residuals) == 0:
            raise ValueError(
                "No residuals have been collected yet. "
                "Call fit or update with training data first."
            )

        samples = []
        for r in residuals:
            sample = y_pred.copy()
            sample.iloc[:, 0] = sample.iloc[:, 0] + r
            samples.append(sample)

        y_pred_df = pd.concat(samples, axis=0, keys=range(len(samples)))
        return Empirical(y_pred_df)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        from sklearn.linear_model import LinearRegression

        from skpro.regression.residual import ResidualDouble

        params1 = {"estimator": ResidualDouble(LinearRegression())}
        params2 = {
            "estimator": ResidualDouble(LinearRegression()),
            "window_size": 20,
        }
        return [params1, params2]
