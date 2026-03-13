"""Tests for the PyBoostLSS regressor."""

import numpy as np
import pandas as pd
import pytest

from skpro.regression.pyboostlss import PyBoostLSS
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(PyBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pyboostlss_mvn_basic():
    """Test basic MVN fit and predict with PyBoostLSS."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=5, n_targets=3, random_state=42)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    y = pd.DataFrame(y, columns=[f"y{i}" for i in range(y.shape[1])])

    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    reg = PyBoostLSS(dist="MVN", ntrees=10, n_trials=0, verbose=0)
    reg.fit(X_train, y_train)

    y_pred = reg.predict_proba(X_test)
    assert y_pred.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(PyBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pyboostlss_mvt_basic():
    """Test basic MVT fit and predict with PyBoostLSS."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=5, n_targets=3, random_state=42)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    y = pd.DataFrame(y, columns=[f"y{i}" for i in range(y.shape[1])])

    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    reg = PyBoostLSS(dist="MVT", ntrees=10, n_trials=0, verbose=0)
    reg.fit(X_train, y_train)

    y_pred = reg.predict_proba(X_test)
    assert y_pred.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(PyBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pyboostlss_mvn_lra_basic():
    """Test basic MVN_LRA fit and predict with PyBoostLSS."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=5, n_targets=3, random_state=42)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    y = pd.DataFrame(y, columns=[f"y{i}" for i in range(y.shape[1])])

    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    reg = PyBoostLSS(dist="MVN_LRA", rank=2, ntrees=10, n_trials=0, verbose=0)
    reg.fit(X_train, y_train)

    y_pred = reg.predict_proba(X_test)
    assert y_pred.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(PyBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pyboostlss_dirichlet_basic():
    """Test basic Dirichlet fit and predict with PyBoostLSS."""
    rng = np.random.RandomState(42)

    X = pd.DataFrame(rng.randn(100, 5), columns=[f"x{i}" for i in range(5)])

    # Dirichlet target: rows must sum to 1 and be in (0, 1)
    raw = rng.dirichlet(alpha=[2, 3, 5], size=100)
    y = pd.DataFrame(raw, columns=["y0", "y1", "y2"])

    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    reg = PyBoostLSS(dist="Dirichlet", ntrees=10, n_trials=0, n_samples=50, verbose=0)
    reg.fit(X_train, y_train)

    y_pred = reg.predict_proba(X_test)
    y_mean = y_pred.mean()
    assert y_mean.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(PyBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pyboostlss_train_params():
    """Test that training parameters are correctly passed through."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=5, n_targets=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    y = pd.DataFrame(y, columns=[f"y{i}" for i in range(y.shape[1])])

    reg = PyBoostLSS(
        dist="MVN",
        ntrees=5,
        lr=0.1,
        max_depth=3,
        n_trials=0,
        verbose=0,
    )
    reg.fit(X, y)

    y_pred = reg.predict_proba(X)
    assert y_pred.shape == y.shape
