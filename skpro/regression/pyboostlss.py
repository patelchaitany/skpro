"""Interface for pyboostlss probabilistic regressor."""

from skpro.regression.base import BaseProbaRegressor


class PyBoostLSS(BaseProbaRegressor):
    """Interface to pyboostlss regressor from the pyboostlss package.

    Direct interface to ``PyBoostLSS`` from ``pyboostlss`` package by ``StatMixedML``.
    Py-BoostLSS is a GPU-accelerated probabilistic extension of Py-Boost (SketchBoost)
    for multivariate distributional regression.

    Parameters
    ----------
    dist : str, optional, default="MVN"
        Form of predictive distribution.

        Valid options are:

        * "MVN": Multivariate Normal (Cholesky decomposition).
        * "MVN_LRA": Multivariate Normal (Low-Rank Approximation).
        * "MVT": Multivariate Student-T (Cholesky decomposition).
        * "Dirichlet": Dirichlet distribution.

    rank : int, optional, default=2
        Rank parameter for MVN_LRA (Low-Rank Approximation).
        Only used when ``dist="MVN_LRA"``.

    ntrees : int, optional, default=100
        Maximum number of boosting trees.

    lr : float, optional, default=0.05
        Learning rate.

    max_depth : int, optional, default=6
        Maximum tree depth. Setting to large values (>12) may cause
        out-of-memory for wide datasets.

    min_data_in_leaf : int, optional, default=10
        Minimal leaf size.

    lambda_l2 : float, optional, default=1
        L2 leaf regularization.

    colsample : float, optional, default=1.0
        Subsample of columns to construct trees.

    subsample : float, optional, default=1.0
        Subsample of rows to construct trees.

    min_gain_to_split : float, optional, default=0
        Minimal gain to split.

    gd_steps : int, optional, default=1
        Number of gradient steps.

    quantization : str, optional, default="Quantile"
        Method for quantization. One of ``"Quantile"``, ``"Uniform"``,
        ``"Uniquant"``.

    quant_sample : int, optional, default=2000000
        Subsample to quantize features.

    max_bin : int, optional, default=256
        Maximum number of bins to quantize features. Must be in [2, 256].

    min_data_in_bin : int, optional, default=3
        Minimal bin size.

    es : int, optional, default=100
        Early stopping rounds. If 0, no early stopping.

    seed : int, optional, default=123
        Random state.

    verbose : int, optional, default=10
        Verbosity frequency.

    sketch_outputs : int, optional, default=1
        Number of outputs to keep in sketching.

    sketch_method : str, optional, default="proj"
        Sketching strategy. Options: ``"topk"``, ``"rand"``, ``"proj"``.

    use_hess : bool, optional, default=True
        Use hessians in multioutput training.

    n_samples : int, optional, default=1000
        Number of samples drawn for Dirichlet distribution prediction.
        Only used when ``dist="Dirichlet"``.

    n_trials : int, optional, default=0
        Number of Optuna hyperparameter optimization trials.
        If 0, no tuning is done and user-specified parameters are used.
        If None, there is no limitation on the number of trials.

    max_minutes : int, optional, default=10
        Time budget in minutes for hyperparameter optimization.
        Ignored if ``n_trials=0``.

    hp_seed : int, optional, default=None
        Seed for the random number generator used in the Bayesian
        hyperparameter search. Ignored if ``n_trials=0``.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["StatMixedML", "chpatel"],
        # StatMixedML for the original pyboostlss package
        "maintainers": ["chpatel"],
        "python_dependencies": ["pyboostlss"],
        #
        # estimator tags
        # --------------
        "capability:multioutput": True,
        "capability:missing": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
        # CI and test flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(
        self,
        dist="MVN",
        rank=2,
        ntrees=100,
        lr=0.05,
        max_depth=6,
        min_data_in_leaf=10,
        lambda_l2=1,
        colsample=1.0,
        subsample=1.0,
        min_gain_to_split=0,
        gd_steps=1,
        quantization="Quantile",
        quant_sample=2000000,
        max_bin=256,
        min_data_in_bin=3,
        es=100,
        seed=123,
        verbose=10,
        sketch_outputs=1,
        sketch_method="proj",
        use_hess=True,
        n_samples=1000,
        n_trials=0,
        max_minutes=10,
        hp_seed=None,
    ):
        self.dist = dist
        self.rank = rank
        self.ntrees = ntrees
        self.lr = lr
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_l2 = lambda_l2
        self.colsample = colsample
        self.subsample = subsample
        self.min_gain_to_split = min_gain_to_split
        self.gd_steps = gd_steps
        self.quantization = quantization
        self.quant_sample = quant_sample
        self.max_bin = max_bin
        self.min_data_in_bin = min_data_in_bin
        self.es = es
        self.seed = seed
        self.verbose = verbose
        self.sketch_outputs = sketch_outputs
        self.sketch_method = sketch_method
        self.use_hess = use_hess
        self.n_samples = n_samples
        self.n_trials = n_trials
        self.max_minutes = max_minutes
        self.hp_seed = hp_seed

        super().__init__()

        if n_trials is not None and n_trials != 0:
            self.set_tags(**{"python_dependencies": ["pyboostlss", "optuna"]})

    def _get_pyblss_distr(self, dist, D):
        """Get pyboostlss distribution object from string.

        Parameters
        ----------
        dist : str
            Distribution name, as in self.dist
        D : int
            Number of target dimensions

        Returns
        -------
        Distribution instance from pyboostlss
        """
        import importlib

        DIST_CLASS_MAP = {
            "MVN": ("pyboostlss.distributions.MVN", "MVN"),
            "MVN_LRA": ("pyboostlss.distributions.MVN_LRA", "MVN_LRA"),
            "MVT": ("pyboostlss.distributions.MVT", "MVT"),
            "Dirichlet": ("pyboostlss.distributions.DIRICHLET", "DIRICHLET"),
        }

        if dist not in DIST_CLASS_MAP:
            raise ValueError(
                f"Unknown distribution '{dist}'. "
                f"Valid options: {list(DIST_CLASS_MAP.keys())}"
            )

        module_str, class_name = DIST_CLASS_MAP[dist]
        module = importlib.import_module(module_str)
        distr_cls = getattr(module, class_name)

        if dist == "MVN_LRA":
            return distr_cls(r=self.rank, D=D)
        else:
            return distr_cls(D=D)

    def _get_skpro_distr(self, dist):
        """Get skpro distribution class from distribution name string.

        Parameters
        ----------
        dist : str
            Distribution name, as in self.dist

        Returns
        -------
        skpro distribution class
        """
        if dist == "Dirichlet":
            from skpro.distributions.empirical import Empirical

            return Empirical

        from skpro.distributions import Normal, TDistribution

        DIST_MAP = {
            "MVN": Normal,
            "MVN_LRA": Normal,
            "MVT": TDistribution,
        }

        return DIST_MAP[dist]

    def _get_skpro_val_dict(self, dist, pred_params, D):
        """Convert pyboostlss predicted parameters to skpro distribution kwargs.

        Parameters
        ----------
        dist : str
            Distribution name, as in self.dist
        pred_params : pd.DataFrame
            DataFrame of predicted parameters from pyboostlss predict
        D : int
            Number of target dimensions

        Returns
        -------
        dict
            Keyword arguments for skpro distribution constructor
        """
        import numpy as np

        if dist in ("MVN", "MVN_LRA"):
            loc_cols = [f"location_{i+1}" for i in range(D)]
            scale_cols = [f"scale_{i+1}" for i in range(D)]
            mu = pred_params[loc_cols].values
            sigma = pred_params[scale_cols].values
            return {"mu": mu, "sigma": sigma}

        elif dist == "MVT":
            loc_cols = [f"location_{i+1}" for i in range(D)]
            scale_cols = [f"scale_{i+1}" for i in range(D)]
            mu = pred_params[loc_cols].values
            sigma = pred_params[scale_cols].values
            df_vals = pred_params["df"].values.reshape(-1, 1)
            df_broadcast = np.broadcast_to(df_vals, (df_vals.shape[0], D)).copy()
            return {"mu": mu, "sigma": sigma, "df": df_broadcast}

        else:
            raise ValueError(f"Unsupported distribution for parameter mapping: {dist}")

    def _get_train_params(self):
        """Collect training parameters as a dict for pyboostlss train call.

        Returns
        -------
        dict
            Training parameters for PyBoostLSS.train()
        """
        return {
            "ntrees": self.ntrees,
            "lr": self.lr,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_l2": self.lambda_l2,
            "colsample": self.colsample,
            "subsample": self.subsample,
            "min_gain_to_split": self.min_gain_to_split,
            "gd_steps": self.gd_steps,
            "quantization": self.quantization,
            "quant_sample": self.quant_sample,
            "max_bin": self.max_bin,
            "min_data_in_bin": self.min_data_in_bin,
            "es": self.es,
            "seed": self.seed,
            "verbose": self.verbose,
            "sketch_outputs": self.sketch_outputs,
            "sketch_method": self.sketch_method,
            "use_hess": self.use_hess,
        }

    def _fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        from pyboostlss.model import PyBoostLSS

        self._y_cols = y.columns
        self._D = y.shape[1]

        pyblss_distr = self._get_pyblss_distr(self.dist, self._D)
        self._pyblss_distr = pyblss_distr

        pyblss = PyBoostLSS(pyblss_distr)

        dtrain = {"X": X.values, "y": y.values}

        train_params = self._get_train_params()

        if self.n_trials is not None and self.n_trials != 0:
            opt_params = self._hyper_opt(pyblss, dtrain)
            opt_rounds = opt_params.pop("opt_rounds", self.ntrees)
            train_params.update(opt_params)
            train_params["ntrees"] = opt_rounds

        self._model = pyblss.train(dtrain=dtrain, **train_params)
        self._pyblss = pyblss

        return self

    def _hyper_opt(self, pyblss, dtrain):
        """Run hyperparameter optimization using Optuna.

        Parameters
        ----------
        pyblss : pyboostlss.model.PyBoostLSS
            PyBoostLSS model instance
        dtrain : dict
            Training data in Py-BoostLSS format ``{"X": array, "y": array}``

        Returns
        -------
        opt_params : dict
            Optimized hyperparameters
        """
        import numpy as np

        param_dict = {
            "lr": [1e-5, 1.0],
            "max_depth": [1, 10],
            "sketch_outputs": [1, max(1, self._D)],
            "lambda_l2": [1e-8, 100.0],
            "colsample": [0.2, 1.0],
            "subsample": [0.2, 1.0],
            "min_gain_to_split": [0.0, 40.0],
        }

        n = dtrain["X"].shape[0]
        split = int(0.8 * n)
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n)
        train_idx, eval_idx = indices[:split], indices[split:]

        dtrain_split = {"X": dtrain["X"][train_idx], "y": dtrain["y"][train_idx]}
        eval_sets = [{"X": dtrain["X"][eval_idx], "y": dtrain["y"][eval_idx]}]

        opt_params = pyblss.hyper_opt(
            params=param_dict,
            dtrain=dtrain_split,
            eval_sets=eval_sets,
            ntrees=self.ntrees,
            lr=self.lr,
            min_gain_to_split=self.min_gain_to_split,
            lambda_l2=self.lambda_l2,
            gd_steps=self.gd_steps,
            max_depth=self.max_depth,
            min_data_in_leaf=self.min_data_in_leaf,
            colsample=self.colsample,
            subsample=self.subsample,
            quantization=self.quantization,
            quant_sample=self.quant_sample,
            max_bin=self.max_bin,
            min_data_in_bin=self.min_data_in_bin,
            es=self.es,
            seed=self.seed,
            hp_seed=self.hp_seed,
            verbose=int(1e04),
            sketch_outputs=self.sketch_outputs,
            sketch_method=self.sketch_method,
            use_hess=self.use_hess,
            max_minutes=self.max_minutes,
            n_trials=self.n_trials,
            silence=True,
        )

        return opt_params.copy()

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y_pred : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        import numpy as np
        import pandas as pd

        index = X.index
        columns = self._y_cols
        D = self._D

        if self.dist == "Dirichlet":
            y_samples = self._pyblss_distr.predict(
                model=self._model,
                X_test=X.values,
                pred_type="samples",
                n_samples=self.n_samples,
            )
            # y_samples shape: (n_samples, n_obs, n_target)
            n_spl = y_samples.shape[0]
            n_obs = y_samples.shape[1]

            sample_idx = np.repeat(np.arange(n_spl), n_obs)
            obs_idx = np.tile(index, n_spl)
            mi = pd.MultiIndex.from_arrays([sample_idx, obs_idx])

            spl_df = pd.DataFrame(
                y_samples.reshape(-1, D),
                index=mi,
                columns=columns,
            )

            from skpro.distributions.empirical import Empirical

            return Empirical(spl=spl_df)

        pred_params = self._pyblss_distr.predict(
            model=self._model,
            X_test=X.values,
            pred_type="parameters",
        )

        skpro_distr_cls = self._get_skpro_distr(self.dist)
        skpro_vals = self._get_skpro_val_dict(self.dist, pred_params, D)

        return skpro_distr_cls(**skpro_vals, index=index, columns=columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params0 = {"dist": "MVN", "ntrees": 10, "n_trials": 0, "verbose": 0}
        params1 = {"dist": "MVT", "ntrees": 10, "n_trials": 0, "verbose": 0}
        params2 = {
            "dist": "MVN_LRA",
            "rank": 2,
            "ntrees": 10,
            "n_trials": 0,
            "verbose": 0,
        }
        params3 = {
            "dist": "Dirichlet",
            "ntrees": 10,
            "n_trials": 0,
            "n_samples": 100,
            "verbose": 0,
        }

        return [params0, params1, params2, params3]
