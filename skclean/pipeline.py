"""
The :mod:`imblearn.pipeline` module implements utilities to build a
composite estimator, as a chain of transforms, samples and estimators.
"""
# Adapted from scikit-learn

# Author: Edouard Duchesnay
#         Gael Varoquaux
#         Virgile Fritsch
#         Alexandre Gramfort
#         Lars Buitinck
#         Christos Aridas
#         Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: BSD

from sklearn import pipeline as skpipeline
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_memory

__all__ = ["Pipeline", "make_pipeline"]


# TODO: Support random_state (?)
class Pipeline(skpipeline.Pipeline):
    """
    Sequentially applies a list of transforms and a final estimator. \
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.
    The transformers in the pipeline can be cached using ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.
    A step's estimator may be replaced entirely by setting the parameter
    with its name to another estimator, or a transformer removed by setting
    it to 'passthrough' or ``None``.

    See `link <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_ \
    for details.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed
    """

    def _fit(self, X, y=None, **fit_params_steps):
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(skpipeline._fit_transform_one)

        conf_score = None
        for (step_idx,
             name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):
            if transformer is None or transformer == 'passthrough':
                with _print_elapsed_time('Pipeline',
                                         self._log_message(step_idx)):
                    continue

            if hasattr(memory, 'location'):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            elif hasattr(memory, 'cachedir'):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            else:
                cloned_transformer = clone(transformer)

            # Fit or load from cache the current transformer
            if hasattr(cloned_transformer, "transform") or hasattr(
                    cloned_transformer, "fit_transform"
            ):
                res, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, X, y, None,
                    message_clsname='Pipeline',
                    message=self._log_message(step_idx),
                    **fit_params_steps[name]
                )
                # This ugly if/else can be removed if Transformers return
                # additional values (i.e. `conf_score`) in dict. Can be
                # appended to `fit_params_steps` dict.
                if type(res) == tuple:
                    if len(res) == 3:
                        X, y, conf_score = res
                    elif len(res) == 2:
                        X, y = res
                else:
                    X = res

            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)

        return X, y, conf_score

    def fit(self, X, y=None, **fit_params):
        """Fit the model.

        Fit all the transforms/samplers one after the other and
        transform/sample the data, then fit the transformed/sampled
        data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        obj : Pipeline
            This estimator.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt, conf_score = self._fit(X, y, **fit_params_steps)

        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last = fit_params_steps[self.steps[-1][0]]
                if conf_score is not None:
                    self._final_estimator.fit(Xt, yt, conf_score=conf_score,
                                              **fit_params_last)
                else:
                    self._final_estimator.fit(Xt, yt, **fit_params_last)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.

        Fits all the transformers/samplers one after the other and
        transform/sample the data, then uses fit_transform on
        transformed data with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        last_step = self._final_estimator

        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt, conf_score = self._fit(X, y, **fit_params_steps)
        assert conf_score is None, "a Detector must always be the last step of Pipeline"

        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt

            fit_params_last = fit_params_steps[self.steps[-1][0]]
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(Xt, yt, **fit_params_last)
            else:
                return last_step.fit(Xt, yt, **fit_params_last).transform(Xt)

    @if_delegate_has_method(delegate="_final_estimator")
    def fit_predict(self, X, y=None, **fit_params):
        """Apply `fit_predict` of last step in pipeline after transforms.

        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted target.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt, conf_score = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            fit_params_last = fit_params_steps[self.steps[-1][0]]

            if conf_score:
                y_pred = self.steps[-1][-1].fit_predict(Xt, yt, conf_score,
                                                        **fit_params_last)
            else:
                y_pred = self.steps[-1][-1].fit_predict(Xt, yt, **fit_params_last)
        return y_pred


def make_pipeline(*steps, **kwargs):
    """Construct a Pipeline from the given estimators.

    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, their names will be set
    to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators
        A list of estimators.

    memory : None, str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Returns
    -------
    p : Pipeline

    See Also
    --------
    imblearn.pipeline.Pipeline : Class for creating a pipeline of
        transforms with a final estimator.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    ... # doctest: +NORMALIZE_WHITESPACE
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('gaussiannb', GaussianNB())])
    """
    memory = kwargs.pop("memory", None)
    verbose = kwargs.pop('verbose', False)
    if kwargs:
        raise TypeError(
            'Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0])
        )
    return Pipeline(skpipeline._name_estimators(steps),
                    memory=memory, verbose=verbose)


