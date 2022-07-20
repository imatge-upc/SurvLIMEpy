from typing import Union
import sys
import inspect
import types

import numpy
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from torch.nn import Module
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

    
def compare_survival_times(bb_model : Union[CoxPHSurvivalAnalysis, Module, RandomSurvivalForest],
                           coefs : numpy.ndarray,
                           X_train : pd.DataFrame, y_train : numpy.ndarray, X_test : pd.DataFrame):
    """Trains a Cox model with the coefs obtained with cvxpy and plots the survival 
       times as well as the 
    
    Args:
        bb_model: model that we want to find the model_interpretable to
        coefs: values obtained during the convex optimization problem to be used as Cox coefficients
        X_train: DataFrame with the training data
        y_train: numpy.ndarray witht the training labels
        X_test:  DataFrame with the test data

    Returns:
        Plots
    """
    model_interpretable = CoxPHSurvivalAnalysis()
    model_interpretable.fit(X_train, y_train)
    model_interpretable.coef_ = coefs

    preds_cox      = bb_model.predict_survival_function(X_test)
    preds_survlime = model_interpretable.predict_survival_function(X_test)

    preds_cox_y  = numpy.mean([x.y for x in preds_cox], axis=0)
    preds_surv_y = numpy.mean([x.y for x in preds_survlime], axis=0)

    plt.plot(preds_cox_y, label='CoxPH')
    plt.plot(preds_surv_y, label='SurvLIME')
    plt.title('Mean survival time comparison')
    plt.legend()
    rmse = sqrt(mean_squared_error(preds_cox_y, preds_surv_y))
    print(f' RMSE between the two curves is {round(rmse, 3)}')

    if isinstance(bb_model, CoxPHSurvivalAnalysis):
        df = pd.DataFrame(columns=bb_model.feature_names_in_, 
                  data=[coefs, bb_model.coef_], index=['SurvLIME','CoxPH'])
        df.transpose().plot.bar()


def has_arg(fn, arg_name):
    """Checks if a callable accepts a given keyword argument.

    Args:
        fn: callable to inspect
        arg_name: string, keyword argument name to check

    Returns:
        bool, whether `fn` accepts a `arg_name` keyword argument.
    """
    if sys.version_info < (3,):
        if isinstance(fn, types.FunctionType) or isinstance(fn, types.MethodType):
            arg_spec = inspect.getargspec(fn)
        else:
            try:
                arg_spec = inspect.getargspec(fn.__call__)
            except AttributeError:
                return False
        return (arg_name in arg_spec.args)
    elif sys.version_info < (3, 6):
        arg_spec = inspect.getfullargspec(fn)
        return (arg_name in arg_spec.args or
                arg_name in arg_spec.kwonlyargs)
    else:
        try:
            signature = inspect.signature(fn)
        except ValueError:
            # handling Cython
            signature = inspect.signature(fn.__call__)
        parameter = signature.parameters.get(arg_name)
        if parameter is None:
            return False
        return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY))
