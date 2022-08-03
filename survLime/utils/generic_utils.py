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
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis



def fill_matrix_with_total_times(total_times : list,
                                 predicted_surv : numpy.ndarray,
                                 event_times : numpy.ndarray):
    """The model only outputs a prediction for the event times
       this function fills the gaps between them

    Args:
    total_times    : total number of time steps
    predicted_surv : array with the predicted survival times
    event_times    : array with the times where an event happened

    Returns:
    gl : list with a prediction for all the survival times

    """
    gl = [1]
    for time in range(1, int(max(total_times))):
        try:
            if time in event_times:
                time_index = event_times.index(time)
                gl.append(predicted_surv[time_index])
            else:
                gl.append(gl[-1])
        except:
            import ipdb;ipdb.set_trace()
    return gl

def compare_survival_times(bb_model : Union[CoxPHSurvivalAnalysis, Module, RandomSurvivalForest],
                           coefs : numpy.ndarray,
                           X_train : pd.DataFrame, y_train : numpy.ndarray, X_test : pd.DataFrame):
    """Trains a Cox model with the coefs obtained with cvxpy and plots the survival 
       times.
    
    Args:
        bb_model: model that we want to find the model_interpretable to
        coefs: values obtained during the convex optimization problem to be used as Cox coefficients
        X_train: DataFrame with the training data
        y_train: numpy.ndarray witht the training labels
        X_test:  DataFrame with the test data

    Returns:
        Plots
    """
    times_train = [x[1] for x in y_train]
    times_to_fill = list(set(times_train)); times_to_fill.sort()
    
    model_interpretable = CoxPHSurvivalAnalysis()
    model_interpretable.fit(X_test, y_train)
    model_interpretable.coef_ = coefs
    
    # Obtain the predictions from both models
    preds_bb      = bb_model.predict_survival_function(X_test)
    preds_survlime = model_interpretable.predict_survival_function(X_test)
    #import ipdb;ipdb.set_trace()
    preds_bb_y  = numpy.mean([x.y for x in preds_bb], axis=0)
   
    # We need to do this to have the same size as the cox output
    if isinstance(bb_model, RandomSurvivalForest):
        preds_bb_y  = numpy.mean([fill_matrix_with_total_times(times_to_fill, x.y, list(x.x)) for x in preds_bb], axis=0)

    preds_surv_y = numpy.mean([x.y for x in preds_survlime], axis=0)

    # 
    rmse = sqrt(mean_squared_error(preds_bb_y, preds_surv_y))
    if isinstance(bb_model, CoxPHSurvivalAnalysis):
        plot_num=2
        # Create axes and access them through the returned array
        fig, axs = plt.subplots(1, plot_num, figsize=(15,5))
        df = pd.DataFrame(columns=bb_model.feature_names_in_, 
                  data=[bb_model.coef_, coefs], index=['SurvLIME','CoxPH'])
        df.transpose().plot.bar(ax=axs[0])
        axs[0].set_title('Coefficient values for bb model and survlime')
        axs[1].step(preds_bb[0].x, preds_bb_y, where="post", label='BB model')
        axs[1].step(preds_survlime[0].x, preds_surv_y, where="post", label='SurvLIME')
        axs[1].set_title(f'Mean survival time comparison RMSE: {rmse:.3}')
    # If we are using other model, we don't have coefficients to compare with
    else:
        plt.step(preds_survlime[0].x, preds_bb_y, where="post", label='BB model')
        plt.step(preds_survlime[0].x, preds_surv_y, where="post", label='SurvLIME')
        plt.legend()
        plt.title(f'Mean survival time comparison RMSE: {rmse:.3}')

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
