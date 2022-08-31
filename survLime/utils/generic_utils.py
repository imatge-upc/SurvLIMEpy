from typing import Union, List
import sys
import inspect
import types
import scipy as sp

import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from torch.nn import Module
from sklearn.metrics import mean_squared_error
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis


def fill_matrix_with_total_times(total_times : list,
                                 predicted_surv : np.ndarray,
                                 event_times : np.ndarray):
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
    for time in total_times:
        try:
            if time in event_times:
                time_index = event_times.index(time)
                gl.append(predicted_surv[time_index])
            else:
                gl.append(gl[-1])
        except:
            import ipdb;ipdb.set_trace()
    # Quickfix to delete the one added at the beggining
    del gl[0]
    return gl

def compare_cum_hazard_curves(bb_model : Union[CoxPHSurvivalAnalysis, Module, RandomSurvivalForest],
                           coefs : np.ndarray,
                           X_train : pd.DataFrame, y_train : np.ndarray, X_test : pd.DataFrame):
    """
    Computes a Kolmogorov-Smirnov test for two given models and a test set.

    Args:
    bb_model: model that we want to find the model_interpretable to
    coefs: values obtained during the convex optimization problem to be used as Cox coefficients
    X_train: DataFrame with the training data
    y_train: np.ndarray witht the training labels
    X_test:  DataFrame with the test data
    
    Returns:
    test: KS statistic and pvalue

    """
    times_train = [x[1] for x in y_train]
    times_to_fill = list(set(times_train)); times_to_fill.sort()
    
    model_interpretable = CoxPHSurvivalAnalysis()
    model_interpretable.fit(X_train, y_train)
    model_interpretable.coef_ = coefs
    # Obtain the predictions from both models
    preds_bb      = bb_model.predict_cumulative_hazard_function(X_test)
    preds_survlime = model_interpretable.predict_cumulative_hazard_function(X_test)
   # We need to do this to have the same size as the cox output
    if isinstance(bb_model, RandomSurvivalForest):
        preds_bb_y  = np.mean([fill_matrix_with_total_times(times_to_fill, x.y, list(x.x)) for x in preds_bb], axis=0)
    else:
        preds_bb_y  = np.mean([x.y for x in preds_bb], axis=0)

    preds_surv_y = np.mean([x.y for x in preds_survlime], axis=0)
    
    test = compute_kolmogorov_test(preds_bb_y, preds_surv_y)

    return test

def compare_survival_times(bb_model : Union[CoxPHSurvivalAnalysis, Module, RandomSurvivalForest],
                           coefs : np.ndarray,
                           X_train : pd.DataFrame, y_train : np.ndarray, X_test : pd.DataFrame, true_coef : List[float] = None):
    """Trains a Cox model with the coefs obtained with cvxpy and plots the survival 
       times.
    
    Args:
        bb_model: model that we want to find the model_interpretable to
        coefs: values obtained during the convex optimization problem to be used as Cox coefficients
        X_train: DataFrame with the training data
        y_train: np.ndarray witht the training labels
        X_test:  DataFrame with the test data
        true_coef: true coeficcients used for the simulated data 

    Returns:
        Plots
    """
    times_train = [x[1] for x in y_train]
    times_to_fill = list(set(times_train)); times_to_fill.sort()
    
    model_interpretable = CoxPHSurvivalAnalysis(alpha=0.0001)
    model_interpretable.fit(X_train, y_train)
    model_interpretable.coef_ = coefs
    # Obtain the predictions from both models
    preds_bb      = bb_model.predict_survival_function(X_test)
    preds_survlime = model_interpretable.predict_survival_function(X_test)

   
    # We need to do this to have the same size as the cox output
    if isinstance(bb_model, RandomSurvivalForest):
        preds_bb_y  = np.mean([fill_matrix_with_total_times(times_to_fill, x.y, list(x.x)) for x in preds_bb], axis=0)
    else:
        preds_bb_y  = np.mean([x.y for x in preds_bb], axis=0)

    preds_surv_y = np.mean([x.y for x in preds_survlime], axis=0)

    rmse = sqrt(mean_squared_error(preds_bb_y, preds_surv_y))
    if isinstance(bb_model, CoxPHSurvivalAnalysis):
        plot_num=2
        
        if true_coef:
            data  =  [bb_model.coef_, coefs, true_coef]
            index =  ['CoxPH', 'SurvLIME', 'True coef']
        else:
            data  = [bb_model.coef_, coefs]
            index = ['CoxPH','SurvLIME']
        df = pd.DataFrame(columns=bb_model.feature_names_in_, 
                  data=data, index=index)

        # Create axes and access them through the returned array
        fig, axs = plt.subplots(1, plot_num, figsize=(15,5))
        df.transpose().plot.bar(ax=axs[0])
        axs[0].set_title('Coefficient values for bb model and survlime')
        axs[1].step(preds_bb[0].x, preds_bb_y, where="post", label='BB model')
        axs[1].step(preds_survlime[0].x, preds_surv_y, where="post", label='SurvLIME')
        axs[1].legend()
        axs[1].set_title(f'Mean survival time comparison RMSE: {rmse:.3}')
    # If we are using other model, we don't have coefficients to compare with
    else:
        plt.step(preds_survlime[0].x, preds_bb_y, where="post", label='BB model')
        plt.step(preds_survlime[0].x, preds_surv_y, where="post", label='SurvLIME')
        plt.legend()
        plt.title(f'Mean survival time comparison RMSE: {rmse:.3}')


def compute_rmse(compt_weights : pd.DataFrame, coefficients : List):
    return np.sqrt(np.mean(np.sum(np.square(compt_weights - coefficients), axis=1)))

def compute_kolmogorov_test(first_sample : np.ndarray, second_sample : np.ndarray):
    """
    Computes the Kolmogorov Smirnoff tests given two samples

    Args:
    first_sample  : np.ndarray : Cumulative hazard as computed by one model
    second_sample : np.ndarray : Cumulative hazard as computed by a second model
    """
    norm = np.linalg.norm(first_sample)
    first_sample = first_sample/norm

    norm = np.linalg.norm(second_sample)
    second_sample = second_sample/norm

    test_result = sp.stats.ks_2samp(first_sample, second_sample)

    return test_result
