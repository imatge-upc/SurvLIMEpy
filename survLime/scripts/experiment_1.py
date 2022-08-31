from typing import List, Union

from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from sklearn.model_selection import train_test_split
from survLime.datasets.load_datasets import RandomSurvivalData
from survLime import survlime_tabular



def experiment_1():
    
    cluster_0, cluster_1 = create_clusters()
    
    # Experiment 1.1
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(cluster_0[0], cluster_0[1], test_size=0.1, random_state=10)
    df = experiment([x_train_1, y_train_1], [x_test_1, y_test_1], exp_name='1.1')

    # Experiment 1.2
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(cluster_1[0], cluster_1[1], test_size=0.1, random_state=10)
    df = experiment([x_train_2, y_train_2], [x_test_2, y_test_2], exp_name='1.2')

    # Experiment 1.3
    # here we train with all the data but we test it with one cluster at a time
    X_3 = np.concatenate([cluster_0[0], cluster_1[0]]); y_3 = np.concatenate([cluster_0[1], cluster_1[1]])
    x_train, x_test, y_train, y_test = train_test_split(X_3, y_3, test_size=0.5, random_state=10)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=10)
    df = experiment([x_train, y_train], [[x_test_1, x_test_2], [y_test_1, y_test_2]], exp_name='1.3')

def experiment(train : List, test : List, model_type : str='cox', exp_name : str='1.1'):
    """
    This is going to be the same for all the experiments, we should define it generally

    """
    x_train = train[0]; y_train = train[1]
    x_test = test[0];
    if model_type=='cox':
        model = CoxPHSurvivalAnalysis(alpha=0.0001)
    elif model_type=='rsf':
        model = RandomSurvivalForest()
    else:
        raise AssertionError(f'The model {model_type} needs to be either [cox or rsf]')

    times_to_fill = list(set([x[1] for x in y_train])); times_to_fill.sort()
    columns = ['one','two',' three', 'four', 'five']
    model.fit(x_train, y_train)
    model.feature_names_in_ = columns 
    
    H0 = model.cum_baseline_hazard_.y.reshape(len(times_to_fill), 1)
    explainer = survlime_tabular.LimeTabularExplainer(x_train,
                                                      y_train
                                                      )

    if exp_name=='1.3':
       x_test = test[0][0]
    computation_exp = compute_weights(explainer, x_test, model)
    computation_exp.to_csv(f'/home/carlos.hernandez/PhD/SurvLIME/exp_{exp_name}_surv_weights_na.csv', index=False)
    # These three lines are not pretty but they get the job done
    if exp_name=='1.3':
        exp_name='1.3.2'
        x_test = test[0][1]
        computation_exp = compute_weights(explainer, x_test, model)
        computation_exp.to_csv(f'/home/carlos.hernandez/PhD/SurvLIME/exp_{exp_name}_surv_weights_na.csv', index=False)
    return computation_exp

def compute_weights(explainer : survlime_tabular.LimeTabularExplainer, x_test : np.ndarray, model : Union[CoxPHSurvivalAnalysis, RandomSurvivalForest]):
    compt_weights = []
    num_pat = 1000
    predict_chf = partial(model.predict_cumulative_hazard_function, return_array=True)
    for test_point in tqdm(x_test):
        b, result= \
                        explainer.explain_instance(test_point, predict_chf, verbose=False, num_samples = num_pat)

        import ipdb;ipdb.set_trace()
        b = [x[0] for x in b]
        compt_weights.append(b)
    columns = ['one','two','threen', 'four', 'five']
    computation_exp = pd.DataFrame(compt_weights, columns=columns)

    return computation_exp 


def create_clusters():
    """
    Creates the clusters proposed in the paper: https://arxiv.org/pdf/2003.08371.pdf

    Returns:
    cluster 0: list[Data, target]
    cluster 1: List[Data, target]
    """
    
    # These values are shared among both clusters
    radius = 8
    num_points = 1000
    prob_event = 0.9
    lambda_weibull = 10**(-6)
    v_weibull = 2
  
    # First cluster
    center = [0, 0, 0, 0, 0]
    coefficients = [10**(-6), 0.1,  -0.15, 10**(-6), 10**(-6)]
    rds = RandomSurvivalData(center, radius, coefficients, prob_event, lambda_weibull, v_weibull, random_seed=23)
    X_0, T_0, delta_0 = rds.random_survival_data(num_points)
    z_0 = [(d, int(t)) for d, t in zip(delta_0, T_0)]
    y_0 = np.array(z_0, dtype=[('delta', np.bool_), ('time_to_event', np.float32)])
    
    # From page 6 of the paper (I think)
    center = [4, -8, 2, 4, 2]
    coefficients = [10**(-6), -0.15, 10**(-6), 10**(-6), -0.1]
    rds = RandomSurvivalData(center, radius, coefficients, prob_event, lambda_weibull, v_weibull, random_seed=23)
    X_1, T_1, delta_1 = rds.random_survival_data(num_points)
    z_1 = [(d, int(t)) for d, t in zip(delta_1, T_1)]
    y_1 = np.array(z_1, dtype=[('delta', np.bool_), ('time_to_event', np.float32)])

    return [X_0, y_0], [X_1, y_1]

if __name__=='__main__':
    experiment_1()


