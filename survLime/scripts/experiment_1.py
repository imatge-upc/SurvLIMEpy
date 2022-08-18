from typing import List, Union

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
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(cluster_0[0], cluster_0[1], test_size=0.1)
    df = experiment([x_train_1, y_train_1], [x_test_1, y_test_1], exp_name='1.1')

    # Experiment 1.2
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(cluster_1[0], cluster_1[1], test_size=0.1)
    df = experiment([x_train_2, y_train_2], [x_test_2, y_test_2], exp_name='1.2')

    # Experiment 1.3
    # here we train with all the data but we test it with one cluster at a time
    X_3 = np.concatenate([cluster_0[0], cluster_1[0]]); y_3 = np.concatenate([cluster_0[1], cluster_1[1]])
    x_train, x_test, y_train, y_test = train_test_split(X_3, y_3, test_size=0.5)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
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

    times_to_fill = list(set(y_train[1])); times_to_fill.sort()
    columns = ['one','two',' three', 'four', 'five']

    explainer = survlime_tabular.LimeTabularExplainer(x_train, target_data=y_train, feature_names=columns, class_names=None,
                                                   categorical_features=None, verbose=True, discretize_continuous=False)
    import ipdb;ipdb.set_trace()
    if exp_name=='1.3':
       x_test = test[0] 
   #computation_exp = compute_weights(explainer, x_test, model)
   #computation_exp.to_csv(f'/home/carlos.hernandez/PhD/SurvLIME/exp_{exp_name}_surv_weights.csv', index=False)

    # These three lines are not pretty but they get the job done
    if exp_name=='1.3':
      exp_name='1.3.2'
      x_test = test[1]
    computation_exp = 'hehe'
     # computation_exp = compute_weights(explainer, x_test, model)
     #computation_exp.to_csv(f'/home/carlos.hernandez/PhD/SurvLIME/exp_{exp_name}_surv_weights.csv', index=False)
    return computation_exp

def compute_weights(explainer : survlime_tabular.LimeTabularExplainer, x_test : np.ndarray, model : Union[CoxPHSurvivalAnalysis, RandomSurvivalForest]):
    compt_weights = []
    num_pat = 200
    for test_point in tqdm(x_test):
        H_i_j_wc, weights, log_correction, Ho_t_, scaled_data = \
                        explainer.explain_instance(test_point, model.predict_survival_function, num_samples = num_pat)

        b = explainer.solve_opt_problem(H_i_j_wc, weights, log_correction, Ho_t_, scaled_data, verbose=False)
        compt_weights.append(b)
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


