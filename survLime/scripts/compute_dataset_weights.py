import numpy as np
import pandas as pd
from tqdm import tqdm

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest


# Our very own survLime!
from survLime.datasets.load_datasets import Loader 
from survLime import survlime_tabular
import argparse

def main(args):
    loader = Loader(dataset_name=args.dataset)
    x,events, times = loader.load_data()

    train, val, test = loader.preprocess_datasets(x, events, times)
    events_train = [x[0] for x in train[1]]
    times_train  = [x[1] for x in train[1]]

    events_val = [x[0] for x in val[1]]
    times_val  = [x[1] for x in val[1]]

    events_test = [x[0] for x in test[1]]
    times_test  = [x[1] for x in test[1]]
    
    if args.model=='cox':
        model = CoxPHSurvivalAnalysis(alpha=0.0001)
    elif args.model=='rsf':
        model = RandomSurvivalForest()
    else:
        raise AssertionError

    model.fit(train[0], train[1])
    print(f'C-index is - {round(model.score(test[0], test[1]), 3)}')
    columns = test[0].columns.tolist()
    num_pat = 500

    explainer = survlime_tabular.LimeTabularExplainer(train[0], target_data=train[1], feature_names=columns, class_names=None,
                                                       categorical_features=None, verbose=True, mode='regression', discretize_continuous=False)
    compt_weights = []
    for test_point in tqdm(test[0].values):
        H_i_j_wc, weights, log_correction, Ho_t_, scaled_data = \
                        explainer.explain_instance(test_point, model.predict_survival_function, num_samples = num_pat)
        
        b_ = explainer.solve_opt_problem(H_i_j_wc, weights, log_correction, Ho_t_, scaled_data)
        compt_weights.append(b_)
    
    computation = pd.DataFrame(compt_weights, columns=columns)

    computation.to_csv(f'/home/carlos.hernandez/PhD/SurvLIME/{args.dataset}_surv_weights.csv', index=False)
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, default='veterans', help='either veterans, lungs, udca or pbc')
    parser.add_argument('--model', type=str, default='cox', help='bb model either cox or rsf')
    args = parser.parse_args()
    main(args)
