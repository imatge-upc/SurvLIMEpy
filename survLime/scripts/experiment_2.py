from typing import Union
import argparse
from functools import partial

from tqdm import tqdm
import numpy as np
import pandas as pd

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

# Our very own survLime!
from survLime.datasets.load_datasets import Loader 
from survLime import survlime_tabular

def main(args):
    if args.dataset=='all':
        datasets = ['veterans', 'udca', 'lung', 'pbc']
    else:
        datasets = [args.dataset]
    for dataset in datasets:
        loader = Loader(dataset_name=dataset)
        x, events, times = loader.load_data()

        train, val, test = loader.preprocess_datasets(x, events, times, random_seed=args.rs)

        if args.model=='cox':
            model = CoxPHSurvivalAnalysis(alpha=0.0001)
        elif args.model=='rsf':
            model = RandomSurvivalForest()
        else:
            raise AssertionError

        model.fit(train[0], train[1])
        print(f'C-index is - {round(model.score(test[0], test[1]), 3)}')

        times_to_fill = list(set([x[1] for x in train[1]])); times_to_fill.sort()
        H0 = model.cum_baseline_hazard_.y.reshape(len(times_to_fill), 1)

        explainer = survlime_tabular.LimeTabularExplainer(train[0],
                                                          train[1],
                                                          H0=H0)

        computation_exp = compute_weights(explainer, test[0], model)
        save_path = f'/home/carlos.hernandez/PhD/SurvLIME/survLime/computed_weights_csv/exp_{dataset}_surv_weights.csv'
        computation_exp.to_csv(save_path, index=False)

def compute_weights(explainer : survlime_tabular.LimeTabularExplainer,
                    x_test : np.ndarray, model : Union[CoxPHSurvivalAnalysis, RandomSurvivalForest]):
    compt_weights = []
    num_pat = 1000
    predict_chf = partial(model.predict_cumulative_hazard_function, return_array=True)
    for test_point in tqdm(x_test.to_numpy()):
        b, _= explainer.explain_instance(test_point,
                                         predict_chf, verbose=False, num_samples = num_pat)

        b = [x[0] for x in b]
        compt_weights.append(b)

    return pd.DataFrame(compt_weights, columns=model.feature_names_in_)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Obtain SurvLIME results for a given dataset')
    parser.add_argument('--dataset', type=str, default='veterans', help='either veterans, lungs, udca or pbc')
    parser.add_argument('--model', type=str, default='cox', help='bb model either cox or rsf')
    parser.add_argument('--rs', type=int, default=0, help='Random seed for the splits')
    args = parser.parse_args()
    main(args)
