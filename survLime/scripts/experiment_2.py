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
    columns = test[0].columns.tolist()
    num_pat = 1000

    explainer = survlime_tabular.LimeTabularExplainer(train[0], model=model, target_data=train[1], feature_names=columns, class_names=None,
                                                       categorical_features=None, verbose=True, mode='regression', discretize_continuous=False)
    compt_weights = []
    for test_point in tqdm(test[0].values):
        b = explainer.explain_instance(test_point, model.predict_survival_function,
                                                                    num_samples = num_pat)
        b = [x[0] for x in b]
        print(b)
        compt_weights.append(b)
    
    computation = pd.DataFrame(compt_weights, columns=columns)
    computation.to_csv(f'/home/carlos.hernandez/PhD/SurvLIME/{args.dataset}_surv_weights_rs_{args.rs}_na.csv', index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Obtain SurvLIME results for a given dataset')
    parser.add_argument('--dataset', type=str, default='veterans', help='either veterans, lungs, udca or pbc')
    parser.add_argument('--model', type=str, default='cox', help='bb model either cox or rsf')
    parser.add_argument('--rs', type=int, default=0, help='Random seed for the splits')
    args = parser.parse_args()
    main(args)
