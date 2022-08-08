from typing import Union
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.util import Surv


# TODO make it work for any path
user_path    = '/home/carlos.hernandez/PhD/'

veteran_path = user_path + 'SurvLIME/survLime/datasets/veteran.csv'
udca_path    = user_path + 'SurvLIME/survLime/datasets/udca_dataset.csv'
pbc_path     = user_path + 'SurvLIME/survLime/datasets/pbc_dataset.csv'
lung_path    = user_path + 'SurvLIME/survLime/datasets/lung_dataset.csv'

class Loader():

    def __init__(self, dataset_name : str='veterans'):
        
        if dataset_name=='veterans':
            self.feature_columns = ['celltype', 'trt', 'karno', 'diagtime', 'age', 'prior']
            self.categorical_columns = ['celltype']
            self.df = pd.read_csv(veteran_path) 
        elif dataset_name=='udca':
            self.feature_columns = ['bili', 'stage', 'riskscore', 'trt']
            self.categorical_columns = []
            self.df = pd.read_csv(udca_path)
        elif dataset_name=='pbc':
            self.feature_columns =['age','bili','chol','albumin', 'ast', 'ascites',
                                'copper','alk.phos', 'trig', 'platelet', 'protime',
                                'trt', 'sex', 'hepato', 'spiders', 'edema', 'stage']
            self.categorical_columns = ['edema', 'stage']
            self.df = pd.read_csv(pbc_path)
            self.df['sex'] = [1 if x=='f' else 0 for x in self.df['sex']]
        elif dataset_name=='lung':
            self.feature_columns = [ 'inst', 'age', 'sex', 'ph.ecog','ph.karno',
                                        'pat.karno', 'meal.cal', 'wt.loss']
            self.categorical_columns = ['ph.ecog']
            self.df = pd.read_csv(lung_path)
        else:
            raise AssertionError(f'The give name {dataset_name} was not found in [veterans, udca, pbc, lung]')

    def load_data(self) -> list([pd.DataFrame, np.ndarray]):
        """
        Loads a survival dataset

        Returns:
        x : pd.DataFrame with the unprocessed features
        y : np.ndarray of tuples with (status, time)
        """
        self.df['status'] = [True if x==1 else False for x in self.df['status']]
        self.df['y'] = [(x,y) for x,y in zip(self.df['status'], self.df['time'])]
        y = self.df.pop('y').to_numpy()
        events = [x[0] for x in y]
        times  = [x[1] for x in y]
        x = self.df[self.feature_columns]

        x.fillna(value=x.median(), inplace=True)
        
        return x, events, times 
   
    def preprocess_datasets(self, x : pd.DataFrame,
                            events : list, times : list) -> list([pd.DataFrame, np.ndarray]):
        """
        Preprocesses the data to be used as model input.

        For now it only converts categorical features to OHE and
        standarizes the data
        """

        # Deal with categorical features
        x_pre = x.copy()
        for cat_feat in self.categorical_columns:
            names = [cat_feat+'_'+str(value) for value in x_pre[cat_feat].unique()]
            x_pre[names] = pd.get_dummies(x_pre[cat_feat])
            x_pre.drop(cat_feat, inplace=True, axis=1)
        
        # Then convert the data and the features to three splits
        # and standarize them
        y = Surv.from_arrays(events, times)
        X_train, X_test, y_train, y_test = train_test_split(x_pre.copy(), y, test_size=0.40, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_test.copy(), y_test, test_size=0.5, random_state=1)



        scaler = StandardScaler()
        X_train_processed = pd.DataFrame(data=scaler.fit_transform(X_train, y_train),
                                         columns=X_train.columns, index=X_train.index)

        X_val_processed   = pd.DataFrame(data=scaler.transform(X_val),
                                         columns=X_val.columns, index=X_val.index)

        X_test_processed  = pd.DataFrame(data=scaler.transform(X_test),
                                         columns=X_test.columns, index=X_test.index)

        return [X_train_processed, y_train], [X_val_processed, y_val], [X_test_processed, y_test]

    

        
    
    
