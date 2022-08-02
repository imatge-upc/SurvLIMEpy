import pandas as pd
import numpy as np

veteran_path = '/home/carlos.hernandez/PhD/SurvLIME/survLime/datasets/veteran.csv'
udca_path    = './udca_dataset.csv'
pbc_path     = './pbc_dataset.csv'
lung_path    = './lung_dataset.csv'

def Loader(dataset_name : str='veterans') -> list([pd.DataFrame, np.ndarray]):
    """
    Loads a survival dataset, for now it only loads
    the veteran dataset
    """
    if dataset_name=='veterans':
        feature_columns = ['celltype', 'trt', 'karno', 'diagtime', 'age', 'prior']
        df = pd.read_csv(veteran_path) 
    elif dataset_name=='udca':
        feature_columns = ['bili', 'stage', 'riskscore', 'trt']
        df = pd.read_csv(udca_path)
    elif dataset_name=='pbc':
        feature_columns =['age','bili','chol','albumin', 'ast', 'ascites',
                            'copper','alk.phos', 'trig', 'platelet', 'protime',
                            'trt', 'sex', 'hepato', 'spiders', 'edema', 'stage']
        df = pd.read_csv(pbc_path)
    elif dataset_name='lung':
        feature_columns = [ 'inst', 'age', 'sex', 'ph.ecog','ph.karno',
                                    'pat.karno', 'meal.cal', 'wt.loss']
        df = pd.read_csv(lung_path)
        
        
    
    df['status'] = [True if x==1 else False for x in df['status']]
    df['y'] = [(x,y) for x,y in zip(df['status'], df['time'])]
    y = df.pop('y').to_numpy()
    x = df[feature_columns]
    
    
    
    return x, y
    
    
