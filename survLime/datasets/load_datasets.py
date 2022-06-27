import pandas as pd
import numpy as np

veteran_path = '/home/carlos.hernandez/PhD/SurvLIME/survLime/datasets/veteran.csv'
def Loader(dataset_name : str='veterans') -> list([pd.DataFrame, np.ndarray]):
    df = pd.read_csv(veteran_path) 
     
    df['status'] = [True if x==1 else False for x in df['status']]
    df['y'] = [(x,y) for x,y in zip(df['status'], df['time'])]
    y = df.pop('y').to_numpy()
    x = df[['celltype', 'trt', 'karno', 'diagtime', 'age', 'prior']]
    
    return x, y
    
    
