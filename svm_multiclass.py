from typing import List
import numpy as np
from svm_binary import Trainer
import pandas as pd
class Trainer_OVO:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers 
        self.svms = []
        for i in range(self.n_classes) : 
            for j in range(i + 1,self.n_classes): 
                t = Trainer(self.kernel,self.C,a=self.kwargs['a'],classif=(i+1,j+1))
                self.svms.append(t)
        
    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        #Store the trained svms in self.svms
        if(len(self.svms) == 0):
            self._init_trainers()
        tf = pd.read_csv(train_data_path)
        y = np.array(tf['y'].values)
        X = np.array(tf.values[0:,2 :])
        
        for i in range(len(self.svms)) : 
            t = self.svms[i]
            t.fit_gen(X,y,0)

    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels
        vf = pd.read_csv(test_data_path)
        
        n_samples = len(np.array(vf.values[0:,0]))
    
        y_vot = np.empty((len(self.svms),n_samples))
        for i in range(len(self.svms)) : 
            t = self.svms[i]
            c1 = t.kwargs['classif'][0]
            c2 = t.kwargs['classif'][1]
            y_pred = t.predict_gen(test_data_path,0)
            y_pred = np.where(y_pred == 1,c1,c2)
            y_vot[i] = y_pred
        y_vot = y_vot.astype('int64')
        
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=y_vot)
    
class Trainer_OVA:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers
        self.svms = [] 
        for i in range(self.n_classes) : 
            t = Trainer(self.kernel,self.C,a=self.kwargs['a'],classif=(i+1))
            self.svms.append(t)
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        #Store the trained svms in self.svms
        self._init_trainers()
        tf = pd.read_csv(train_data_path)
        y = np.array(tf['y'].values)
        X = np.array(tf.values[0:,2 :])
        for i in range(len(self.svms)) : 
            t = self.svms[i]
            t.fit_gen(X,y,1)
        
    
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels
        vf = pd.read_csv(test_data_path)
        
        n_samples = len(np.array(vf.values[0:,0]))
        
        y_vot = np.empty((len(self.svms),n_samples))
        for i in range(len(self.svms)) : 
            t = self.svms[i]
            
            y_pred = t.predict_gen(test_data_path,1)
            
            y_vot[i] = y_pred
        return y_vot.argmax(axis=0) + 1

