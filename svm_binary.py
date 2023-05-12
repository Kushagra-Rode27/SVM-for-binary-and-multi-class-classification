from typing import List
import numpy as np
import pandas as pd
import qpsolvers
import kernel as kp
class Trainer:
    def __init__(self,kernel,C=None,**kwargs) -> None:
        self.kernel = kernel
        self.kwargs = kwargs
        self.C=C
        self.support_vectors:List[np.ndarray] = []
        self.b:List[np.ndarray] = [] #storing the bias
        self.sv_alp:List[np.ndarray] = [] #storing the support vector alpha's
        self.sv_y:List[np.ndarray] = [] #storing the y corresponding to support vectors 
        
    def fit(self, train_data_path:str)->None:
        #TODO: implement
        #store the support vectors in self.support_vectors
        tf = pd.read_csv(train_data_path)
        y = np.array(tf['y'].values)
        y = np.where(y == 1,1,-1)
        X = np.array(tf.values[0:,1 : 2049])
        if self.kernel == 'l':
            k = kp.linear(X)
        elif self.kernel == 'p':
            k = kp.polynomial(X, a=self.kwargs['a'],b=self.kwargs['b'],q=self.kwargs['q'])
        elif self.kernel == 'rbf':
            k = kp.rbf(X,a=self.kwargs['a'])
        elif self.kernel == 's':
            k = kp.sigmoid(X,a=self.kwargs['a'],b=self.kwargs['b'])
        elif self.kernel == 'la':
            k = kp.laplacian(X,a=self.kwargs['a'])
        else:
            raise ValueError('Invalid kernel')

        n_samples = X.shape[0]
        
        P = np.outer(y, y) * k
        q = -np.ones(n_samples)
        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        
        h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))
        A = y.reshape(1, -1)
        b = np.array([0.0])

        alpha = qpsolvers.solve_qp(P, q, G, h, A, b,solver='quadprog')
    
        sup_vecs = alpha > 1e-5
        
        ind = np.arange(len(alpha))[sup_vecs]
        self.sv_alp = alpha[sup_vecs]
        
        self.support_vectors = X[sup_vecs]
        
        self.sv_y = y[sup_vecs]
        self.b = 0
        for n in range(len(self.sv_alp)):
            self.b += self.sv_y[n]
            for a, sv_y, sv in zip(self.sv_alp, self.sv_y, ind):
                self.b -= a * sv_y * k[sv,ind[n]]
        self.b /= len(self.sv_alp)
    
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels as a numpy array of dimension n_samples on test data
        tf = pd.read_csv(test_data_path)
        if 'y' in tf.columns : 
            tf.drop(columns='y',inplace=True)
        X = np.array(tf.values[0:,1 :])

        if self.kernel == 'l':
            k = kp.linear(self.support_vectors,X_o=X)
        elif self.kernel == 'p':
            k = kp.polynomial(self.support_vectors,X_o=X,a=self.kwargs['a'],b=self.kwargs['b'],q=self.kwargs['q'])
        elif self.kernel == 'rbf':
            k = kp.rbf(self.support_vectors,X_o=X,a=self.kwargs['a'])
        elif self.kernel == 's':
            k = kp.sigmoid(self.support_vectors,X_o=X,a=self.kwargs['a'],b=self.kwargs['b'])
        elif self.kernel == 'la':
            k = kp.laplacian(self.support_vectors,X_o=X,a=self.kwargs['a'])
        else:
            raise ValueError('Invalid kernel')
        
        ind = np.arange(len(self.support_vectors))
        y_predict = np.empty(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.sv_alp, self.sv_y, ind):
                s += a * sv_y * k[i,sv]
            y_predict[i] = s
            
        y_ot = np.sign(y_predict + self.b)
        y_ot = np.where(y_ot == 1,1,0)
        return y_ot

    # I have added two new functions here for implementing multiclass classification
    def fit_gen(self,X,y,flag) : 
        if(flag == 0) : 
            classif = self.kwargs["classif"]
            class1 = classif[0]
            class2 = classif[1]
            curr = ((y == class1) | (y == class2)) 
            y_new = y[curr]
            y_new = np.where(y_new == class1,1,-1)
            X_new = X[curr]
            
        else : 
            classif = self.kwargs["classif"] 
            y_new = y
            y_new = np.where(y_new == classif,1,-1)
            X_new = X

        k = np.zeros(X_new.shape)
        if self.kernel == 'l':
            k = kp.linear(X_new)

        elif self.kernel == 'p':
            k = kp.polynomial(X_new, a=self.kwargs['a'],b=self.kwargs['b'],q=self.kwargs['q'])
        elif self.kernel == 'rbf':
            k = kp.rbf(X_new,a=self.kwargs['a'])
        elif self.kernel == 's':
            k = kp.sigmoid(X_new,a=self.kwargs['a'],b=self.kwargs['b'])
        elif self.kernel == 'la':
            k = kp.laplacian(X_new,a=self.kwargs['a'])
        else:
            raise ValueError('Invalid kernel')

        n_samples = X_new.shape[0]

        P = np.outer(y_new, y_new) * k

        P = P + np.identity(X_new.shape[0]) * 1e-8
        q = -np.ones(n_samples)
        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        
        h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))
        A = y_new.reshape(1, -1)
        b = np.array([0.0])

        alpha = qpsolvers.solve_qp(P, q, G, h, A, b,solver='quadprog')
        
        sup_vecs = alpha > 1e-6
        ind = np.arange(len(alpha))[sup_vecs]
        self.sv_alp = alpha[sup_vecs]
        
        self.support_vectors = X_new[sup_vecs]
        
        self.sv_y = y_new[sup_vecs]
        self.b = 0

        for n in range(len(self.sv_alp)):
            self.b += self.sv_y[n]
            for a, sv_y, sv in zip(self.sv_alp, self.sv_y, ind):
                self.b -= a * sv_y * k[sv,ind[n]]
        self.b /= len(self.sv_alp)


    def predict_gen(self, test_data_path:str,flag)->np.ndarray:
        
        tf = pd.read_csv(test_data_path)
        if 'y' in tf.columns : 
            tf.drop(columns='y',inplace=True)
        X = np.array(tf.values[0:,1 :])
        
        if self.kernel == 'l':
            k = kp.linear(self.support_vectors,X_o=X)
        elif self.kernel == 'p':
            k = kp.polynomial(self.support_vectors,X_o=X,a=self.kwargs['a'],b=self.kwargs['b'],q=self.kwargs['q'])
        elif self.kernel == 'rbf':
            k = kp.rbf(self.support_vectors,X_o=X,a=self.kwargs['a'])
        elif self.kernel == 's':
            k = kp.sigmoid(self.support_vectors,X_o=X,a=self.kwargs['a'],b=self.kwargs['b'])
        elif self.kernel == 'la':
            k = kp.laplacian(self.support_vectors,X_o=X,a=self.kwargs['a'])
        else:
            raise ValueError('Invalid kernel')
        
        ind = np.arange(len(self.support_vectors))
        y_predict = np.empty(len(X))
       
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.sv_alp, self.sv_y, ind):
                s += a * sv_y * k[i,sv]
            y_predict[i] = s
        if(flag == 0) : 
            return np.sign(y_predict + self.b)
        else :
            return 1 / (1 + np.exp(-1*(y_predict + self.b))) #acts as probability of each class

    







