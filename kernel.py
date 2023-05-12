import numpy as np

# Do not change function signatures
#
# input:
#   X is the input matrix of size n_samples x n_features.
#   pass the parameters of the kernel function via kwargs.
# output:
#   Kernel matrix of size n_samples x n_samples 
#   K[i][j] = f(X[i], X[j]) for kernel function f()

def linear(X: np.ndarray, **kwargs)-> np.ndarray:
    assert X.ndim == 2
    if('X_o' in kwargs.keys()) : 
       return kwargs['X_o'] @ X.T
    else : 
        kernel_matrix = X @ X.T
        return kernel_matrix

def polynomial(X:np.ndarray,**kwargs)-> np.ndarray:
    a = kwargs['a']
    b = kwargs['b']
    q = kwargs['q']
    assert X.ndim == 2
    if('X_o' in kwargs.keys()) : 
       return ((kwargs['X_o'] @ X.T)*a + b)**q
    else :
        kernel_matrix = ((X @ X.T)*a + b)**q #degree Q polynomial kernel
        return kernel_matrix
   

def rbf(X:np.ndarray,**kwargs)-> np.ndarray:
    assert X.ndim == 2
    a = kwargs['a']
    if('X_o' in kwargs.keys()) : 
        D = np.sum(kwargs['X_o']**2, axis=-1)[:, None] + np.sum(X**2, axis=-1)[None,:] - 2 * np.dot(kwargs['X_o'] , X.T)
        return np.exp(-a*D)
    else :
        X_norm = np.sum(X ** 2, axis = -1)
        kernel_matrix = np.exp(-a * (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(X, X.T)))
        return kernel_matrix


def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
    assert X.ndim == 2
    a = kwargs['a']
    b = kwargs['b']
    if('X_o' in kwargs.keys()) : 
       return np.tanh(a*(kwargs['X_o']@X.T) + b)
    else :
        kernel_matrix =  np.tanh(a*(X@X.T) + b)
        return kernel_matrix

def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
    assert X.ndim == 2
    a = kwargs['a']
    if('X_o' in kwargs.keys()) : 
        n = X.shape[0]
        m = kwargs['X_o'].shape[0]
        K = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                diff = kwargs['X_o'][i] - X[j]
                K[i, j] = (-a * np.linalg.norm(diff, ord=1))
        return np.exp(K)
    else : 
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = X[i] - X[j]
                K[i, j] = (-a * np.linalg.norm(diff, ord=1))
        return np.exp(K)

