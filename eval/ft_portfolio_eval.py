import torch
import pandas as pd
import numpy as np
import numba
import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from mosek.fusion import *
import mosek.fusion.pythonic
import sys
import seaborn as sns

def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        for key in self.keys():
            self[key] = self.__from_nested_dict(self[key])

    def __setattr__(self, key, value):
        if hasattr(dict, key) and callable(getattr(dict, key)):  # Raise error if attempting to override dict method
            raise AttributeError(f'Attempting to override dict method: {key}')
        super(DotDict, self).__setattr__(key, value)

    @classmethod
    def __from_nested_dict(cls, data):
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.__from_nested_dict(data[key]) for key in data})


def estimate_nfactor_act(X, C=1):
    """
    estimate number of factors given data matrix X (n*p)
    threshold on eigenvalues of correlation matrix (bias corrected)
    https://arxiv.org/abs/1909.10710
    K = # eigenvalues of sample corr that > 1 + sqrt(p / (n-1))
    """
    n, p = X.shape

    # 1. get sample correlation matrix and eigenvalues
    corr = np.corrcoef(X.T)
    evals = np.flip(np.linalg.eigvalsh(corr))  # order large to small

    # 2. get bias corrected eigenvalues
    evals_adj = np.zeros(p - 1)
    for i in range(p - 1):
        mi = (
            np.sum(1.0 / (evals[(i + 1) :] - evals[i]))
            + 4.0 / (evals[i + 1] - evals[i])
        ) / (p - i)
        rho = (p - i) / (n - 1)
        evals_adj[i] = -1.0 / (-(1 - rho) / evals[i] + rho * mi)

    # 3. threshold to estimate number of factors
    thres = 1.0 + np.sqrt(p / (n - 1)) * C
    return np.where(evals_adj > thres)[0][-1] + 1  # max_j that lambda_j > thres



sign = lambda x: x and (1 if x >0 else -1)

def _estimate_K(Y):
    POETKhat_y = POETKhat(Y)
    K1=0.25*(POETKhat_y.K1HL+POETKhat_y.K2HL+POETKhat_y.K1BN+POETKhat_y.K2BN)
    K=np.floor(K1)+1
    return K
    
def POET(Y, K=-np.inf, C=-np.inf, thres='soft', matrix='cor'):
    """
    Estimates large covariance matrices in approximate factor models by thresholding principal orthogonal complements.

    Y:p by n matrix of raw data, where p is the dimensionality, n is the sample size. It is recommended that Y is de-meaned, i.e., each row has zero mean
    K: number of factors
    C: float, the positive constant for thresholding.C=0.5 performs quite well for soft thresholding
    thres: str, choice of thresholding. K=0 corresponds to threshoding the sample covariance directly
    matrix: the option of thresholding either correlation or covairance matrix.'cor': threshold the error correlation matrix then transform back to covariance matrix. 'vad': threshold the error covariance matrix directly.

    Return:
    SigmaY: estimated p by p covariance matrix of y_t
    SigmaU: estimated p by p covariance matrix of u_t
    factors: estimated unobservable factors in a K by T matrix form
    loadings: estimated factor loadings in a p by K matrix form
    """
    if K == -np.inf:
        try:
            K = estimate_nfactor_act(Y)
        except IndexError:
            print("ill-formed matrix Y, provide K with suggestion (K>0 and K<8)")
            return
        
#     if K==-np.inf:
#         K=_estimate_K(Y)

    # Y: p feature * n obs
    p, n = Y.shape
    Y = Y- Y.mean(axis=1)[:, np.newaxis]

    if K>0:
        Dd, V = np.linalg.eigh(Y.T @ Y)
        Dd = Dd[::-1]
        V = np.flip(V,axis=1)
        F = np.sqrt(n)*V[:,:K]  #F is n by K
        LamPCA = Y @ F / n
        uhat = Y - LamPCA @ F.T  # p by n
        Lowrank = LamPCA @ LamPCA.T
        rate = 1/np.sqrt(p)+np.sqrt((np.log(p))/n)
    else:
        uhat = Y # Sigma_y itself is sparse
        rate = np.sqrt((np.log(p))/n)
        Lowrank = np.zeros([p,p])

    SuPCA = uhat @ uhat.T / n
    SuDiag = np.diag(np.diag(SuPCA))
    
    if matrix == 'cor':
        R = np.linalg.inv(SuDiag**(1/2)) @ SuPCA @ np.linalg.inv(SuDiag**(1/2))
    if matrix == 'vad':
        R = SuPCA

    if C == -np.inf:
        C = POETCmin(Y,K,thres,matrix)+0.1

    uu = np.zeros([p,p,n])
    roottheta = np.zeros([p,p])
    lambda_ = np.zeros([p,p])

    for i in range(p):
        for j in range(i+1): # symmetric matrix
            uu[i,j,:] = uhat[i,] * uhat[j,]
            roottheta[i,j] = np.std(uu[i,j,:],ddof=1)
            lambda_[i,j] = roottheta[i,j]*rate*C
            lambda_[j,i] = lambda_[i,j]

    Rthresh = np.zeros([p,p])

    if thres == 'soft':
        for i in range(p):
            for j in range(i+1):
                if np.abs(R[i,j]) < lambda_[i,j] and j < i:
                    Rthresh[i,j] = 0
                elif j == i:
                    Rthresh[i,j] = R[i,j]
                else:
                    Rthresh[i,j]=sign(R[i,j])*(abs(R[i,j])-lambda_[i,j])
                Rthresh[j,i] = Rthresh[i,j]

    elif thres == 'hard':
        for i in range(p):
            for j in range(i+1):
                if np.abs(R[i,j]) < lambda_[i,j] and j < i:
                    Rthresh[i,j] = 0
                else:
                    Rthresh[i,j] = R[i,j]
                Rthresh[j,i] = Rthresh[i,j]

    elif thres == 'scad':
        for i in range(p):
            for j in range(i+1):
                if j == i:
                    Rthresh[i,j] = R[i,j]
                elif abs(R[i,j] < lambda_[i,j]):
                    Rthresh[i,j] = 0
                elif abs(R[i,j])<2*lambda_[i,j]:
                    Rthresh[i,j]=sign(R[i,j])*(abs(R[i,j])-lambda_[i,j])
                elif abs(R[i,j])<3.7*lambda_[i,j]:
                    Rthresh[i,j]=((3.7-1)*R[i,j]-sign(R[i,j])*3.7*lambda_[i,j])/(3.7-2)
                else:
                    Rthresh[i,j] = R[i,j]
                Rthresh[j,i] = Rthresh[i,j]

    SigmaU = np.zeros([p,p])
    if matrix == 'cor':
        SigmaU = SuDiag**(1/2) @ Rthresh * SuDiag**(1/2)
    if matrix == 'vad':
        SigmaU = Rthresh

    SigmaY = SigmaU + Lowrank

    result = DotDict({'SigmaU':SigmaU,
              'SigmaY':SigmaY,
              'factors':F.T,
              'loadings':LamPCA})
    return result


def POETCmin(Y,K,thres,matrix):
    """
    This function is for determining the minimum constant in the threshold that guarantees the positive
definiteness of the POET estimator.
    """
    p, n = Y.shape

    def mineig(Y,K,C,thres,matrix):
        SigmaU = POET(Y,K,C,thres,matrix).SigmaU
        f = min(np.linalg.eigvals(SigmaU))
        return f

    def f(x):
        return mineig(Y,K,x,thres,matrix)

    if f(50)*f(-50)<0:
        roots = scipy.optimize.fsolve(f,[-50,50])
        result = max(0,roots)
    else:
        result = 0
    return result



def POETKhat(Y):
    """
    This function is for calculating the optimal number of factors in an approximate factor model.
    """
    p, n = Y.shape
    Y = Y- Y.mean(axis=1)[:, np.newaxis]
    #Hallin and Liska method

    c=np.arange(0.05, 5.05,0.05)
    re=20
    rmax=10
    IC=np.zeros([2,re,rmax,100])
    gT1HL, gT2HL, pi, ni=np.ones(20),np.ones(20),np.ones(20),np.ones(20)

    for i in range(re): #generate the subsets, "re" of them
        pi[i]=min(i*np.floor(p/re)+min(p,5),p)
        ni[i]=min(i*np.floor(n/re)+min(n,5),n)
        if i==re-1:
            pi[i]=p
            ni[i]=n

        Yi=Y[:int(pi[i]),:int(ni[i])]
        frob=np.zeros(rmax)
        penal=np.zeros(rmax)

        for k in range(min(int(pi[i]),int(ni[i]),rmax)):
            Dd, V = np.linalg.eigh(Yi.T @ Yi)
            Dd = Dd[::-1]
            V = np.flip(V,axis=1)
            F = V[:,:k+1]
            LamPCA = Yi @ F / ni[i]
            uhat = Yi - LamPCA @ (F.T) # pi by ni

            frob[k]=sum(np.diag(uhat @ (uhat.T)))/(pi[i]*ni[i])
            gT1HL[i]=np.log((pi[i]*ni[i])/(pi[i]+ni[i]))*(pi[i]+ni[i])/(pi[i]*ni[i])
            gT2HL[i]=np.log(min(pi[i],ni[i]))*(pi[i]+ni[i])/(pi[i]*ni[i])

            for l in range(100): # only fills in the ICs up to k, which may be <rmax
                IC[0,i,k,l]=np.log(frob[k])+c[l]*(k+1)*gT1HL[i]
                IC[1,i,k,l]=np.log(frob[k])+c[l]*(k+1)*gT2HL[i]


    rhat=np.zeros([2,re,100])
    for i in range(re):
        for l in range(100):
            m=min(pi[i],ni[i],rmax)
            temp1=np.argmin(IC[0,i,:int(m),l])
            rhat[0,i,l]=temp1
            temp2=np.argmin(IC[1,i,:int(m),l])
            rhat[1,i,l]=temp2

    rhat+=1
    sc1, sc2 = np.zeros(100), np.zeros(100)

    for l in range(100):
        sc1[l] = np.std(rhat[0,:,l],ddof=1)
        sc2[l] = np.std(rhat[1,:,l],ddof=1)

    c1vec=np.where(sc1==0)
    ctemp1=c1vec[0][0]

    c1=c[ctemp1]
    K1HL=rhat[0,0,ctemp1]

    c2vec=np.where(sc2==0)
    ctemp2=c2vec[0][0]
    c2=c[ctemp2]
    K2HL=rhat[1,0,ctemp2]

    c=1
    rmax=10
    IC=np.zeros([2,rmax])
    frob, penal = np.zeros(rmax), np.zeros(rmax)

    for k in range(rmax):
        Dd, V = np.linalg.eigh(Y.T @ Y)
        Dd = Dd[::-1]
        V = np.flip(V,axis=1)
        F = V[:,:k+1]
        
        LamPCA = Y @ F / n
        uhat = Y - LamPCA @ (F.T) # p by n
        frob[k]=sum(np.diag(uhat @ uhat.T))/(p*n)
        gT1BN=np.log(np.log((p*n))/(p+n))*(p+n)/(p*n)
        gT2BN=np.log(min(p,n))*(p+n)/(p*n)
        IC[0,k]=np.log(frob[k]) +(k+1)*gT1BN
        IC[1,k]=np.log(frob[k]) +(k+1)*gT2BN


    K1BN = np.argmin(IC[0,:])
    K2BN = np.argmin(IC[1,:])

    result = DotDict({"K1HL":K1HL,"K2HL":K2HL,"K1BN":K1BN,"K2BN":K2BN,"IC":IC})
    return result

def calculate_FF_tangent_portfolio(factor_data, rf_data, k_values=[3, 5, 6, 8]):
    """
    Compute Fama-French factor results for tangent portfolio and Sharpe ratio.
    
    Parameters:
    -----------
    factor_data : np.ndarray
        Array containing factor data (n_samples, n_features)
    rf_data : np.ndarray
        Array containing risk-free rate data (n_samples,)
    k_values : list
        List of k values to use for principal components
        
    Returns:
    --------
    dict
        Dictionary containing tangent portfolio and Sharpe ratio results
    """
    sqrt_252 = np.sqrt(252)
    n_samples = factor_data.shape[0]
    
    results = {
        'tangent': {k: [] for k in k_values},
        'sharpe': {k: [] for k in k_values}
    }
    
    for k in k_values:
        projected_data = factor_data[:, :k]
        excess_returns = projected_data - rf_data[:, np.newaxis]
        
        mu = np.mean(excess_returns, axis=0)
        sigma = np.cov(excess_returns, rowvar=False)
        
        optimal_w = np.linalg.inv(sigma) @ mu
        optimal_w = optimal_w / optimal_w.sum() / (optimal_w.T @ sigma @ optimal_w)
        
        tangent_result = projected_data @ optimal_w
        sharpe_result = sqrt_252 * (mu.T @ optimal_w) / np.sqrt(optimal_w.T @ sigma @ optimal_w)
        
        if mu.T @ optimal_w < 0:
            tangent_result = -tangent_result
            sharpe_result = np.abs(sharpe_result)
            
        results['tangent'][k].append(tangent_result)
        results['sharpe'][k].append(sharpe_result)
            
    return results

def calculate_Emp_tangent_portfolio(test_data, train_data, rf_data, k_values=[3, 5, 6, 8], max_k=8):
    """
    Compute empirical results for tangent portfolio and Sharpe ratio using PCA.
    
    Args:
        test_data: np.ndarray, shape (n_samples, n_features)
        train_data: np.ndarray, shape (n_samples, n_features)
        rf_data: np.ndarray, shape (n_samples,)
        k_values: list of k values to use
        max_k: maximum number of principal components
        
    Returns:
        dict: Dictionary containing tangent portfolio and Sharpe ratio results
    """
    sqrt_252 = np.sqrt(252)
    
    results = {
        'tangent': {k: [] for k in k_values},
        'sharpe': {k: [] for k in k_values}
    }
    
    # Center training data
    train_data = train_data - np.mean(train_data, axis=0)
    
    # Perform PCA
    cov = np.cov(train_data, rowvar=False)
    _, _, V = np.linalg.svd(cov)
    all_eigenvectors = V.T[:, :max_k]
    
    for k in k_values:
        top_k_eigenvectors = all_eigenvectors[:, :k]
        projected_data = test_data @ top_k_eigenvectors
        excess_returns = projected_data - rf_data[:, np.newaxis]
        
        mu = np.mean(excess_returns, axis=0)
        sigma = np.cov(excess_returns, rowvar=False)
        
        optimal_w = np.linalg.inv(sigma) @ mu
        optimal_w = optimal_w / optimal_w.sum() / (optimal_w.T @ sigma @ optimal_w)
        
        tangent_result = projected_data @ optimal_w
        sharpe_result = sqrt_252 * (mu.T @ optimal_w) / np.sqrt(optimal_w.T @ sigma @ optimal_w)
        
        if mu.T @ optimal_w < 0:
            tangent_result = -tangent_result
            sharpe_result = np.abs(sharpe_result)
            
        results['tangent'][k].append(tangent_result)
        results['sharpe'][k].append(sharpe_result)
            
    return results

def calculate_POET_tangent_portfolio(test_data, cov_matrix, rf_data, max_k=8):
    """
    Compute tangent portfolio results using POET covariance matrix.
    
    Args:
        test_data: np.ndarray, shape (n_samples, n_features)
        cov_matrix: np.ndarray, shape (n_features, n_features)
        rf_data: np.ndarray, shape (n_samples,)
        max_k: maximum number of principal components
        
    Returns:
        Dictionary containing tangent portfolio and Sharpe ratio results
    """
    sqrt_252 = np.sqrt(252)
    
    results = {
        'tangent': {k: [] for k in [3, 5, 6, 8]},
        'sharpe': {k: [] for k in [3, 5, 6, 8]}
    }
    
    # Get eigenvectors from POET covariance matrix
    _, _, V = np.linalg.svd(cov_matrix)
    all_eigenvectors = V.T[:, :max_k]
    
    for k in [3, 5, 6, 8]:
        top_k_eigenvectors = all_eigenvectors[:, :k]
        projected_data = test_data @ top_k_eigenvectors
        excess_returns = projected_data - rf_data[:, np.newaxis]
        
        mu = np.mean(excess_returns, axis=0)
        sigma = np.cov(excess_returns, rowvar=False)
        
        optimal_w = np.linalg.inv(sigma) @ mu
        optimal_w = optimal_w / optimal_w.sum()
        
        tangent_result = projected_data @ optimal_w
        sharpe_result = sqrt_252 * (mu.T @ optimal_w) / np.sqrt(optimal_w.T @ sigma @ optimal_w)
        
        if mu.T @ optimal_w < 0:
            tangent_result = -tangent_result
            sharpe_result = sharpe_result
            
        results['tangent'][k].append(tangent_result)
        results['sharpe'][k].append(sharpe_result)
    
    return results

def calculate_RPPCA_tangent_portfolio(test_data, train_data, rf_data, gamma=10, max_k=8):
    """
    Compute RP-PCA empirical results
    
    Args:
        test_data: np.ndarray, shape (n_samples, n_features)
        train_data: np.ndarray, shape (n_samples, n_features)
        rf_data: np.ndarray, shape (n_samples,)
        gamma: Gamma parameter for RP-PCA
        max_k: maximum number of principal components
        
    Returns:
        Dictionary containing tangent portfolio and Sharpe ratio results
    """
    sqrt_252 = np.sqrt(252)
    
    results = {
        'tangent': {k: [] for k in [3, 5, 6, 8]},
        'sharpe': {k: [] for k in [3, 5, 6, 8]}
    }
    
    # Compute RP-PCA covariance matrix
    mean_returns = np.mean(train_data, axis=0)
    cov = 1/train_data.shape[0] * train_data.T @ train_data \
          + gamma * mean_returns[:, np.newaxis] @ mean_returns[np.newaxis, :]
    
    # Get eigenvectors
    _, _, V = np.linalg.svd(cov)
    all_eigenvectors = V.T[:, :max_k]
    
    for k in [3, 5, 6, 8]:
        top_k_eigenvectors = all_eigenvectors[:, :k]
        projected_data = test_data @ top_k_eigenvectors
        excess_returns = projected_data - rf_data[:, np.newaxis]
        
        mu = np.mean(excess_returns, axis=0)
        sigma = np.cov(excess_returns, rowvar=False)
        
        optimal_w = np.linalg.inv(sigma) @ mu
        optimal_w = optimal_w / optimal_w.sum() / (optimal_w.T @ sigma @ optimal_w)
        
        tangent_result = projected_data @ optimal_w
        sharpe_result = sqrt_252 * (mu.T @ optimal_w) / np.sqrt(optimal_w.T @ sigma @ optimal_w)
        
        if mu.T @ optimal_w < 0:
            tangent_result = -tangent_result
            sharpe_result = np.abs(sharpe_result)
            
        results['tangent'][k].append(tangent_result)
        results['sharpe'][k].append(sharpe_result)
    
    return results

def plot_cumulative_log_returns(portfolio_returns, risk_free_returns, figsize=(6, 4), dpi=300):
    """
    Plot cumulative log returns for different portfolio strategies.
    
    Parameters:
    -----------
    portfolio_returns : np.ndarray
        Array containing portfolio returns (n_strategies, n_samples)
    risk_free_returns : np.ndarray
        Array containing risk-free rate returns (n_samples,)
    figsize : tuple, optional
        Figure size, default (6, 4)
    dpi : int, optional
        Figure DPI, default 300
    """
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Calculate cumulative log returns
    excess_returns = portfolio_returns - risk_free_returns[:, np.newaxis]
    cumulative_returns = 1 + np.log(1 + excess_returns).cumsum(axis=0)
    cumulative_returns[0, :] = 1
    
    # Define line properties
    line_widths = [3, 1.5, 1.5, 0.7, 0.7, 0.7, 0.7] 
    line_styles = ['-', '-', '-', '-', '-', '-', '-'] 
    line_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    alphas = [1, 1, 1, 0.6, 0.6, 0.6, 0.6]
    
    # Plot each strategy
    for i in range(cumulative_returns.shape[1]):
        plt.plot(
            cumulative_returns[:, i],
            linewidth=line_widths[i],
            linestyle=line_styles[i],
            color=line_colors[i],
            alpha=alphas[i]
        )
    
    # Add legend and labels
    plt.legend(['Diff+Emp', 'Diff+Shr', 'E-Diff', 'EW', 'VW', 'Emp', 'Shr'], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Cumulative Log-Return", fontsize=16)
    plt.tight_layout()
    plt.show()