from sklearn.covariance import LedoitWolf
import numpy as np

def calculate_mean_cov(data, shr=False, clip=None):
    """
    Calculate mean and covariance matrix for input data.
    
    Args:
        data (numpy.ndarray): Input data array
        shr (bool): Whether to use LedoitWolf shrinkage, default True
        clip (float or None): If not None, winsorize data at this quantile level (e.g., 0.01 for 1% quantile)
        
    Returns:
        tuple: (mean_vector, cov_matrix)
            - mean_vector: Mean vector of the data
            - cov_matrix: Covariance matrix of the data
    """
    # Winsorize data if requested
    if clip is not None:
        lower_quantile = np.quantile(data, clip, axis=0)
        upper_quantile = np.quantile(data, 1-clip, axis=0)
        data = np.clip(data, lower_quantile, upper_quantile)
    
    # Calculate mean vector
    mean_vector = np.mean(data, axis=0)
    
    # Calculate covariance matrix
    if shr:
        # Use LedoitWolf shrinkage for better estimation
        cov_matrix = LedoitWolf().fit(data).covariance_
    else:
        # Use sample covariance matrix
        cov_matrix = np.cov(data.T)
    
    return mean_vector, cov_matrix