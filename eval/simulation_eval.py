import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from scipy.stats import norm
from scipy.stats.mstats import winsorize
from models.diffusion_factor_model import GaussianLatentSampler2D_Finance
from tqdm.auto import tqdm
import seaborn as sns
sns.set_style('white')

def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)


def comparision_histplot_simulation(stock_i, training_data_path, generated_data_path, ground_truth_mean, ground_truth_cov, bins_num=50, x_bound=3, y_bound=0.04, zoomin_bound=0.5):
    """
    Plot histogram comparison between generated and training data for a given stock.
    
    Args:
        stock_i (int): Index of the stock to plot
        training_data_path (str): Path to training data file
        generated_data_path (str): Path to generated data file
        ground_truth_mean (np.ndarray): Ground truth mean vector
        ground_truth_cov (np.ndarray): Ground truth covariance matrix
        bins_num (int): Number of bins for histogram
        x_bound (float): X-axis limit
        y_bound (float): Y-axis limit
        zoomin_bound (float): Zoom-in area boundary
    """
    # Load data
    training_return_data = np.load(training_data_path)
    generated_return_data = np.load(generated_data_path)

    # Print statistics
    print("Generated Samples:", generated_return_data[:, stock_i].min(), generated_return_data[:, stock_i].max(),
          np.round(generated_return_data[:, stock_i].mean(), 3), np.round(generated_return_data[:, stock_i].var(), 3))
    print("Training Samples:", training_return_data[:, stock_i].min(), training_return_data[:, stock_i].max(),
          np.round(training_return_data[:, stock_i].mean().item(), 3), np.round(training_return_data[:, stock_i].var().item(), 3))
    print("Ground Truth:", np.round(ground_truth_mean[stock_i], 3), np.round(ground_truth_cov[stock_i, stock_i], 3))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=400)
    
    # Plot generated data
    sns.histplot(ax=axes[0], data=generated_return_data[:, stock_i], bins=bins_num, alpha=1,
                stat="proportion", color="C0", label="Generated")
    bin_edges = np.histogram(generated_return_data[:, stock_i], bins=bins_num, density=False)[1]
    return_cdf = norm.cdf(bin_edges, ground_truth_mean[stock_i], np.sqrt(ground_truth_cov[stock_i, stock_i]))
    cdf_diff = np.diff(return_cdf)
    axes[0].plot(bin_edges[1:], cdf_diff, label='Ground truth', color="C3", linestyle="--", linewidth=3)
    
    # Plot training data
    sns.histplot(ax=axes[1], data=training_return_data[:, stock_i], bins=bin_edges, alpha=1,
                stat="proportion", color="C2", label="Training")
    axes[1].plot(bin_edges[1:], cdf_diff, label='Ground truth', color="C3", linestyle="--", linewidth=3)

    # Configure axes
    for ax in axes:
        ax.set_xlim(-x_bound, x_bound)
        ax.set_xticks(range(-x_bound, x_bound+1, 1))
        ax.tick_params(axis='x', labelsize=12)
        ax.set_ylim(0, y_bound)
        ax.set_yticks(np.linspace(0, y_bound, 4))
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(fontsize=12, loc='upper right')
        
        # Set border style
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linestyle("-")
            spine.set_linewidth(1)

    # Add zoom-in effect if requested
    if zoomin_bound > 0:
        def custom_formatter(x, pos):
            return f"{x:.2f}"

        for ax in axes:
            axins = ax.inset_axes([0.1, 0.4, 0.28, 0.5])
            sns.histplot(ax=axins, data=generated_return_data[:, stock_i] if ax == axes[0] else training_return_data[:, stock_i],
                        bins=bins_num if ax == axes[0] else bin_edges, alpha=1,
                        color="C0" if ax == axes[0] else "C2", stat="proportion")
            axins.plot(bin_edges[1:], cdf_diff, color="C3", linestyle="--", linewidth=2)
            
            axins.set_xlim(-zoomin_bound, zoomin_bound)
            axins.set_ylim(0, y_bound)
            axins.set_yticks([])
            axins.set_ylabel("")
            axins.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
            
            for spine in axins.spines.values():
                spine.set_linestyle((0, (5, 4, 1, 4)))
                spine.set_linewidth(1)

    plt.tight_layout()
    plt.show()
    
    return bin_edges


def svd(A, k):
    """
    Perform Singular Value Decomposition (SVD) on the given matrix and return the top k eigenvalues and eigenvectors.
    
    Args:
        A: Input matrix
        k: Number of top components to return
        
    Returns:
        eigenvalues: Diagonal matrix of top k eigenvalues
        latent_subspace: Matrix of top k components
    """
    U, S, VT = np.linalg.svd(A)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    A_k = U_k @ S_k @ VT_k
    return np.diag(S_k), A_k

def calculate_latent_subspace(cov_matrix, latent_dim, save_path, prefix):
    """
    Calculate and save the latent subspace for a given covariance matrix.
    
    Args:
        cov_matrix: Input covariance matrix
        latent_dim: Dimension of the latent space
        save_path: Base path to save files
        prefix: Prefix for file naming
    """
    eigenvalues, latent_subspace = svd(cov_matrix, latent_dim)
    
    np.save(f"{save_path}/{prefix}_eigenvalues.npy", eigenvalues)
    np.save(f"{save_path}/{prefix}_latent_subspace.npy", latent_subspace)

def calculate_frobenius_norm_errors(synthetic_sample_latent_subspace, real_sample_latent_subspace, real_latent_subspace, output_path):
    """
    Calculate and save the Frobenius norm errors for synthetic and real sample covariance matrices.
    
    Parameters:
    - synthetic_sample_latent_subspace: Synthetic sample latent subspace matrix
    - real_sample_latent_subspace: Real sample latent subspace matrix
    - real_latent_subspace: Ground truth latent subspace matrix
    - output_path: Path to save the results
    
    Returns:
    - None
    """
    # Calculate Frobenius norm errors
    synthetic_error = (np.linalg.norm(synthetic_sample_latent_subspace - real_latent_subspace, ord=None) / np.linalg.norm(real_latent_subspace, ord=None)).round(3)
    real_error = (np.linalg.norm(real_sample_latent_subspace - real_latent_subspace, ord=None) / np.linalg.norm(real_latent_subspace, ord=None)).round(3)
    ratio = (synthetic_error / real_error).round(3)
    
    # Save results
    results = {
        'synthetic_error': synthetic_error,
        'real_error': real_error,
        'ratio': ratio
    }
    np.save(output_path, results)
    return None

def calculate_eigenvalues_relative_error(synthetic_sample_eigenvalues, real_sample_eigenvalues, real_eigenvalues, output_path):
    """
    Calculate and save the relative errors of eigenvalues for synthetic and real samples.
    
    Parameters:
    - synthetic_sample_eigenvalues: Eigenvalues from synthetic samples
    - real_sample_eigenvalues: Eigenvalues from real samples
    - real_eigenvalues: Ground truth eigenvalues
    - output_path: Path to save the results
    
    Returns:
    - None
    """
    # Calculate relative errors
    synthetic_error = np.abs(synthetic_sample_eigenvalues/real_eigenvalues - 1).mean().round(3)
    real_error = np.abs(real_sample_eigenvalues/real_eigenvalues - 1).mean().round(3)
    ratio = (synthetic_error / real_error).round(3)
    
    # Save results
    results = {
        'synthetic_error': synthetic_error,
        'real_error': real_error,
        'ratio': ratio
    }
    np.save(output_path, results)
    return None