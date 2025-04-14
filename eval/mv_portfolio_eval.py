import torch
import pandas as pd
import numpy as np
import numba
import scipy
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
import sys
import seaborn as sns
import os

sns.set_style('white')

def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

def comparision_histplot(stock_i, training_data_path, generated_data_path, bins_num=50, x_bound=3, y_bound=0.04, zoomin_bound=0.5):
    """
    Plot histogram comparison between generated and training data for a given stock.
    
    Args:
        stock_i (int): Index of the stock to plot
        training_data_path (str): Path to training data file
        generated_data_path (str): Path to generated data file
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

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=400)
    
    # Plot generated data
    sns.histplot(ax=axes[0], data=generated_return_data[:, stock_i], bins=bins_num, alpha=1,
                stat="proportion", color="C0", label="Generated")
    bin_edges = np.histogram(generated_return_data[:, stock_i], bins=bins_num, density=False)[1]
    
    # Plot training data
    sns.histplot(ax=axes[1], data=training_return_data[:, stock_i], bins=bin_edges, alpha=1,
                stat="proportion", color="C2", label="Training")

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


def calculate_objective_constrained(mu, sigma, eta=1, lower_bound=-np.inf, upper_bound=np.inf):
    """
    Solve the constrained optimization problem using MOSEK:
    max w^T * mu - 0.5 * eta * lambda * w^T * sigma * w
    subject to sum(w) = 1.
    """
    n = len(mu)  # Number of asset
    mu = mu.astype(np.float64)
    sigma = sigma.astype(np.float64)

    with Model("Portfolio Optimization") as M:
        
        # Defines the variables (holdings). Shortselling is not allowed.
        w = M.variable("w", n, Domain.greaterThan(lower_bound)) # Portfolio variables
        s = M.variable("s", 1, Domain.unbounded())      # Variance variable
        GT = np.linalg.cholesky(sigma).T

        # Total budget constraint
        M.constraint('budget', Expr.sum(w), Domain.equalsTo(1.0))
        M.constraint('max_w', w, Domain.lessThan(upper_bound))

        # Computes the risk
        M.constraint('variance', Expr.vstack(s, 0.5, GT @ w), Domain.inRotatedQCone())

        # Define objective as a weighted combination of return and variance
        M.objective('obj', ObjectiveSense.Maximize, w.T @ mu - s * 0.5 * eta)

        # Solve the problem
        M.solve()

        # Obtain the result
        w_optimal = w.level()

    return w_optimal


def calculate_all_weights(mean_paths, cov_paths, bound=0.05, eta=3, year=2001):
    """
    Calculate portfolio weights using different methods.
    
    Parameters:
    - mean_paths: Dictionary of mean file paths for each method
    - cov_paths: Dictionary of covariance file paths for each method
    - bound: Weight bound for optimization
    - eta: Risk aversion parameter
    - year: Base year for data
    
    Returns:
    - Dictionary containing weights for different methods
    """
    results = {
        'Diff_Emp_Method': [], 
        'Diff_Shr_Method': [], 
        'E_Diff_Method': [], 
        'EW_Method': [], 
        'VW_Method': [], 
        'Emp_Method': [], 
        'Shr_Method': []
    }
    
    # Diff Method
    w_diff = calculate_objective_constrained(np.load(mean_paths['diff_emp']), np.load(cov_paths['diff_emp']), eta, -bound, bound)
    results['Diff_Emp_Method'].append(w_diff / w_diff.sum())
    
    # Diff_Shr Method
    w_diff_shr = calculate_objective_constrained(np.load(mean_paths['diff_shr']), np.load(cov_paths['diff_shr']), eta, -bound, bound)
    results['Diff_Shr Method'].append(w_diff_shr / w_diff_shr.sum())
    
    # EmpDiff Method
    w_e_diff = calculate_objective_constrained(np.load(mean_paths['e_diff']), np.load(cov_paths['e_diff']), eta, -bound, bound)
    results['E_Diff_Method'].append(w_e_diff / w_e_diff.sum())
    
    # Equal Weight Method
    n_assets = len(results['Diff_Emp_Method'][0])
    results['EW_Method'].append(np.ones(n_assets)/n_assets)
    
    # Value Weight Method
    w_vw = pd.read_csv(mean_paths['vw'], index_col=0).iloc[0, :].values
    w_vw = w_vw / w_vw.sum()
    w_vw[w_vw > 0.05] = 0.05
    results['VW_Method'].append(w_vw / w_vw.sum())
    
    # Empirical Method
    w_emp = calculate_objective_constrained(np.load(mean_paths['emp']), np.load(cov_paths['emp']), eta, -bound, bound)
    results['Emp_Method'].append(w_emp / w_emp.sum())
    
    # Shrinkage Method
    w_shr = calculate_objective_constrained(np.load(mean_paths['shr']), np.load(cov_paths['shr']), eta, -bound, bound)
    results['Shr_Method'].append(w_shr / w_shr.sum())
    
    return results

def calculate_portfolio_returns(returns, weights, transaction_fee_rate=0.002):
    """
    Calculate portfolio returns with transaction costs.
    
    Args:
        returns (np.ndarray): Daily returns matrix of shape (n_days, n_assets)
        weights (np.ndarray): Portfolio weights matrix of shape (n_days, n_assets)
        transaction_fee_rate (float): Transaction fee rate per trade, default 0.002 (0.2%)
    
    Returns:
        tuple: (portfolio_returns, total_transaction_cost, turnover)
            - portfolio_returns (np.ndarray): Daily portfolio returns
            - total_transaction_cost (float): Total transaction costs incurred
            - turnover (float): Average daily turnover percentage
    """
    n_days, n_assets = returns.shape
    portfolio_returns = np.zeros(n_days)
    holdings = 1
    total_transaction_cost = 0
    turnover = 0

    for t in range(n_days-1):
        current_weight = weights[t] * (1 + returns[t])
        target_weight = weights[t+1]
        
        daily_transaction_cost = transaction_fee_rate * np.sum(np.abs(target_weight * np.sum(current_weight) - current_weight))
        total_transaction_cost += daily_transaction_cost
        
        portfolio_returns[t] = np.sum(weights[t] * returns[t]) - daily_transaction_cost
        holdings *= (1 + portfolio_returns[t])
        
        if np.sum(current_weight) > 0:
            turnover += 100 * np.sum(np.abs(target_weight - current_weight / np.sum(current_weight)))
    
    portfolio_returns[-1] = np.sum(weights[-1] * returns[-1])
    
    return portfolio_returns, total_transaction_cost, turnover / (n_days - 1) if n_days > 1 else 0

def calculate_portfolio_statistics(portfolio_returns, risk_free_data=None, eta=3):
    """
    Calculate portfolio performance statistics.
    
    Args:
        portfolio_returns (np.ndarray): Daily portfolio returns
        risk_free_data (np.ndarray, optional): Daily risk-free rate data
        eta (float): Risk aversion parameter, default 3
    
    Returns:
        dict: Dictionary containing portfolio statistics
            - Mean: Annualized mean return
            - Std: Annualized standard deviation
            - SR: Sharpe ratio
            - CER: Certainty equivalent return
            - CR: Cumulative return
            - MDD (%): Maximum drawdown percentage
    """
    # Calculate mean return
    if risk_free_data is not None:
        mean_return = np.mean(portfolio_returns - risk_free_data) * 252
    else:
        mean_return = np.mean(portfolio_returns) * 252
    
    # Calculate standard deviation
    std_dev = np.std(portfolio_returns) * np.sqrt(252)
    
    # Calculate Sharpe ratio
    sharpe_ratio = mean_return / std_dev
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = 100 * (peak - cumulative_returns) / (peak + 1e-5)
    max_drawdown = np.max(drawdown)
    
    # Calculate CER and cumulative return
    cer = mean_return - eta / 2 * (std_dev**2)
    cumulative_return = cumulative_returns[-1]
    
    return {
        "Mean": mean_return,
        "Std": std_dev,
        "SR": sharpe_ratio,
        "CER": cer,
        "CR": cumulative_return,
        "MDD (\%)": max_drawdown
    }

def calculate_portfolio_metrics(returns, weights, risk_free_data=None, transaction_fee_rate=0.002, eta=3):
    """
    Calculate comprehensive portfolio metrics.
    
    Args:
        returns (np.ndarray): Daily returns matrix of shape (n_days, n_assets)
        weights (np.ndarray): Portfolio weights matrix of shape (n_days, n_assets)
        risk_free_data (np.ndarray, optional): Daily risk-free rate data
        transaction_fee_rate (float): Transaction fee rate per trade, default 0.002 (0.2%)
        eta (float): Risk aversion parameter, default 3
    
    Returns:
        tuple: (portfolio_returns, metrics)
            - portfolio_returns (np.ndarray): Daily portfolio returns
            - metrics (dict): Dictionary containing portfolio statistics
    """
    portfolio_returns, _, turnover = calculate_portfolio_returns(returns, weights, transaction_fee_rate)
    metrics = calculate_portfolio_statistics(portfolio_returns, risk_free_data, eta)
    metrics["TO (\%)"] = turnover
    return portfolio_returns, metrics


def test_main(start_year, end_year, test_data_path, mean_paths_template, cov_paths_template, eta=3, fee=0.002):
    """
    Test portfolio performance over a specified period using different methods.
    
    Args:
        start_year (int): Starting year for testing
        end_year (int): Ending year for testing
        test_data_path (str): Template path for test data files, should contain {year} placeholder
        mean_paths_template (dict): Template paths for mean files, should contain {year} placeholder
        cov_paths_template (dict): Template paths for covariance files, should contain {year} placeholder
        eta (float): Risk aversion parameter, default 3
        fee (float): Transaction fee rate, default 0.002
        
    Returns:
        tuple: (portfolio_df, metrics_df)
            - portfolio_df: DataFrame containing daily portfolio returns
            - metrics_df: DataFrame containing portfolio performance metrics
    """
    # Initialize empty DataFrames for each method
    method_dfs = {
        'diff': pd.DataFrame(),
        'diff_shr': pd.DataFrame(),
        'emp_diff': pd.DataFrame(),
        'ew': pd.DataFrame(),
        'vw': pd.DataFrame(),
        'emp': pd.DataFrame(),
        'shr': pd.DataFrame()
    }
    final_test_data = pd.DataFrame()
    
    # Process each year in the test period
    for year in range(start_year, end_year):
        # Load and prepare test data
        test_data = pd.read_csv(
            test_data_path.format(year=year+5, next_year=year+6),
            index_col=0
        )
        test_data.index = pd.to_datetime(test_data.index)
        test_data.columns = test_data.columns.astype("int64")
        final_test_data = pd.concat([final_test_data, test_data], axis=0).fillna(0.0)
        
        # Format paths for the current year
        mean_paths = {k: v.format(year=year) for k, v in mean_paths_template.items()}
        cov_paths = {k: v.format(year=year) for k, v in cov_paths_template.items()}
        
        # Calculate weights for each method
        weights = calculate_all_weights(mean_paths, cov_paths, bound=0.05, eta=eta, year=year)
        
        # Update each method's DataFrame
        for method, df in zip(method_dfs.keys(), weights.values()):
            method_dfs[method] = pd.concat([method_dfs[method], df], axis=0).fillna(0.0)
    
    # Calculate portfolio metrics for each method
    portfolio_results = []
    summary_results = []
    for method_df in method_dfs.values():
        portfolio_returns, metrics = calculate_portfolio_metrics(
            final_test_data.values,
            method_df.values,
            transaction_fee_rate=fee,
            eta=eta
        )
        portfolio_results.append(portfolio_returns)
        summary_results.append(metrics)
    
    # Create DataFrames for results
    portfolio_index = [
        'Diff Method',
        'Diff_Shr Method',
        'EmpDiff Method',
        'EW Method',
        'VW Method',
        'Emp Method',
        'Shr Method'
    ]
    
    portfolio_df = pd.DataFrame(
        portfolio_results,
        index=portfolio_index,
        columns=final_test_data.index
    )
    metrics_df = pd.DataFrame(summary_results, index=portfolio_index)
    
    return portfolio_df, metrics_df