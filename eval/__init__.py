"""
Evaluation package for Diffusion Factor Model.

This package provides functions for evaluating diffusion models, including:
- Mean and covariance calculations
- Simulation-based evaluations
- Mean-variance portfolio evaluation
- Factor-timing portfolio evaluation
"""

from eval.mean_cov import calculate_mean_cov
from eval.simulation_eval import (
    comparision_histplot_simulation,
    calculate_latent_subspace,
    calculate_frobenius_norm_errors,
    calculate_eigenvalues_relative_error
)
from eval.mv_portfolio_eval import (
    comparision_histplot,
    calculate_objective_constrained,
    calculate_all_weights,
    calculate_portfolio_returns,
    calculate_portfolio_statistics
)
from eval.ft_portfolio_eval import (
    POET,
    calculate_FF_tangent_portfolio,
    calculate_Emp_tangent_portfolio,
    calculate_POET_tangent_portfolio,
    plot_cumulative_log_returns
)

__all__ = [
    # Mean and covariance utilities
    'calculate_mean_cov',
    
    # Simulation evaluation
    'comparision_histplot_simulation',
    'calculate_latent_subspace',
    'calculate_frobenius_norm_errors',
    'calculate_eigenvalues_relative_error',
    
    # Mean-variance portfolio evaluation
    'comparision_histplot',
    'calculate_objective_constrained',
    'calculate_all_weights',
    'calculate_portfolio_returns',
    'calculate_portfolio_statistics',
    
    # Factor timing portfolio evaluation
    'POET',
    'calculate_FF_tangent_portfolio',
    'calculate_Emp_tangent_portfolio',
    'calculate_POET_tangent_portfolio',
    'plot_cumulative_log_returns'
] 