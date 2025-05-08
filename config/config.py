"""
Configuration for Diffusion Factor Model
"""

import os
import torch
import numpy as np

# Project directory paths - Using relative paths from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "model_results")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Experiment naming
EXP_PREFIX = "dfm"  # Prefix for experiment IDs

# Core settings
SEED = 1234
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model parameters
MODEL_DIM = 128            # Base dimension for U-Net
MODEL_CHANNELS = 1        # Number of channels in input data
MODEL_FILTER_SIZE = 7     # Filter size for convolutions

# Dimension multipliers for different input sizes
DIM_MULTS_LARGE = (1, 2, 4, 16)     # For inputs where min_dim >= 32
DIM_MULTS_MEDIUM = (1, 2, 4)       # For inputs where min_dim >= 16
DIM_MULTS_SMALL = (1, 2)           # For inputs where min_dim >= 8
DIM_MULTS_TINY = (1, 2)            # For inputs where min_dim >= 4
DIM_MULTS_MINIMAL = (1,)           # For very small inputs

# Diffusion parameters
TIMESTEPS = 200
OBJECTIVE = 'pred_noise'
BETA_SCHEDULE = 'cosine'
AUTO_NORMALIZE = False

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 300
WEIGHT_DECAY = 0.01
USE_COSINE_SCHEDULER = True
USE_WARM_UP = True
WARMUP_STEPS = 10
COSINE_CYCLE_LENGTH = 40  # T_MAX
COSINE_LR_MIN = 1e-6      # ETA_MIN
GRADIENT_ACCUMULATION = 1
EMA_DECAY = 0.999
SPLIT_BATCHES = False
SAVE_INTERVAL = 50        # Save checkpoint every N epochs

# Sampling parameters
SAMPLE_BATCHES = 128      # Number of batches to sample
SAMPLES_PER_BATCH = 64    # Number of samples per batch

# Mixed precision settings
USE_AMP = True            # Mixed precision training

# Data parameters
TRAIN_SAMPLES = 2**11     # Number of samples to use for training

# File naming and paths
def get_experiment_id(seed=None, num_samples=None):
    """Generate a unique experiment identifier"""
    if seed is None:
        seed = SEED
    
    if num_samples is None:
        num_samples = TRAIN_SAMPLES
    
    return (f"finance_2D_dim{MODEL_DIM}_"
           f"latent{MODEL_DIM}_"
           f"Tmax{COSINE_CYCLE_LENGTH}_"
           f"etamin{COSINE_LR_MIN}_"
           f"batchsize{BATCH_SIZE}_"
           f"samples{int(np.log2(num_samples))}_"
           f"seed{seed}")

def set_seed(seed=None):
    """Set random seed for reproducibility"""
    if seed is None:
        seed = SEED
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    return seed

def get_model_path(exp_id=None):
    """Get the full path to the model directory for an experiment"""
    if exp_id is None:
        exp_id = get_experiment_id()
    return os.path.join(MODELS_DIR, exp_id)

def get_samples_path(exp_id=None):
    """Get the full path to the samples directory for an experiment"""
    if exp_id is None:
        exp_id = get_experiment_id()
    return os.path.join(SAMPLES_DIR, exp_id) 
