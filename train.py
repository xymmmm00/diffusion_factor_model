"""
Training script for Diffusion Factor Model
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset
import os
import gc
import argparse
import time

from diffusion_factor_model import Unet, GaussianDiffusion, Trainer
import config.config as config

def get_dim_mults_for_size(height, width):
    """
    Determine appropriate dimension multipliers for UNet based on input dimensions.
    The dimensions must be divisible by the maximum downsampling factor.
    
    Args:
        height: Height of input
        width: Width of input
        
    Returns:
        Tuple of dimension multipliers suitable for the input size
    """
    # Calculate the maximum downsampling factor possible
    min_dim = min(height, width)
    
    if min_dim >= 32:
        return config.DIM_MULTS_LARGE  # Standard for large inputs
    elif min_dim >= 16:
        return config.DIM_MULTS_MEDIUM  # For medium inputs
    elif min_dim >= 8:
        return config.DIM_MULTS_SMALL   # For small inputs
    elif min_dim >= 4:
        return config.DIM_MULTS_TINY    # For very small inputs
    else:
        return config.DIM_MULTS_MINIMAL # Minimal case

def train_model(data_path, seed=None, num_samples=None, gpu_id=0, epochs=None):
    """
    Train the diffusion model using a specific data file
    
    Args:
        data_path: Path to the data file to use for training
        seed: Random seed for reproducibility
        num_samples: Number of training samples to use (None = use all)
        gpu_id: GPU ID to use
        epochs: Number of epochs to train (None = use config.EPOCHS)
    """
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Set seed and get timestamp for experiment ID
    seed = config.set_seed(seed)
    timestamp = int(time.time())
    
    # Get filename from path for experiment ID
    filename = os.path.basename(data_path)
    data_id = os.path.splitext(filename)[0]
    
    # Create experiment ID
    exp_id = f"{config.EXP_PREFIX}_{data_id}_ts{timestamp}_seed{seed}"
    
    # Load data to determine shape and dimensions
    data_np = np.load(data_path)
    data_shape = data_np.shape
    print(f"Loaded data with shape: {data_shape}, dtype: {data_np.dtype}")
    
    # Limit number of samples if specified
    if num_samples is not None and num_samples < data_shape[0]:
        data_np = data_np[:num_samples]
        print(f"Using {num_samples} samples from the data")
    
    # Determine data dimensions and reshape strategy
    if len(data_shape) == 2:
        # data (samples, features) - reshape to 2D format
        samples, features = data_shape
        
        # Try to make the image as square as possible
        height = int(np.sqrt(features))
        width = features // height
        
        if height * width != features:
            # If not perfectly divisible, use a simple reshape
            height, width = 1, features
        
        # Reshape data to [samples, 1, height, width]
        data = torch.from_numpy(data_np).float()
        if data.shape[1] != features:
            print(f"Warning: Data dimension ({data.shape[1]}) doesn't match expected features ({features})")
        
        data = data.reshape(-1, 1, height, width)
        print(f"Reshaped 2D data to: {data.shape} with dimensions [batch, channels, height={height}, width={width}]")
        
    elif len(data_shape) == 3:
        # data (samples, height, width) - add channel dimension
        samples, height, width = data_shape
        
        # Convert to tensor and add channel dimension
        data = torch.from_numpy(data_np).float()
        data = data.unsqueeze(1)  # Add channel dimension [samples, 1, height, width]
        print(f"Reshaped 3D data to: {data.shape} with dimensions [batch, channels, height={height}, width={width}]")
        
    else:
        raise ValueError(f"Unsupported data shape: {data_shape}, expected 2D or 3D array")
    
    # Get appropriate dimension multipliers for UNet
    dim_mults = get_dim_mults_for_size(height, width)
    print(f"Using dimension multipliers: {dim_mults} for input size ({height}, {width})")
    
    # Calculate maximum downsampling factor
    max_downsample = 2**(len(dim_mults)-1) if len(dim_mults) > 0 else 1
    print(f"Maximum downsampling factor: {max_downsample}")
    
    # Create directories for this experiment
    model_dir = os.path.join(config.MODELS_DIR, exp_id)
    sample_dir = os.path.join(config.SAMPLES_DIR, exp_id)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create dataset
    dataset = TensorDataset(data)
    
    # Calculate latent dimension (total number of features)
    latent_dim = height * width
    
    # Use epochs from argument or config
    if epochs is None:
        epochs = config.EPOCHS
    
    # Initialize model with appropriate dimension multipliers
    model = Unet(
        dim=config.MODEL_DIM,
        channels=config.MODEL_CHANNELS,
        filter_size=config.MODEL_FILTER_SIZE,
        dim_mults=dim_mults  # Use appropriate multipliers for this input size
    )
    
    print("Model initialized")
    
    # Initialize diffusion process with proper image size
    diffusion = GaussianDiffusion(
        model,
        image_size=(height, width),  # Use our reshaped dimensions
        latent_dim=latent_dim,
        timesteps=config.TIMESTEPS,
        objective=config.OBJECTIVE,
        beta_schedule=config.BETA_SCHEDULE,
        auto_normalize=config.AUTO_NORMALIZE
    )
    
    print("Diffusion process initialized")
    
    # Initialize Trainer with custom epochs
    trainer = Trainer(
        diffusion,
        dataset,
        train_batch_size=min(config.BATCH_SIZE, len(dataset)),  # Ensure batch size doesn't exceed dataset size
        train_lr=config.LEARNING_RATE,
        train_epochs=epochs,
        adamw_weight_decay=config.WEIGHT_DECAY,
        cosine_scheduler=config.USE_COSINE_SCHEDULER,
        warm_up=config.USE_WARM_UP,
        warmup_iters=config.WARMUP_STEPS,
        T_max=config.COSINE_CYCLE_LENGTH,
        eta_min=config.COSINE_LR_MIN,
        gradient_accumulate_every=config.GRADIENT_ACCUMULATION,
        ema_decay=config.EMA_DECAY,
        split_batches=config.SPLIT_BATCHES,
        save_and_sample_every=config.SAVE_INTERVAL,
        results_folder=model_dir,
        param_path="",
        amp=config.USE_AMP,
    )
    
    print("Trainer initialized")
    
    # Train model
    print(f"Starting training for {epochs} epochs...")
    trainer.train()
    
    # Generate samples
    print("Generating samples...")
    sample_batches = config.SAMPLE_BATCHES
    samples_per_batch = config.SAMPLES_PER_BATCH
    
    config.set_seed(seed)  # Reset seed for reproducibility
    
    for i in range(sample_batches):
        samples = diffusion.sample(batch_size=samples_per_batch)
        samples = samples.view(samples.size(0), -1).cpu().numpy()
        
        sample_file = os.path.join(sample_dir, f"sample_batch{i+1}.npy")
        np.save(sample_file, samples)
        
        # Clean up to prevent memory issues
        del samples
        gc.collect()
    
    # Clean up
    del trainer, model, diffusion, data, dataset
    gc.collect()
    
    print(f"Training and sampling complete for {exp_id}")
    print(f"Models saved to: {model_dir}")
    print(f"Samples saved to: {sample_dir}")
    
    return model_dir, sample_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion factor model on specific data file")
    parser.add_argument("--data_path", type=str, required=True, 
                      help="Path to the data file for training")
    parser.add_argument("--seed", type=int, default=None, 
                      help="Random seed")
    parser.add_argument("--num_samples", type=int, default=None, 
                      help="Number of training samples (None = use all)")
    parser.add_argument("--gpu", type=int, default=0, 
                      help="GPU ID")
    parser.add_argument("--epochs", type=int, default=None, 
                      help="Number of epochs to train (None = use config value)")
    
    args = parser.parse_args()
    
    train_model(args.data_path, args.seed, args.num_samples, args.gpu, args.epochs) 