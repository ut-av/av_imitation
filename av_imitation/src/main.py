import argparse
import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader import get_dataloader
from models import CNN, MLP, Transformer

def train(args):
    # Setup paths
    dataset_dir = os.path.expanduser('~/roboracer_ws/data/rosbags_processed/datasets')
    metadata_file = os.path.join(dataset_dir, f"{args.dataset}.json")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Dataset metadata file not found: {metadata_file}")
        
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{timestamp}_{args.model}_{args.dataset}"
    experiment_dir = os.path.expanduser(f"~/roboracer_ws/data/experiments/{experiment_name}")
    models_dir = os.path.join(experiment_dir, "models")
    log_dir = os.path.join(experiment_dir, "tensorboard")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Starting experiment: {experiment_name}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset to get dimensions
    print("Loading dataset metadata...")
    
    # 1. Estimate Data Loader Memory Usage
    # Create a temporary loader to fetch one sample
    temp_loader = get_dataloader(metadata_file, split='train', batch_size=1, shuffle=False, 
                                num_workers=0, pin_memory=False, prefetch_factor=None, persistent_workers=False)
    
    print("Estimating memory usage...")
    sample_batch = next(iter(temp_loader))
    
    # Calculate memory per sample (in bytes)
    # image: (B, T, C, H, W) -> float32 (4 bytes) typically
    # action: (B, T, 2)
    img_mem = sample_batch['image'].element_size() * sample_batch['image'].nelement()
    act_mem = sample_batch['action'].element_size() * sample_batch['action'].nelement()
    # Add overhead safety factor (Python objects, internal buffers) - conservatively 1.5x
    sample_mem_est = (img_mem + act_mem) * 1.5
    
    batch_mem_est = sample_mem_est * args.batch_size
    print(f"Estimated memory per sample: {sample_mem_est / 1024**2:.2f} MB")
    print(f"Estimated memory per batch (batch_size={args.batch_size}): {batch_mem_est / 1024**2:.2f} MB")
    
    # Target Memory Limit for Data Loading Buffers (e.g. 90GB)
    MEMORY_LIMIT_GB = 90
    MEMORY_LIMIT_BYTES = MEMORY_LIMIT_GB * 1024**3
    
    # Total Buffer Memory ~= (num_workers * prefetch_factor * batch_size_memory)
    # We want Total Buffer Memory < MEMORY_LIMIT_BYTES
    
    # Solve for max_workers
    # max_workers = MEMORY_LIMIT / (prefetch_factor * batch_mem)
    prefetch_factor = args.prefetch_factor
    
    max_workers_mem = int(MEMORY_LIMIT_BYTES / (prefetch_factor * batch_mem_est))
    # Ensure at least 1 worker if possible, but dont crash
    max_workers_mem = max(0, max_workers_mem)
    
    print(f"Max workers allowed by memory limit ({MEMORY_LIMIT_GB} GB): {max_workers_mem}")
    
    # Select final num_workers
    train_num_workers = min(args.num_workers, max_workers_mem)
    # Drastically reduce val workers since validation is just inference
    val_num_workers = min(2, train_num_workers)
    
    print(f"Using Train Workers: {train_num_workers}")
    print(f"Using Val Workers: {val_num_workers}")

    # Helper to clean args for get_dataloader
    train_dl_kwargs = {
        'num_workers': train_num_workers,
        'pin_memory': args.pin_memory,
        'prefetch_factor': args.prefetch_factor if train_num_workers > 0 else None, # prefetch_factor requires num_workers > 0
        'persistent_workers': args.persistent_workers if train_num_workers > 0 else False
    }
    
    val_dl_kwargs = {
        'num_workers': val_num_workers,
        'pin_memory': args.pin_memory,
        'prefetch_factor': args.prefetch_factor if val_num_workers > 0 else None,
        'persistent_workers': args.persistent_workers if val_num_workers > 0 else False
    }
    
    if train_num_workers == 0:
         print("Warning: num_workers set to 0. Data loading will be on the main process and might be slow.")
    
    train_loader = get_dataloader(metadata_file, split='train', batch_size=args.batch_size, shuffle=True, **train_dl_kwargs)
    val_loader = get_dataloader(metadata_file, split='val', batch_size=args.batch_size, shuffle=False, **val_dl_kwargs)
    
    # Inspect one sample to determine input/output shapes (we already have one from temp_loader)
    # Re-use sample_batch but we need to check if dimensions match what model expects (B, T, C, H, W)
    # temp_loader had batch_size=1, so we should be careful.
    # The shapes in sample_batch are (1, ...).
    # Model expects (B, ...).
    
    input_shape = list(sample_batch['image'].shape)
    input_shape[0] = args.batch_size # Pretend it is the full batch size for logging
    input_shape = torch.Size(input_shape)
    
    print(f"Input Shape (simulated): {input_shape}")
    action_shape = sample_batch['action'].shape 
    print(f"Action Shape (from sample): {action_shape}")
    
    if len(input_shape) == 5:
        B, T, C, H, W = input_shape
    else:
        # Fallback for legacy or unexpected shapes
        # Assuming (B, C, H, W) where C includes time
        raise ValueError(f"Expected 5D input (B, T, C, H, W), got {input_shape}")
    
    output_steps = action_shape[1]
    input_channels = C
    
    print(f"Num Frames (T): {T}")
    print(f"Channels per Frame (C): {C}")
    print(f"Output Steps: {output_steps}")
    
    # Initialize Model
    if args.model == 'cnn':
        # CNN flattens T into channels
        model = CNN(T * C, output_steps, dropout=args.dropout)
    elif args.model == 'mlp':
        # MLP flattens T into channels/input dim
        model = MLP(T * C, output_steps, dropout=args.dropout)
    elif args.model == 'transformer':
        # Transformer handles T internally, takes channels per frame
        model = Transformer(C, output_steps, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
        
    model = model.to(device)
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for i, batch in enumerate(pbar):
            images = batch['image'].to(device)
            actions = batch['action'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                actions = batch['action'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, actions)
                val_loss += loss.item()
                
        if len(val_loader) > 0:
            avg_val_loss = val_loss / len(val_loader)
        else:
            avg_val_loss = 0.0
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}")
        
        # Logging
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        # Early Stopping and Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            model_path = os.path.join(models_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
            
            # Save training metadata
            # Save training metadata
            
            # Load dataset metadata to get color space and rates
            with open(metadata_file, 'r') as f:
                ds_meta = json.load(f)
            ds_params = ds_meta.get('parameters', {})
            color_space = ds_params.get('channels', 'rgb')
            history_rate = ds_params.get('history_rate')
            future_rate = ds_params.get('future_rate')
            
            training_meta = {
                'model_type': args.model,
                'dataset': args.dataset,
                'input_channels': input_channels, # This is usually per-frame * n_frames or just total channels
                'input_height': input_shape[3], # (B, T, C, H, W) -> H is index 3
                'input_width': input_shape[4],  # (B, T, C, H, W) -> W is index 4
                'history_frames': T,
                'future_frames': output_steps,
                'color_space': color_space,
                'history_rate': history_rate,
                'future_rate': future_rate,
                'dropout': args.dropout,
                'best_val_loss': best_val_loss,
                'epoch': epoch + 1,
            }
            meta_path = os.path.join(models_dir, "training_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(training_meta, f, indent=2)
            print(f"Saved training metadata to {meta_path}")
        else:
            patience_counter += 1
            
        if patience_counter >= args.early_stopping_epochs:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Car Action Prediction Model')
    
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'mlp', 'transformer'], help='Model architecture to use')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (without .json extension)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--early-stopping-epochs', type=int, default=10, help='Number of epochs to wait for improvement before early stopping')
    
    # Data Loading Optimizations
    parser.add_argument('--num-workers', type=int, default=min(os.cpu_count() or 4, 16), help='Number of data loading workers')
    parser.add_argument('--pin-memory', action='store_true', help='Pin memory for faster host-to-device transfer')
    parser.add_argument('--no-pin-memory', dest='pin_memory', action='store_false')
    parser.set_defaults(pin_memory=True)
    parser.add_argument('--prefetch-factor', type=int, default=2, help='Number of batches loaded in advance by each worker')
    parser.add_argument('--persistent-workers', action='store_true', help='Keep workers alive between epochs')
    parser.add_argument('--no-persistent-workers', dest='persistent_workers', action='store_false')
    parser.set_defaults(persistent_workers=True)
    
    args = parser.parse_args()
    train(args)
