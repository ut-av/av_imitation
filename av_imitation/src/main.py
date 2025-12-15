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
    print("Loading dataset...")
    train_loader = get_dataloader(metadata_file, split='train', batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(metadata_file, split='val', batch_size=args.batch_size, shuffle=False)
    
    # Inspect one sample to determine input/output shapes
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch['image'].shape # (B, T, C, H, W)
    print(f"Input Shape: {input_shape}")
    action_shape = sample_batch['action'].shape # (B, T, 2)
    print(f"Action Shape: {action_shape}")
    
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
            
            # Load dataset metadata to get color space
            with open(metadata_file, 'r') as f:
                ds_meta = json.load(f)
            color_space = ds_meta.get('parameters', {}).get('channels', 'rgb')
            
            training_meta = {
                'model_type': args.model,
                'dataset': args.dataset,
                'input_channels': input_channels, # This is usually per-frame * n_frames or just total channels
                'input_height': input_shape[3], # (B, T, C, H, W) -> H is index 3
                'input_width': input_shape[4],  # (B, T, C, H, W) -> W is index 4
                'n_frames': T,
                'color_space': color_space,
                'output_steps': output_steps,
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
    
    args = parser.parse_args()
    train(args)
