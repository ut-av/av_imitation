import argparse
import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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
    input_shape = sample_batch['image'].shape # (B, C, H, W)
    action_shape = sample_batch['action'].shape # (B, T, 2)
    
    input_channels = input_shape[1]
    output_steps = action_shape[1]
    
    print(f"Input Channels: {input_channels}")
    print(f"Output Steps: {output_steps}")
    
    # Initialize Model
    if args.model == 'cnn':
        model = CNN(input_channels, output_steps, dropout=args.dropout)
    elif args.model == 'mlp':
        model = MLP(input_channels, output_steps, dropout=args.dropout)
    elif args.model == 'transformer':
        model = Transformer(input_channels, output_steps, dropout=args.dropout)
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
        
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            actions = batch['action'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
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
