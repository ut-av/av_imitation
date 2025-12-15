#!/usr/bin/env python3
"""
Script to convert trained PyTorch model weights to ONNX format for Jetson Orin deployment.

Usage:
    # Using experiment name (recommended - reads metadata automatically):
    ros2 run av_imitation export_onnx --experiment-name 20251124_235857_mlp_outdoor_v1 --output /path/to/model.onnx
    
    # Manual override (if metadata is missing):
    ros2 run av_imitation export_onnx --experiment-name 20251124_235857_mlp_outdoor_v1 --output /path/to/model.onnx \
                          --model cnn --input-channels 9 --output-steps 10
"""

import argparse
import os
import json

import torch
import torch.onnx

# Import models from the package
from av_imitation.src.models import CNN, CNNOnnx, MLP, MLPOnnx, Transformer, TransformerOnnx

# Default experiments directory
EXPERIMENTS_DIR = os.path.expanduser('~/roboracer_ws/data/experiments')
DATASETS_DIR = os.path.expanduser('~/roboracer_ws/data/rosbags_processed/datasets')


def load_dataset_metadata(dataset_name: str) -> dict:
    """
    Load dataset metadata to get input/output dimensions.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'outdoor_v1')
        
    Returns:
        Dictionary containing dataset parameters
    """
    dataset_path = os.path.join(DATASETS_DIR, f"{dataset_name}.json")
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset file not found: {dataset_path}")
        return {}
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Extract relevant info from samples
    samples = data.get('samples', [])
    
    result = {}
    if samples:
        sample = samples[0]
        # Number of history images + current image = total frames
        num_history = len(sample.get('history_images', []))
        # Input channels = (history + 1 current) * 3 RGB channels
        result['input_channels'] = (num_history + 1) * 3
        # Output steps = number of future actions
        result['output_steps'] = len(sample.get('future_actions', []))
        
        print(f"Inferred from dataset '{dataset_name}':")
        print(f"  History frames: {num_history}, Input channels: {result['input_channels']}")
        print(f"  Output steps: {result['output_steps']}")
    
    return result


def load_training_metadata(experiment_name: str) -> dict:
    """
    Load training metadata from an experiment by name.
    
    Args:
        experiment_name: Name of the experiment folder (e.g., '20251124_235857_mlp_outdoor_v1')
        
    Returns:
        Dictionary containing training metadata
    """
    experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
    
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Check for training_metadata.json in models subdirectory
    meta_path = os.path.join(experiment_dir, 'models', 'training_metadata.json')
    if not os.path.exists(meta_path):
        # Also check directly in experiment_dir
        meta_path = os.path.join(experiment_dir, 'training_metadata.json')
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    
    # Try to infer model type from experiment name
    # Expected format: {timestamp}_{model}_{dataset}
    # e.g., 20251124_235857_mlp_outdoor_v1
    inferred = {}
    parts = experiment_name.split('_')
    if len(parts) >= 3:
        # Skip timestamp parts (first two: date and time)
        model_type = parts[2].lower()
        if model_type in ['cnn', 'mlp', 'transformer']:
            inferred['model_type'] = model_type
            print(f"Inferred model type from experiment name: {model_type}")
        # Dataset name is the rest
        if len(parts) > 3:
            inferred['dataset'] = '_'.join(parts[3:])
    
    return inferred


def export_to_onnx(
    experiment_name: str = None,
    weights_path: str = None,
    output_path: str = None,
    model_type: str = None,
    input_channels: int = None,
    output_steps: int = None,
    input_height: int = None,
    input_width: int = None,
    dropout: float = None,
    opset_version: int = 11,
    dynamic_batch: bool = True,
):
    """
    Export a trained PyTorch model to ONNX format.
    
    Args:
        experiment_name: Name of the experiment folder (e.g., '20251124_235857_mlp_outdoor_v1')
        weights_path: Path to the saved .pth weights file (optional if experiment_name provided)
        output_path: Path to save the .onnx model
        model_type: Type of model ('cnn', 'mlp', 'transformer') - auto-detected from metadata if not provided
        input_channels: Number of input channels - auto-detected from metadata if not provided
        output_steps: Number of output timesteps - auto-detected from metadata if not provided
        input_height: Input image height - auto-detected from metadata if not provided
        input_width: Input image width - auto-detected from metadata if not provided
        dropout: Dropout rate - auto-detected from metadata if not provided
        opset_version: ONNX opset version (11+ recommended for Jetson)
        dynamic_batch: Whether to use dynamic batch size
    """
    # Load metadata from experiment if provided
    metadata = {}
    experiment_dir = None
    if experiment_name:
        experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
        metadata = load_training_metadata(experiment_name)
        if metadata:
            print(f"Loaded training metadata: {metadata}")
        
        # If we have a dataset name but missing input/output dimensions, load from dataset
        dataset_name = metadata.get('dataset')
        if dataset_name and (not metadata.get('input_channels') or not metadata.get('output_steps')):
            dataset_meta = load_dataset_metadata(dataset_name)
            # Merge dataset metadata (don't override existing values)
            for key, value in dataset_meta.items():
                if key not in metadata:
                    metadata[key] = value
        
        # Default weights path if not provided
        if not weights_path:
            weights_path = os.path.join(experiment_dir, 'models', 'best_model.pth')
    
    # Use metadata values as defaults, allow command-line overrides
    model_type = model_type or metadata.get('model_type')
    input_channels = input_channels or metadata.get('input_channels')
    output_steps = output_steps or metadata.get('output_steps')
    input_height = input_height or metadata.get('input_height', 240)
    input_width = input_width or metadata.get('input_width', 320)
    dropout = dropout if dropout is not None else metadata.get('dropout', 0.1)
    
    # Validate required parameters
    if not model_type:
        raise ValueError("model_type is required (either via --model or from training metadata)")
    if not input_channels:
        raise ValueError("input_channels is required (either via --input-channels or from training metadata)")
    if not output_steps:
        raise ValueError("output_steps is required (either via --output-steps or from training metadata)")
    if not weights_path:
        raise ValueError("weights_path is required (either via --weights or --experiment-name)")
    
    # Default output path: same as weights but with .onnx extension
    if not output_path:
        output_path = os.path.splitext(weights_path)[0] + '.onnx'
        print(f"Output path not specified, using: {output_path}")
    
    # Create model
    print(f"Creating {model_type} model...")
    if model_type == 'cnn':
        model = CNN(input_channels, output_steps, dropout=dropout)
    elif model_type == 'mlp':
        # Load into standard MLP first, then convert to ONNX-compatible version
        model = MLP(input_channels, output_steps, dropout=dropout)
    elif model_type == 'transformer':
        model = Transformer(input_channels, output_steps, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    print(f"Loading weights from {weights_path}...")
    weights_path = os.path.expanduser(weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Convert to ONNX-compatible version after loading weights
    if model_type == 'cnn':
        print("Converting CNN to ONNX-compatible version...")
        model = CNNOnnx.from_cnn(model)
    elif model_type == 'mlp':
        print("Converting MLP to ONNX-compatible version...")
        model = MLPOnnx.from_mlp(model)
    elif model_type == 'transformer':
        print("Converting Transformer to ONNX-compatible version...")
        model = TransformerOnnx.from_transformer(model)
    
    model.eval()
    
    # Create dummy input for tracing
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, input_channels, input_height, input_width)
    
    # Define input/output names
    input_names = ['image']
    output_names = ['actions']
    
    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'actions': {0: 'batch_size'}
        }
    
    # Ensure output directory exists
    output_path = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Export to ONNX
    print(f"Exporting to ONNX (opset version {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    
    print(f"Model exported to: {output_path}")
    
    # Save metadata alongside the ONNX model
    metadata = {
        'model_type': model_type,
        'input_channels': input_channels,
        'output_steps': output_steps,
        'input_height': input_height,
        'input_width': input_width,
        'n_frames': metadata.get('n_frames', 1), # Default to 1 if not present
        'color_space': metadata.get('color_space', 'rgb'),
        'opset_version': opset_version,
        'source_weights': os.path.basename(weights_path),
    }
    
    metadata_path = output_path.replace('.onnx', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    # Verify the exported model
    try:
        import onnx
        print("Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
        
        # Print model info
        print("\nModel Info:")
        print(f"  Input: {input_names[0]} - shape: (batch, {input_channels}, {input_height}, {input_width})")
        print(f"  Output: {output_names[0]} - shape: (batch, {output_steps}, 2)")
        
    except ImportError:
        print("Warning: 'onnx' package not installed. Skipping verification.")
        print("Install with: pip install onnx")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")
    
    # Optional: Test with ONNX Runtime
    try:
        import onnxruntime as ort
        print("\nTesting with ONNX Runtime...")
        
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        
        # Run inference with dummy input
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: dummy_input.numpy()})
        
        print("ONNX Runtime test passed!")
        print(f"  Output shape: {output[0].shape}")
        
    except ImportError:
        print("Warning: 'onnxruntime' package not installed. Skipping runtime test.")
        print("Install with: pip install onnxruntime")
    except Exception as e:
        print(f"Warning: ONNX Runtime test failed: {e}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert trained PyTorch model to ONNX format for Jetson Orin deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using experiment name (recommended - reads metadata automatically):
  ros2 run av_imitation export_onnx --experiment-name 20251124_235857_mlp_outdoor_v1 --output model.onnx
  
  # Manual override:
  ros2 run av_imitation export_onnx --experiment-name 20251124_235857_mlp_outdoor_v1 --output model.onnx --model cnn
"""
    )
    
    parser.add_argument(
        '--experiment-name', type=str,
        help='Name of the experiment folder in ~/roboracer_ws/data/experiments/ (e.g., 20251124_235857_mlp_outdoor_v1)'
    )
    parser.add_argument(
        '--weights', type=str,
        help='Path to the trained model weights (.pth file). Optional if --experiment-name is provided.'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output path for the ONNX model (.onnx file). Defaults to same path as weights with .onnx extension.'
    )
    parser.add_argument(
        '--model', type=str,
        choices=['cnn', 'mlp', 'transformer'],
        help='Model architecture type (auto-detected from metadata if not provided)'
    )
    parser.add_argument(
        '--input-channels', type=int,
        help='Number of input channels (auto-detected from metadata if not provided)'
    )
    parser.add_argument(
        '--output-steps', type=int,
        help='Number of output timesteps to predict (auto-detected from metadata if not provided)'
    )
    parser.add_argument(
        '--input-height', type=int,
        help='Input image height (auto-detected from metadata if not provided)'
    )
    parser.add_argument(
        '--input-width', type=int,
        help='Input image width (auto-detected from metadata if not provided)'
    )
    parser.add_argument(
        '--dropout', type=float,
        help='Dropout rate (auto-detected from metadata if not provided)'
    )
    parser.add_argument(
        '--opset-version', type=int, default=11,
        help='ONNX opset version (default: 11, recommended for TensorRT)'
    )
    parser.add_argument(
        '--no-dynamic-batch', action='store_true',
        help='Disable dynamic batch size (fix batch size to 1)'
    )
    
    args = parser.parse_args()
    
    # Validate that at least experiment-name or weights is provided
    if not args.experiment_name and not args.weights:
        parser.error("Either --experiment-name or --weights must be provided")
    
    export_to_onnx(
        experiment_name=args.experiment_name,
        weights_path=args.weights,
        output_path=args.output,
        model_type=args.model,
        input_channels=args.input_channels,
        output_steps=args.output_steps,
        input_height=args.input_height,
        input_width=args.input_width,
        dropout=args.dropout,
        opset_version=args.opset_version,
        dynamic_batch=not args.no_dynamic_batch,
    )


if __name__ == '__main__':
    main()
