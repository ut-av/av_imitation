import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

class AVDataset(Dataset):
    def __init__(self, metadata_file, split='train', transform=None):
        """
        metadata_file: Path to the dataset metadata json file.
        split: 'train', 'val', or 'test'.
        transform: Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.split = split
        
        with open(metadata_file, 'r') as f:
            meta = json.load(f)
        
        # Store parameters directly
        self.parameters = meta.get('parameters', {})
        
        print(f"Loaded metadata file {metadata_file} for dataset {meta['dataset_name']} with parameters {self.parameters}.")
            
        # Handle path mismatch if dataset was generated on another machine
        # We assume the data is located relative to the rosbags_processed directory
        # which is the parent of the datasets directory where the metadata file is.
        # So if metadata_file is /a/b/data/rosbags_processed/datasets/xyz.json
        # then root_dir should probably be /a/b/data/rosbags_processed
        
        default_root = os.path.dirname(os.path.dirname(os.path.abspath(metadata_file)))
        
        # Get root_dir from local meta
        meta_root_dir = meta['root_dir']
        
        # Check if the metadata root_dir exists
        if os.path.exists(meta_root_dir):
            self.root_dir = meta_root_dir
        else:
            print(f"Warning: Metadata root_dir {meta_root_dir} does not exist. Using inferred root {default_root}")
            self.root_dir = default_root
        self.samples = self._filter_split(meta['samples'], split)
        
        # Cache for images if needed? No, let's load on fly.
        
    def _filter_split(self, samples, split):
        # Deterministic split based on bag name or hash
        # Or maybe the metadata file already has splits?
        # The requirement says "handles train/test/val splits so that the splits are the same each time".
        # We can hash the sample ID or bag name.
        # Better: Split by bag to avoid data leakage (highly correlated frames).
        
        # Group samples by bag
        bag_samples = {}
        for s in samples:
            bag = s['bag']
            if bag not in bag_samples:
                bag_samples[bag] = []
            bag_samples[bag].append(s)
            
        bags = sorted(list(bag_samples.keys()))
        
        # Simple split: 70/15/15
        # Use a fixed seed for reproducibility
        import random
        random.seed(42)
        random.shuffle(bags)
        
        n = len(bags)
        if n > 1:
            n_val = max(1, int(0.15 * n))
        else:
            n_val = 0
            
        n_test = int(0.15 * n)
        n_train = n - n_val - n_test
        
        train_bags = set(bags[:n_train])
        val_bags = set(bags[n_train:n_train+n_val])
        test_bags = set(bags[n_train+n_val:])
        
        filtered = []
        if split == 'train':
            target_bags = train_bags
        elif split == 'val':
            target_bags = val_bags
        else:
            target_bags = test_bags
            
        for s in samples:
            if s['bag'] in target_bags:
                filtered.append(s)
                
        return filtered

    def __len__(self):
        return len(self.samples)

    def _fix_path(self, path):
        """Fix path if it's absolute from another machine."""
        # If path is relative, join with root_dir
        if not os.path.isabs(path):
            return os.path.join(self.root_dir, path)
            
        # If path is absolute and exists, return it
        if os.path.exists(path):
            return path
            
        # If absolute and doesn't exist, try to relativize it
        # Assume standard structure: .../rosbags_processed/...
        if 'rosbags_processed' in path:
            # Split and keep everything after the first occurrence of rosbags_processed
            # actually root_dir typically points to rosbags_processed
            # So we want the part INSIDE rosbags_processed
            parts = path.split('rosbags_processed/')
            if len(parts) > 1:
                rel_path = parts[-1]
                # If root_dir ends with rosbags_processed, join directly
                if self.root_dir.endswith('rosbags_processed'):
                    return os.path.join(self.root_dir, rel_path)
                else:
                    return os.path.join(self.root_dir, 'rosbags_processed', rel_path)
        
        # Fallback: just return assuming it might work or let it fail
        return path

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load images
        # History images
        history_imgs = []
        history_imgs = []
        for img_path in sample_info['history_images']:
            full_path = self._fix_path(img_path)
            img = cv2.imread(full_path)
            if img is None:
                raise RuntimeError(f"Failed to load image: {full_path}")
            history_imgs.append(img)
            
            
        # Current image
        curr_path = self._fix_path(sample_info['current_image'])
        curr_img = cv2.imread(curr_path)
        if curr_img is None:
            raise RuntimeError(f"Failed to load image: {curr_path}")
             
        # Stack images? Or return list?
        # Usually stack channel-wise or sequence dimension.
        # Let's stack channel-wise for history + current
        # If history is [t-2, t-1], and current is t.
        # We might want to return them as a tensor (T, C, H, W) or (T*C, H, W).
        
        all_imgs = history_imgs + [curr_img]
        # Convert to tensor
        tensors = []
        
        # Get channel configuration from metadata parameters
        # Default to rgb if not specified
        channels_conf = self.parameters.get('channels', 'rgb')
        
        for img in all_imgs:
            # Handle channels
            if channels_conf == 'gray':
                 # Ensure 1 channel
                 if img.ndim == 3:
                     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                 if img.ndim == 2:
                     img = img[:, :, np.newaxis]
            elif channels_conf == 'hsv':
                 # Convert BGR to HSV
                 if img.ndim == 3:
                     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            else: # rgb
                 # BGR to RGB
                 if img.ndim == 3:
                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                 elif img.ndim == 2:
                     # If gray but expected RGB, convert back?
                     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # HWC -> CHW
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(img_tensor)
            
        input_tensor = torch.stack(tensors) # (T, C, H, W)
        
        # Targets
        # Future actions
        actions = torch.tensor(sample_info['future_actions'], dtype=torch.float32)
        
        sample = {
            'image': input_tensor,
            'action': actions,
            'metadata': sample_info # For debugging/visualization
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def get_dataloader(metadata_file, split='train', batch_size=32, shuffle=True, 
                   num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True):
    dataset = AVDataset(metadata_file, split=split)
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                           num_workers=num_workers, pin_memory=pin_memory, 
                           prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
