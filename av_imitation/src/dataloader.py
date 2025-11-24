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
            self.meta = json.load(f)
            
        self.root_dir = self.meta['root_dir']
        self.samples = self._filter_split(self.meta['samples'], split)
        
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

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load images
        # History images
        history_imgs = []
        for img_path in sample_info['history_images']:
            full_path = os.path.join(self.root_dir, img_path)
            img = cv2.imread(full_path)
            if img is None:
                # Handle missing image?
                img = np.zeros((240, 320, 3), dtype=np.uint8) # Placeholder
            history_imgs.append(img)
            
        # Current image
        curr_path = os.path.join(self.root_dir, sample_info['current_image'])
        curr_img = cv2.imread(curr_path)
        if curr_img is None:
             curr_img = np.zeros((240, 320, 3), dtype=np.uint8)
             
        # Stack images? Or return list?
        # Usually stack channel-wise or sequence dimension.
        # Let's stack channel-wise for history + current
        # If history is [t-2, t-1], and current is t.
        # We might want to return them as a tensor (T, C, H, W) or (T*C, H, W).
        
        all_imgs = history_imgs + [curr_img]
        # Convert to tensor
        # Assume standard HWC -> CHW
        tensors = []
        for img in all_imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(img)
            
        input_tensor = torch.cat(tensors, dim=0) # ( (H+1)*C, H, W )
        
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

def get_dataloader(metadata_file, split='train', batch_size=32, shuffle=True, num_workers=4):
    dataset = AVDataset(metadata_file, split=split)
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
