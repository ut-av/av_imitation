import cv2
import numpy as np
import os
import torch

class ImageProcessor:
    def __init__(self):
        self.sam_model = None
        self.depth_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process(self, image, options):
        """
        Apply a series of transformations to an image.
        options: dict with keys:
            - resolution: (width, height)
            - channels: 'rgb', 'gray', 'hsv'
            - canny: bool (or dict with thresholds)
            - sam3: bool
            - depth: bool
        """
        processed = image.copy()

        # 1. Resolution
        if 'resolution' in options and options['resolution']:
            w, h = options['resolution']
            processed = cv2.resize(processed, (w, h))

        # 2. Channels
        if 'channels' in options:
            if options['channels'] == 'gray':
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                # Keep it 3 channel for consistency? Or 1?
                # Usually standardizing on 3 channels is easier for visualization/training unless specified.
                # But 'gray' implies 1 channel.
                # Let's keep it 1 channel if requested, but visualization might need handling.
            elif options['channels'] == 'hsv':
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            # Default BGR (OpenCV standard)

        # 3. Canny Edges
        if 'canny' in options and options['canny']:
            # If processed is not gray, convert for Canny
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed
            
            # Default thresholds
            t1, t2 = 100, 200
            if isinstance(options['canny'], dict):
                t1 = options['canny'].get('threshold1', 100)
                t2 = options['canny'].get('threshold2', 200)
                
            edges = cv2.Canny(gray, t1, t2)
            # Canny returns 1 channel.
            processed = edges

        # 4. SAM3 (Segment Anything)
        if 'sam3' in options and options['sam3']:
            processed = self.apply_sam3(processed)

        # 5. Depth Estimation
        if 'depth' in options and options['depth']:
            processed = self.apply_depth(processed)

        return processed

    def apply_sam3(self, image):
        # Placeholder for SAM3
        # In a real implementation, we would load the model and run inference.
        # Since dependencies might be missing, we'll return a mock or try to load.
        print("SAM3 processing requested but not fully implemented/installed.")
        # For now, just draw a overlay to indicate "processed"
        cv2.putText(image, "SAM3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return image

    def apply_depth(self, image):
        # Placeholder for Depth
        print("Depth processing requested but not fully implemented/installed.")
        # Mock depth map (grayscale gradient)
        h, w = image.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            depth_map[i, :] = int(255 * (i / h))
        return depth_map

    def get_output_folder_name(self, options):
        """
        Generate a unique folder name based on options.
        """
        parts = []
        if 'resolution' in options:
            parts.append(f"res{options['resolution'][0]}x{options['resolution'][1]}")
        if 'channels' in options:
            parts.append(options['channels'])
        if 'canny' in options and options['canny']:
            parts.append("canny")
        if 'sam3' in options and options['sam3']:
            parts.append("sam3")
        if 'depth' in options and options['depth']:
            parts.append("depth")
            
        if not parts:
            return "raw"
            
        return "_".join(parts)
