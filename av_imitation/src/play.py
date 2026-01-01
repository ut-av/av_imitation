#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from amrl_msgs.msg import AckermannCurvatureDriveMsg
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os

import json
from collections import deque

class ModelPlayer(Node):
    def __init__(self, experiment_name, camera_topic):
        super().__init__('model_player')
        
        self.declare_parameter('experiment_name', experiment_name)
        self.declare_parameter('camera_topic', camera_topic)
        
        self.experiment_name = self.get_parameter('experiment_name').get_parameter_value().string_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        
        # Construct paths
        experiment_dir = os.path.expanduser(f'~/roboracer_ws/data/experiments/{self.experiment_name}')
        self.model_path = os.path.join(experiment_dir, 'models', 'best_model.onnx')
        metadata_path = self.model_path.replace('.onnx', '_onnx_metadata.json')
        if not os.path.exists(metadata_path):
            self.get_logger().error(f"Metadata file not found: {metadata_path}")
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.meta = json.load(f)
        self.get_logger().info(f"Loaded metadata: {self.meta}")
        
        # Parse required metadata with strict checking
        required_keys = ['input_height', 'input_width', 'history_frames', 'color_space']
        for key in required_keys:
            if key not in self.meta:
                raise RuntimeError(f"'{key}' not found in model metadata")
                
        self.input_height = self.meta['input_height']
        self.input_width = self.meta['input_width']
        self.history_frames = self.meta['history_frames']
        self.color_space = self.meta['color_space']
            
        if 'history_rate' not in self.meta or self.meta['history_rate'] is None:
            raise RuntimeError("history_rate not found in model metadata")
        if 'future_rate' not in self.meta or self.meta['future_rate'] is None:
            raise RuntimeError("future_rate not found in model metadata")
            
        self.framerate = self.meta['history_rate']
        self.future_rate = self.meta['future_rate']

        self.get_logger().info(f"Model expects: {self.history_frames} history frames of {self.input_width}x{self.input_height} in {self.color_space}")
        self.get_logger().info(f"Control Rate: {self.framerate}Hz (history), Planning Rate: {self.future_rate}Hz (future)")

        # Load normalization parameters
        if 'mean' in self.meta and 'std' in self.meta:
            self.mean = np.array(self.meta['mean'], dtype=np.float32).reshape(-1, 1, 1)
            self.std = np.array(self.meta['std'], dtype=np.float32).reshape(-1, 1, 1)
            self.get_logger().info(f"Loaded normalization parameters. Mean shape: {self.mean.shape}")
        else:
            if self.color_space == 'gray':
                c = 1
            else:
                c = 3
            self.mean = np.zeros((c, 1, 1), dtype=np.float32)
            self.std = np.ones((c, 1, 1), dtype=np.float32)
            self.get_logger().warn("Normalization parameters not found in metadata. Using default (0 mean, 1 std).")
            
        # Output Normalization
        if 'action_mean' in self.meta and 'action_std' in self.meta:
            self.action_mean = np.array(self.meta['action_mean'], dtype=np.float32)
            self.action_std = np.array(self.meta['action_std'], dtype=np.float32)
            self.get_logger().info(f"Loaded ALL action normalization parameters.")
        else:
            self.action_mean = np.array([0.0, 0.0], dtype=np.float32)
            self.action_std = np.array([1.0, 1.0], dtype=np.float32)
            self.get_logger().warn("Action normalization parameters not found. Using raw output.")

        self.get_logger().info(f"Loading model from {self.model_path}")
        self.ort_session = ort.InferenceSession(self.model_path)
        
        # Frame history buffer
        self.history = deque(maxlen=self.history_frames)

        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10)
            
        self.publisher = self.create_publisher(AckermannCurvatureDriveMsg, '/ackermann_curvature_drive', 10)
        
        self.get_logger().info(f"Subscribed to {self.camera_topic}")
        self.get_logger().info("Publishing to /ackermann_curvature_drive")
        
        # Rate limiting state
        self.last_process_time = rclpy.time.Time(seconds=0, clock_type=self.get_clock().clock_type)
        
        # For debugging frequency
        self.last_inference_time = self.get_clock().now()
        
        # Optimization state
        self.last_inference_input_timestamp = rclpy.time.Time(seconds=0, clock_type=self.get_clock().clock_type)
        self.current_prediction = None
        self.prediction_step = 0
        
        # Create inference timer at prediction frequency
        self.create_timer(1.0/self.future_rate, self.inference_callback)

    def image_callback(self, msg):
        # Rate limiting: only process frames at history_rate
        now = self.get_clock().now()
        interval_ns = 1e9 / self.framerate
        
        # Check if enough time has passed
        if (now - self.last_process_time).nanoseconds < interval_ns:
            return
            
        self.last_process_time = now

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Preprocess using metadata dimensions
        input_img = cv2.resize(cv_image, (self.input_width, self.input_height))
        
        # Color space conversion
        if self.color_space == 'gray':
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            # Add channel dimension if needed (usually gray is HxW, but model might expect C=1)
            # If we transpose later (2, 0, 1), we need 3 dims.
            if input_img.ndim == 2:
                input_img = input_img[:, :, np.newaxis]
        elif self.color_space == 'hsv':
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        else: # rgb
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Normalize 0-1
        input_img = input_img.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W)
        input_img = input_img.transpose(2, 0, 1) # (C, H, W)
        
        # Apply Normalization
        input_img = (input_img - self.mean) / self.std
        
        # Update history
        self.history.append(input_img)
        
        # Wait until we have enough frames
        if len(self.history) < self.history_frames:
            # Pad with the first frame
            while len(self.history) < self.history_frames:
                self.history.appendleft(input_img)

    def inference_callback(self):
        # Ensure we have enough history
        if len(self.history) < self.history_frames:
            return

        # Calculate frequency
        now = self.get_clock().now()
        dt = (now - self.last_inference_time).nanoseconds / 1e9
        if dt > 0:
            freq = 1.0 / dt
        else:
            freq = 0.0
        self.last_inference_time = now
            
        # Check if inputs have changed
        inputs_changed = (self.last_process_time != self.last_inference_input_timestamp)
        
        if inputs_changed or self.current_prediction is None:
            # Stack frames in channel dimension: (T*C, H, W)
            # deque to list -> stack -> (n_frames, 3, H, W) -> reshape/concatenation
            # We want to concatenate along channel dimension
            # history[0] is (3, H, W), history[1] is (3, H, W)
            # concatenate -> (3*n_frames, H, W)
            stacked_img = np.concatenate(list(self.history), axis=0)
            
            # Add batch dimension (1, C_total, H, W)
            input_tensor = stacked_img[np.newaxis, ...]
            
            # Inference
            try:
                ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor}
                ort_outs = self.ort_session.run(None, ort_inputs)
                output = ort_outs[0] # (1, T_out, 2)
                
                self.current_prediction = output[0]
                self.prediction_step = 0
                self.last_inference_input_timestamp = self.last_process_time
                
            except Exception as e:
                self.get_logger().error(f"Inference failed: {e}")
                return
        else:
            self.prediction_step += 1
            if self.prediction_step >= len(self.current_prediction):
                self.get_logger().warn("Ran out of prediction steps")
                return
        
        # Extract action
        action_raw = self.current_prediction[self.prediction_step]
        
        # Un-scale action
        action_unscaled = action_raw * self.action_std + self.action_mean
        
        # Training data order is [curvature, velocity]
        curvature = float(action_unscaled[0])
        velocity = float(action_unscaled[1])

        # Debugging
        self.get_logger().debug(f"Inference Freq: {freq:.2f} Hz")
        if inputs_changed:
             self.get_logger().debug("New Inference")
        else:
             self.get_logger().debug(f"Reusing prediction step {self.prediction_step}")
             
        self.get_logger().info(f"Cmd: v={velocity:.3f}, c={curvature:.3f}")
            
        # Publish command
        cmd_msg = AckermannCurvatureDriveMsg()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = "base_link"
        cmd_msg.velocity = velocity
        cmd_msg.curvature = curvature
        
        self.publisher.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='Run ONNX model on car')
    parser.add_argument('--experiment-name', type=str, required=True, help='Name of experiment (e.g. 2025...)')
    parser.add_argument('--topic', type=str, default='/camera_0/image_raw', help='Camera topic')
    
    # Filter ROS args
    ros_args = rclpy.utilities.remove_ros_args(args)
    parsed_args = parser.parse_args(ros_args[1:]) # Skip script name

    node = ModelPlayer(parsed_args.experiment_name, parsed_args.topic)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import sys
    main(sys.argv)
