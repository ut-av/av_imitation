#!/usr/bin/env python3
"""
ROS2 Humble inference node for running ONNX imitation learning models on Jetson Orin.

This node:
- Subscribes to camera images
- Maintains a history of frames for temporal context
- Runs the ONNX model to predict steering and throttle actions
- Publishes the predicted actions as AckermannDriveStamped messages

Usage:
    ros2 run av_imitation inference --ros-args -p model_path:=/path/to/model.onnx
"""

import os
import json
from collections import deque
from typing import Optional

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Header

from cv_bridge import CvBridge


class InferenceNode(Node):
    """ROS2 node for running ONNX model inference."""

    def __init__(self):
        super().__init__('imitation_inference')
        
        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('input_height', 240)
        self.declare_parameter('input_width', 320)
        self.declare_parameter('num_history_frames', 3)
        self.declare_parameter('use_tensorrt', True)
        self.declare_parameter('inference_rate', 30.0)
        self.declare_parameter('steering_scale', 1.0)
        self.declare_parameter('speed_scale', 1.0)
        self.declare_parameter('max_steering_angle', 0.4189)  # ~24 degrees in radians
        self.declare_parameter('max_speed', 3.0)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.input_height = self.get_parameter('input_height').get_parameter_value().integer_value
        self.input_width = self.get_parameter('input_width').get_parameter_value().integer_value
        self.num_history_frames = self.get_parameter('num_history_frames').get_parameter_value().integer_value
        self.use_tensorrt = self.get_parameter('use_tensorrt').get_parameter_value().bool_value
        self.inference_rate = self.get_parameter('inference_rate').get_parameter_value().double_value
        self.steering_scale = self.get_parameter('steering_scale').get_parameter_value().double_value
        self.speed_scale = self.get_parameter('speed_scale').get_parameter_value().double_value
        self.max_steering_angle = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        
        # Default channels per frame (assuming RGB if not specified)
        self.channels_per_frame = 3
        
        # Validate model path
        if not self.model_path:
            self.get_logger().error('Model path not specified! Use --ros-args -p model_path:=/path/to/model.onnx')
            raise ValueError('model_path parameter is required')
        
        self.model_path = os.path.expanduser(self.model_path)
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'Model file not found: {self.model_path}')
            raise FileNotFoundError(f'Model file not found: {self.model_path}')
        
        # Load model metadata if available
        self._load_metadata()
        
        # Initialize ONNX Runtime session
        self._init_onnx_session()
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Frame history buffer
        self.frame_history = deque(maxlen=self.num_history_frames)
        
        # Latest prediction
        self.latest_actions: Optional[np.ndarray] = None
        
        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            sensor_qos
        )
        
        # Publishers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.drive_topic,
            10
        )
        
        # Timer for inference at fixed rate
        timer_period = 1.0 / self.inference_rate
        self.inference_timer = self.create_timer(timer_period, self.inference_callback)
        
        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        self.get_logger().info('Inference node initialized')
        self.get_logger().info(f'  Model: {self.model_path}')
        self.get_logger().info(f'  Input: {self.num_history_frames} frames x {self.input_height}x{self.input_width}')
        self.get_logger().info(f'  Channels per frame: {self.channels_per_frame}')
        self.get_logger().info(f'  Image topic: {self.image_topic}')
        self.get_logger().info(f'  Drive topic: {self.drive_topic}')
        self.get_logger().info(f'  Inference rate: {self.inference_rate} Hz')
    
    def _load_metadata(self):
        """Load model metadata if available."""
        metadata_path = self.model_path.replace('.onnx', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Update parameters from metadata if not explicitly set
            if metadata.get('input_height'):
                self.input_height = metadata['input_height']
            if metadata.get('input_width'):
                self.input_width = metadata['input_width']
            
            # Load channels per frame
            if metadata.get('input_channels'):
                self.channels_per_frame = metadata['input_channels']
                
            # Load number of history frames
            if metadata.get('n_frames'):
                self.num_history_frames = metadata['n_frames']
            elif metadata.get('input_channels'):
                 # Legacy heuristic: if input_channels > 3 and n_frames not present, guess
                 pass
            
            self.output_steps = metadata.get('output_steps', 10)
            self.get_logger().info(f'Loaded metadata: {metadata}')
        else:
            self.output_steps = 10  # Default
            self.get_logger().warn(f'No metadata file found at {metadata_path}')
    
    def _init_onnx_session(self):
        """Initialize ONNX Runtime session with appropriate execution provider."""
        try:
            import onnxruntime as ort
        except ImportError:
            self.get_logger().error('onnxruntime not installed! Install with: pip install onnxruntime-gpu')
            raise
        
        # Select execution providers
        providers = []
        
        if self.use_tensorrt:
            # TensorRT provider for Jetson (best performance)
            trt_options = {
                'trt_max_workspace_size': 2147483648,  # 2GB
                'trt_fp16_enable': True,
            }
            providers.append(('TensorrtExecutionProvider', trt_options))
        
        # CUDA provider as fallback
        providers.append(('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
        }))
        
        # CPU provider as last fallback
        providers.append('CPUExecutionProvider')
        
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.get_logger().info(f'Creating ONNX session with providers: {[p if isinstance(p, str) else p[0] for p in providers]}')
        
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        active_providers = self.session.get_providers()
        self.get_logger().info(f'Active execution providers: {active_providers}')
    
    def preprocess_image(self, cv_image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for model input.
        
        Args:
            cv_image: OpenCV image (BGR, HxWxC)
            
        Returns:
            Preprocessed image (RGB/Gray, CxHxW, normalized to [0, 1])
        """
        # Resize
        if cv_image.shape[:2] != (self.input_height, self.input_width):
            cv_image = cv2.resize(cv_image, (self.input_width, self.input_height))
        
        # Color space conversion
        if self.channels_per_frame == 1:
            # BGR to Grayscale
            processed_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # Add channel dimension: (H, W) -> (H, W, 1)
            processed_image = np.expand_dims(processed_image, axis=2)
        elif self.channels_per_frame == 3:
            # BGR to RGB
            processed_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            # Fallback/Pass-through (e.g., if already correct or other format)
            processed_image = cv_image
            # Ensure it has 3 dims
            if len(processed_image.shape) == 2:
                processed_image = np.expand_dims(processed_image, axis=2)
                
        # Normalize to [0, 1] and convert to float32
        normalized = processed_image.astype(np.float32) / 255.0
        
        # HWC to CHW
        chw_image = np.transpose(normalized, (2, 0, 1))
        
        return chw_image
    
    def image_callback(self, msg: Image):
        """Handle incoming camera images."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Preprocess
            processed = self.preprocess_image(cv_image)
            
            # Add to history
            self.frame_history.append(processed)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def inference_callback(self):
        """Run model inference at fixed rate."""
        # Check if we have enough frames
        if len(self.frame_history) < self.num_history_frames:
            self.get_logger().debug(
                f'Waiting for frames: {len(self.frame_history)}/{self.num_history_frames}',
                throttle_duration_sec=1.0
            )
            return
        
        try:
            import time
            start_time = time.perf_counter()
            
            # Stack frames along channel dimension
            # Each frame is (3, H, W), stack to get (num_frames * 3, H, W)
            frames = list(self.frame_history)
            stacked = np.concatenate(frames, axis=0)
            
            # Add batch dimension: (1, C, H, W)
            input_tensor = np.expand_dims(stacked, axis=0)
            
            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )
            
            # outputs[0] shape: (1, output_steps, 2)
            # Each step has [steering, speed/throttle]
            self.latest_actions = outputs[0][0]  # Remove batch dimension
            
            # Calculate inference time
            inference_time = time.perf_counter() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            # Log periodically
            if self.inference_count % 100 == 0:
                avg_time = self.total_inference_time / self.inference_count
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(
                    f'Inference stats: avg_time={avg_time*1000:.2f}ms, fps={fps:.1f}'
                )
            
            # Publish the first action (immediate control)
            self.publish_drive_command()
            
        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
    
    def publish_drive_command(self):
        """Publish the predicted drive command."""
        if self.latest_actions is None:
            return
        
        # Use first predicted action (immediate timestep)
        steering = float(self.latest_actions[0, 0])
        speed = float(self.latest_actions[0, 1])
        
        # Apply scaling
        steering = steering * self.steering_scale
        speed = speed * self.speed_scale
        
        # Clamp values
        steering = np.clip(steering, -self.max_steering_angle, self.max_steering_angle)
        speed = np.clip(speed, 0.0, self.max_speed)
        
        # Create and publish message
        msg = AckermannDriveStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        msg.drive.steering_angle = steering
        msg.drive.speed = speed
        
        self.drive_pub.publish(msg)
    
    def destroy_node(self):
        """Cleanup on shutdown."""
        if self.inference_count > 0:
            avg_time = self.total_inference_time / self.inference_count
            self.get_logger().info(
                f'Final stats: {self.inference_count} inferences, '
                f'avg_time={avg_time*1000:.2f}ms'
            )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = InferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
        raise
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
