import argparse
import os
import json
import shutil
from datetime import datetime
import numpy as np
import cv2
import re
import math
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image, CompressedImage, Joy
from amrl_msgs.msg import AckermannCurvatureDriveMsg
from av_imitation.src.vesc_utils import load_vesc_config, calculate_steering, calculate_curvature
def main():
    parser = argparse.ArgumentParser(description="Process ROS 2 bag files for imitation learning.")
    parser.add_argument("bag_name", help="Name of the bag file (directory name) in ~/roboracer_ws/data/rosbags")
    parser.add_argument("--sampling-rate", type=float, default=10.0, help="Image sampling rate in Hz")
    parser.add_argument("--seq-duration", type=float, default=3.0, help="Image sequence duration in seconds (history)")
    parser.add_argument("--action-freq", type=float, default=5.0, help="Action frequency in Hz")
    parser.add_argument("--pred-duration", type=float, default=1.5, help="Prediction duration in seconds (future)")
    parser.add_argument("--discrete-examples", action="store_true", help="If set, examples will not overlap (sliding window disabled)")
    
    args = parser.parse_args()
    
    bag_dir = os.path.expanduser(f"~/roboracer_ws/data/rosbags/{args.bag_name}")
    processed_base_dir = os.path.expanduser("~/roboracer_ws/data/rosbags_processed")
    metadata_path = os.path.join(processed_base_dir, f"{args.bag_name}.json")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(processed_base_dir, f"{args.bag_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load Configuration ---
    # --- Load Configuration ---
    config_data = load_vesc_config()
    print(f"Loaded config: {config_data.keys()}")
            
    # Save combined configuration
    final_config = vars(args)
    final_config['driver_config'] = config_data
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(final_config, f, indent=2)
        
    # Load metadata (cuts)
    cuts = []
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
            cuts = meta.get("cuts", [])
            
    print(f"Processing bag: {args.bag_name}")
    print(f"Output directory: {output_dir}")
    print(f"Cuts: {cuts}")

    # Open bag
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    
    bridge = CvBridge()
    
    # Topics
    IMAGE_TOPIC = '/camera/image_raw' # Adjusted based on user info: /camera_0/image_raw might be it, check both
    IMAGE_TOPIC_ALT = '/camera_0/image_raw'
    COMPRESSED_IMAGE_TOPIC = '/camera/image_raw/compressed'
    ACTION_TOPIC = '/ackermann_curvature_drive'
    JOY_TOPIC = '/joystick'
    
    # Buffers
    images = [] # (timestamp, image_data)
    actions = [] # (timestamp, v, curvature)
    
    print("Reading bag data...")
    
    msg_count = 0
    # We might need to reconstruct actions from joystick if ACTION_TOPIC is empty
    joystick_msgs = [] 
    
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        t_sec = t / 1e9
        msg_count += 1
        if msg_count % 1000 == 0:
            print(f"Read {msg_count} messages...", end='\r')
            
        # Check if in cut section
        in_cut = False
        for cut in cuts:
            if cut['start'] <= t_sec <= cut['end']:
                in_cut = True
                break
        if in_cut:
            continue
            
        if topic == IMAGE_TOPIC or topic == IMAGE_TOPIC_ALT:
            msg = deserialize_message(data, Image)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            images.append((t_sec, cv_img))
        elif topic == COMPRESSED_IMAGE_TOPIC:
            msg = deserialize_message(data, CompressedImage)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            images.append((t_sec, cv_img))
        elif topic == ACTION_TOPIC:
            # IGNORE action topic as it should be empty
            # In the future, if we want to learn from an existing policy, to say do dataset augmentation
            pass
        elif topic == JOY_TOPIC:
            msg = deserialize_message(data, Joy)
            joystick_msgs.append((t_sec, msg))

    print(f"\nLoaded {len(images)} images and {len(actions)} actions.")
    
    # Reconstruct actions from joystick using VESC driver logic
    if joystick_msgs:
        print("Reconstructing actions from Joystick using VESC driver logic...")
        
        joystick_mode = config_data.get('joystick_mode', 'both')
        normal_speed = float(config_data.get('joystick_normal_speed', 1.0))
        turbo_speed = float(config_data.get('joystick_turbo_speed', 2.0))
        
        actions = [] # Clear any existing actions just in case
        
        for t, msg in joystick_msgs:
            # Check for turbo (Axis 2 >= 0.9)
            turbo = False
            if len(msg.axes) > 2 and msg.axes[2] >= 0.9:
                turbo = True
                
            current_max_speed = turbo_speed if turbo else normal_speed
            
            # Extract inputs based on mode
            steer_input = 0.0
            drive_input = 0.0
            
            # Default to 'both' checks
            if joystick_mode == 'left':
                if len(msg.axes) > 1:
                    steer_input = -msg.axes[0]
                    drive_input = -msg.axes[1]
            elif joystick_mode == 'right':
                if len(msg.axes) > 4:
                    steer_input = -msg.axes[3]
                    drive_input = -msg.axes[4]
            else: # 'both' (default)
                if len(msg.axes) > 4:
                    steer_input = -msg.axes[0]  # Left stick horizontal for steering
                    drive_input = -msg.axes[4]  # Right stick vertical for drive
            
            # Calculate final commands
            v = drive_input * current_max_speed
            
            # Bezier steering
            steering_angle = calculate_steering(steer_input, config_data)
            c = calculate_curvature(steering_angle, config_data)
            
            actions.append((t, v, c))
            
        print(f"Reconstructed {len(actions)} actions from joystick.")
    
    if not images or not actions:
        print("No data found!")
        return

    # Sort just in case
    images.sort(key=lambda x: x[0])
    actions.sort(key=lambda x: x[0])
    
    # Convert actions to numpy for interpolation
    action_times = np.array([x[0] for x in actions])
    action_vs = np.array([x[1] for x in actions])
    action_cs = np.array([x[2] for x in actions])
    
    # Generate examples
    num_history_frames = int(args.seq_duration * args.sampling_rate)
    num_pred_frames = int(args.pred_duration * args.action_freq)
    pred_dt = 1.0 / args.action_freq
    stride = num_history_frames if args.discrete_examples else 1
    
    example_idx = 0
    
    for i in range(num_history_frames - 1, len(images), stride):
        t0 = images[i][0]
        
        if t0 + args.pred_duration > action_times[-1]:
            break
            
        t_start_history = images[i - num_history_frames + 1][0]
        if (t0 - t_start_history) > (args.seq_duration + 0.5):
            continue
            
        example_dir = os.path.join(output_dir, f"sample_{example_idx:05d}")
        os.makedirs(example_dir, exist_ok=True)
        img_dir = os.path.join(example_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        history_imgs = images[i - num_history_frames + 1 : i + 1]
        
        for j, (t_img, img) in enumerate(history_imgs):
            cv2.imwrite(os.path.join(img_dir, f"{j:03d}.jpg"), img)
            
        future_actions = []
        for k in range(1, num_pred_frames + 1):
            t_target = t0 + k * pred_dt
            
            idx = np.searchsorted(action_times, t_target)
            if idx == 0:
                v = action_vs[0]
                c = action_cs[0]
            elif idx >= len(action_times):
                v = action_vs[-1]
                c = action_cs[-1]
            else:
                t1, t2 = action_times[idx-1], action_times[idx]
                alpha = (t_target - t1) / (t2 - t1)
                v = (1-alpha)*action_vs[idx-1] + alpha*action_vs[idx]
                c = (1-alpha)*action_cs[idx-1] + alpha*action_cs[idx]
            
            future_actions.append({"time": t_target, "velocity": float(v), "curvature": float(c)})
            
        with open(os.path.join(example_dir, "targets.json"), "w") as f:
            json.dump(future_actions, f, indent=2)
            
        example_idx += 1
        if example_idx % 10 == 0:
            print(f"Generated {example_idx} examples...", end='\r')
            
    print(f"\nFinished! Generated {example_idx} examples in {output_dir}")

if __name__ == "__main__":
    main()
