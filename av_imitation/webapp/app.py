import os
import json
import glob
import rclpy
from flask import Flask, render_template, jsonify, request, send_file
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, CompressedImage, Joy
from cv_bridge import CvBridge
import cv2
import numpy as np
import io
from ament_index_python.packages import get_package_share_directory

package_share_directory = get_package_share_directory('av_imitation')
template_dir = os.path.join(package_share_directory, 'webapp', 'templates')
static_dir = os.path.join(package_share_directory, 'webapp', 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

BAG_DIR = os.path.expanduser("~/roboracer_ws/data/rosbags")
PROCESSED_DIR = os.path.expanduser("~/roboracer_ws/data/rosbags_processed")
bridge = CvBridge()

def get_bag_info(bag_path):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    try:
        reader.open(storage_options, converter_options)
        metadata = reader.get_metadata()
        return {
            "duration": metadata.duration.nanoseconds / 1e9,
            "start_time": metadata.starting_time.nanoseconds / 1e9,
            "message_count": metadata.message_count,
            "topics": [t.topic_metadata.name for t in metadata.topics_with_message_count]
        }
    except Exception as e:
        print(f"Error reading bag {bag_path}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/bags')
def list_bags():
    bags = []
    if not os.path.exists(BAG_DIR):
        os.makedirs(BAG_DIR)
    
    for item in os.listdir(BAG_DIR):
        path = os.path.join(BAG_DIR, item)
        if os.path.isdir(path):
            if glob.glob(os.path.join(path, "*.db3")) or glob.glob(os.path.join(path, "*.mcap")):
                # Get metadata if available
                meta_path = os.path.join(PROCESSED_DIR, f"{item}.json")
                bag_data = {
                    "name": item,
                    "description": "",
                    "cuts": [],
                    "duration": 0
                }
                
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            user_meta = json.load(f)
                            bag_data.update(user_meta)
                    except:
                        pass
                
                # If duration or start_time is missing, try to read metadata.yaml
                if bag_data["duration"] == 0 or "start_time" not in bag_data:
                    try:
                        import yaml
                        meta_yaml_path = os.path.join(path, "metadata.yaml")
                        if os.path.exists(meta_yaml_path):
                            with open(meta_yaml_path, 'r') as f:
                                meta_yaml = yaml.safe_load(f)
                                # Structure: rosbag2_bagfile_information: duration: nanoseconds: ...
                                duration_ns = meta_yaml['rosbag2_bagfile_information']['duration']['nanoseconds']
                                start_time_ns = meta_yaml['rosbag2_bagfile_information']['starting_time']['nanoseconds_since_epoch']
                                
                                if bag_data["duration"] == 0:
                                    bag_data["duration"] = duration_ns / 1e9
                                if "start_time" not in bag_data:
                                    bag_data["start_time"] = start_time_ns / 1e9
                    except Exception as e:
                        print(f"Error reading metadata.yaml for {item}: {e}")

                # If cuts are missing, generate them and cache them
                if not bag_data["cuts"]:
                     # Only generate if we have a valid bag path
                     try:
                         cuts = generate_default_cuts(path)
                         if cuts:
                             bag_data["cuts"] = cuts
                             # Cache it immediately
                             if not os.path.exists(PROCESSED_DIR):
                                 os.makedirs(PROCESSED_DIR)
                             with open(meta_path, 'w') as f:
                                 json.dump(bag_data, f, indent=2)
                     except Exception as e:
                         print(f"Error generating default cuts for {item}: {e}")
                
                bags.append(bag_data)
                
    # Sort bags by start_time descending (most recent first)
    # If start_time is missing (0), it will be at the end (or beginning depending on logic, let's assume 0 is old)
    bags.sort(key=lambda x: x.get("start_time", 0), reverse=True)
    
    return jsonify(bags)

@app.route('/api/bag/<bag_name>', methods=['DELETE'])
def delete_bag(bag_name):
    bag_path = os.path.join(BAG_DIR, bag_name)
    meta_path = os.path.join(PROCESSED_DIR, f"{bag_name}.json")
    
    try:
        if os.path.exists(bag_path):
            import shutil
            shutil.rmtree(bag_path)
            
        if os.path.exists(meta_path):
            os.remove(meta_path)
            
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error deleting bag {bag_name}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/bag/<bag_name>/info')
def bag_info(bag_name):
    bag_path = os.path.join(BAG_DIR, bag_name)
    info = get_bag_info(bag_path)
    
    # Also check for existing metadata
    meta_path = os.path.join(PROCESSED_DIR, f"{bag_name}.json")
    user_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            user_meta = json.load(f)
    else:
        # Generate default cuts based on joystick L1 button
        # L1 is usually button 4. If NOT held (0), it's a cut.
        # We need to scan the bag for this.
        cuts = generate_default_cuts(bag_path)
        if cuts:
            user_meta["cuts"] = cuts
            
    # Check for telemetry data
    telemetry_path = os.path.join(PROCESSED_DIR, f"{bag_name}_telemetry.json")
    telemetry = []
    if os.path.exists(telemetry_path):
        try:
            with open(telemetry_path, 'r') as f:
                telemetry = json.load(f)
        except:
            pass
    else:
        # Extract telemetry if not found
        telemetry = extract_telemetry(bag_path)
        if telemetry:
             if not os.path.exists(PROCESSED_DIR):
                 os.makedirs(PROCESSED_DIR)
             with open(telemetry_path, 'w') as f:
                 json.dump(telemetry, f)

    return jsonify({"info": info, "user_meta": user_meta, "telemetry": telemetry})

from amrl_msgs.msg import AckermannCurvatureDriveMsg

def extract_telemetry(bag_path):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag for telemetry: {e}")
        return []

    metadata = reader.get_metadata()
    joy_topic = None
    ackermann_topic = None
    
    for t in metadata.topics_with_message_count:
        if 'joy' in t.topic_metadata.name:
            joy_topic = t.topic_metadata.name
        if 'ackermann_curvature_drive' in t.topic_metadata.name:
            ackermann_topic = t.topic_metadata.name
            
    if not joy_topic and not ackermann_topic:
        return []

    topics_to_filter = []
    if joy_topic: topics_to_filter.append(joy_topic)
    if ackermann_topic: topics_to_filter.append(ackermann_topic)

    storage_filter = StorageFilter(topics=topics_to_filter)
    reader.set_filter(storage_filter)

    telemetry = []
    joy_msg_type = get_message('sensor_msgs/msg/Joy')
    ack_msg_type = get_message('amrl_msgs/msg/AckermannCurvatureDriveMsg')
    
    # State
    current_state = {
        "steer": 0.0,
        "throttle": 0.0,
        "l1": False,
        "l2": 0.0,
        "velocity": 0.0,
        "curvature": 0.0
    }
    
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        t_sec = (t - metadata.starting_time.nanoseconds) / 1e9
        
        updated = False
        
        if topic == joy_topic:
            msg = deserialize_message(data, joy_msg_type)
            if len(msg.axes) > 0: current_state["steer"] = msg.axes[0]
            if len(msg.axes) > 4: current_state["throttle"] = -msg.axes[4]
            if len(msg.axes) > 2: current_state["l2"] = msg.axes[2]
            if len(msg.buttons) > 4: current_state["l1"] = bool(msg.buttons[4])
            updated = True
            
        elif topic == ackermann_topic:
            msg = deserialize_message(data, ack_msg_type)
            current_state["velocity"] = msg.velocity
            current_state["curvature"] = msg.curvature
            updated = True
            
        if updated:
            # Append a copy of current state
            entry = current_state.copy()
            entry["time"] = t_sec
            telemetry.append(entry)
            
    return telemetry

def generate_default_cuts(bag_path):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag for cuts: {e}")
        return []

    # Filter for joystick topic
    # We assume topic is /joystick or /bluetooth_teleop/joy
    # Let's check topics first
    metadata = reader.get_metadata()
    joy_topic = None
    for t in metadata.topics_with_message_count:
        if 'joy' in t.topic_metadata.name:
            joy_topic = t.topic_metadata.name
            break
    
    if not joy_topic:
        return []

    storage_filter = StorageFilter(topics=[joy_topic])
    reader.set_filter(storage_filter)

    cuts = []
    current_cut_start = None
    
    # We need to track time to close the last cut
    last_time = 0
    start_time = metadata.starting_time.nanoseconds / 1e9
    
    # If the bag starts and L1 is NOT held, we start a cut at 0 (relative)
    # But we don't know the state until the first message.
    # Let's assume it's NOT held at start.
    current_cut_start = 0.0
    
    msg_type = get_message('sensor_msgs/msg/Joy')
    
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        t_sec = (t - metadata.starting_time.nanoseconds) / 1e9
        last_time = t_sec
        
        if topic == joy_topic:
            msg = deserialize_message(data, msg_type)
            # Check button 4 (L1)
            # Ensure we have enough buttons
            if len(msg.buttons) > 4:
                l1_pressed = msg.buttons[4] == 1
                
                if l1_pressed:
                    # If L1 is pressed, we are "recording".
                    # If we were in a cut, close it.
                    if current_cut_start is not None:
                        cuts.append({"start": current_cut_start, "end": t_sec})
                        current_cut_start = None
                else:
                    # L1 not pressed. We should be in a cut.
                    if current_cut_start is None:
                        current_cut_start = t_sec
    
    # If we ended with a cut open, close it at the end
    if current_cut_start is not None:
        # If last_time is 0 (no messages?), use duration if possible, or just 0
        if last_time == 0:
             last_time = metadata.duration.nanoseconds / 1e9
        cuts.append({"start": current_cut_start, "end": last_time})
        
    return cuts

@app.route('/api/bag/<bag_name>/frame/<float:timestamp>')
def get_frame(bag_name, timestamp):
    # This is inefficient for random access, but simple for now.
    # A better approach would be to index the bag or use a persistent reader with seek if possible.
    # ROS 2 bag reader doesn't support seek by time easily without re-opening or iterating.
    # For a "preview", we might just iterate until we find the timestamp.
    # OPTIMIZATION: Cache the reader or index?
    # For this prototype, we will just open and seek (iterate).
    
    bag_path = os.path.join(BAG_DIR, bag_name)
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    
    metadata = reader.get_metadata()
    start_time_ns = metadata.starting_time.nanoseconds
    target_ns = start_time_ns + int(timestamp * 1e9)
    
    # Create a filter for image topics
    # storage_filter = StorageFilter(topics=['/camera/image_raw', '/camera/image_raw/compressed'])
    # reader.set_filter(storage_filter) 
    # set_filter is not directly exposed in simple python wrapper in older versions, check availability.
    # We'll just skip non-image messages manually.
    
    # We'll just skip non-image messages manually.

    
    # We can't seek directly. We have to read.
    # To make this faster for the web app, maybe we should extract a low-res video or thumbnails?
    # Or just accept it might be slow.
    # Alternatively, we can use the seek interface if available in newer rosbag2_py.
    # reader.seek(target_ns) # Check if this exists.
    
    try:
        reader.seek(target_ns)
    except Exception as e:
        print(f"Seek failed: {e}")
        pass

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        if t >= target_ns:
            # print(f"Found msg at {t} (target {target_ns}): {topic}")
            if 'image' in topic:
                msg_type = get_message('sensor_msgs/msg/Image')
                if 'compressed' in topic:
                     msg_type = get_message('sensor_msgs/msg/CompressedImage')
                
                try:
                    msg = deserialize_message(data, msg_type)
                    
                    cv_img = None
                    if isinstance(msg, Image):
                        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    elif isinstance(msg, CompressedImage):
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if cv_img is not None:
                        _, buffer = cv2.imencode('.jpg', cv_img)
                        return send_file(io.BytesIO(buffer), mimetype='image/jpeg')
                except Exception as e:
                    print(f"Error decoding image: {e}")
            
            # If we went too far past (e.g. 0.5s), stop
            if t > target_ns + 5e8:
                print(f"Timeout searching for frame. Current: {t}, Target: {target_ns}")
                break
                
    return "Frame not found", 404

import struct

@app.route('/api/bag/<bag_name>/stream_frames')
def stream_frames(bag_name):
    bag_path = os.path.join(BAG_DIR, bag_name)
    
    def generate():
        reader = SequentialReader()
        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        
        try:
            reader.open(storage_options, converter_options)
        except Exception as e:
            print(f"Error opening bag for stream: {e}")
            return

        metadata = reader.get_metadata()
        start_time_ns = metadata.starting_time.nanoseconds
        
        # Filter for image topics
        image_topics = []
        total_frames = 0
        for t in metadata.topics_with_message_count:
            if 'image' in t.topic_metadata.name:
                image_topics.append(t.topic_metadata.name)
                total_frames += t.message_count
        
        if not image_topics:
            return

        # Send total frames as header (uint32)
        yield struct.pack('<I', total_frames)

        storage_filter = StorageFilter(topics=image_topics)
        reader.set_filter(storage_filter)
        
        msg_type_image = get_message('sensor_msgs/msg/Image')
        msg_type_compressed = get_message('sensor_msgs/msg/CompressedImage')

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            t_sec = (t - start_time_ns) / 1e9
            
            try:
                jpeg_data = None
                
                if 'compressed' in topic:
                    msg = deserialize_message(data, msg_type_compressed)
                    # Already compressed? If it's jpeg, great. If png, maybe convert?
                    # Usually compressed is jpeg.
                    # Let's assume it's usable as is if format is jpeg
                    if 'jpeg' in msg.format or 'jpg' in msg.format:
                        jpeg_data = bytes(msg.data)
                    else:
                        # Decode and re-encode if not jpeg (e.g. png)
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if cv_img is not None:
                            _, buffer = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            jpeg_data = buffer.tobytes()
                else:
                    msg = deserialize_message(data, msg_type_image)
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    if cv_img is not None:
                        # Resize for performance? 320x240 is small enough.
                        # Encode to jpeg
                        _, buffer = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        jpeg_data = buffer.tobytes()
                
                if jpeg_data:
                    # Protocol: [timestamp(double)][size(uint32)][data]
                    header = struct.pack('<dI', t_sec, len(jpeg_data))
                    yield header + jpeg_data
                    
            except Exception as e:
                print(f"Error processing frame in stream: {e}")
                continue

    return app.response_class(generate(), mimetype='application/octet-stream')

@app.route('/api/bag/<bag_name>/metadata', methods=['POST'])
def save_metadata(bag_name):
    data = request.json
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    path = os.path.join(PROCESSED_DIR, f"{bag_name}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        
    return jsonify({"status": "success"})

def main():
    rclpy.init()
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
