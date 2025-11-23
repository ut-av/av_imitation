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
    # Look for both .mcap and .db3 (sqlite3)
    # For sqlite3, the bag is a directory or a file ending in .db3
    # We'll assume standard ros2 bag folder structure: bag_name/bag_name_0.db3
    # So we list directories in BAG_DIR
    if not os.path.exists(BAG_DIR):
        os.makedirs(BAG_DIR)
    
    for item in os.listdir(BAG_DIR):
        path = os.path.join(BAG_DIR, item)
        if os.path.isdir(path):
            # Check if it contains a db3 or mcap
            if glob.glob(os.path.join(path, "*.db3")) or glob.glob(os.path.join(path, "*.mcap")):
                bags.append(item)
    return jsonify(bags)

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
            
    return jsonify({"info": info, "user_meta": user_meta})

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

    # storage_filter = StorageFilter(topics=[joy_topic])
    # reader.set_filter(storage_filter) # If supported

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
