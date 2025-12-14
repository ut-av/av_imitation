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

import threading
import uuid
import time

package_share_directory = get_package_share_directory('av_imitation')
template_dir = os.path.join(package_share_directory, 'webapp', 'templates')
static_dir = os.path.join(package_share_directory, 'webapp', 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

BAG_DIR = os.path.expanduser("~/roboracer_ws/data/rosbags")
PROCESSED_DIR = os.path.expanduser("~/roboracer_ws/data/rosbags_processed")
DATASETS_DIR = os.path.join(PROCESSED_DIR, "datasets")
if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)
bridge = CvBridge()

from av_imitation.src.image_processor import ImageProcessor
processor = ImageProcessor()

def get_bag_info(bag_path):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    try:
        reader.open(storage_options, converter_options)
        metadata = reader.get_metadata()
        image_count = 0
        for t in metadata.topics_with_message_count:
            if 'image' in t.topic_metadata.name:
                image_count += t.message_count

        return {
            "duration": metadata.duration.nanoseconds / 1e9,
            "start_time": metadata.starting_time.nanoseconds / 1e9,
            "message_count": metadata.message_count,
            "image_count": image_count,
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
                                    
                                # Extract image count
                                image_count = 0
                                topics = meta_yaml['rosbag2_bagfile_information']['topics_with_message_count']
                                for t in topics:
                                    if 'image' in t['topic_metadata']['name']:
                                        image_count += t['message_count']
                                bag_data["image_count"] = image_count
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
    bag_path = os.path.join(BAG_DIR, bag_name)
    cv_img = get_cv_image_from_bag(bag_path, timestamp)
    
    if cv_img is not None:
        _, buffer = cv2.imencode('.jpg', cv_img)
        return send_file(io.BytesIO(buffer), mimetype='image/jpeg')
    
    return "Frame not found", 404

def get_cv_image_from_bag(bag_path, timestamp):
    # This is inefficient for random access, but simple for now.
    # A better approach would be to index the bag or use a persistent reader with seek if possible.
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        return None
    
    metadata = reader.get_metadata()
    start_time_ns = metadata.starting_time.nanoseconds
    target_ns = start_time_ns + int(timestamp * 1e9)
    
    try:
        reader.seek(target_ns)
    except Exception as e:
        print(f"Seek failed: {e}")
        pass

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        if t >= target_ns:
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
                    
                    return cv_img
                except Exception as e:
                    print(f"Error decoding image: {e}")
            
            # If we went too far past (e.g. 0.5s), stop
            if t > target_ns + 5e8:
                break
                
    return None

@app.route('/api/preview_processing', methods=['POST'])
def preview_processing():
    data = request.json
    bag_name = data.get('bag_name')
    timestamp = data.get('timestamp')
    options = data.get('options', {})
    
    if not bag_name or timestamp is None:
        return jsonify({"error": "Bag name and timestamp required"}), 400
        
    bag_path = os.path.join(BAG_DIR, bag_name)
    cv_img = get_cv_image_from_bag(bag_path, timestamp)
    
    if cv_img is None:
        return jsonify({"error": "Frame not found"}), 404
        
    try:
        processed_img = processor.process(cv_img, options)
        _, buffer = cv2.imencode('.jpg', processed_img)
        return send_file(io.BytesIO(buffer), mimetype='image/jpeg')
    except Exception as e:
        print(f"Error processing preview: {e}")
        return jsonify({"error": str(e)}), 500

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

TAGS_FILE = os.path.join(PROCESSED_DIR, "tags.json")

@app.route('/api/tags', methods=['GET'])
def get_tags():
    if not os.path.exists(TAGS_FILE):
        return jsonify([])
    try:
        with open(TAGS_FILE, 'r') as f:
            tags = json.load(f)
        return jsonify(tags)
    except Exception as e:
        print(f"Error reading tags: {e}")
        return jsonify([])

@app.route('/api/tags', methods=['POST'])
def add_tags():
    data = request.json
    new_tags = data.get('tags', [])
    if isinstance(new_tags, str):
        new_tags = [new_tags]
        
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    current_tags = []
    if os.path.exists(TAGS_FILE):
        try:
            with open(TAGS_FILE, 'r') as f:
                current_tags = json.load(f)
        except:
            pass
            
    # Add unique
    updated = False
    for tag in new_tags:
        if tag not in current_tags:
            current_tags.append(tag)
            updated = True
            
    if updated:
        current_tags.sort()
        try:
            with open(TAGS_FILE, 'w') as f:
                json.dump(current_tags, f, indent=2)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return jsonify(current_tags)

# Global state for processing jobs
# Key: bag_name, Value: { "status": "running"|"done"|"error"|"cancelled", "progress": 0, "total": 0, "current": 0, "cancel_flag": False }
processing_jobs = {}

@app.route('/api/process_bag', methods=['POST'])
def process_bag():
    data = request.json
    bag_name = data.get('bag_name')
    options = data.get('options', {})
    
    if not bag_name:
        return jsonify({"error": "Bag name required"}), 400
        
    bag_path = os.path.join(BAG_DIR, bag_name)
    if not os.path.exists(bag_path):
        return jsonify({"error": "Bag not found"}), 404
        
    # Determine output folder
    folder_name = processor.get_output_folder_name(options)
    output_dir = os.path.join(PROCESSED_DIR, folder_name, bag_name)
    images_dir = os.path.join(output_dir, "images")
    
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    # Save options
    with open(os.path.join(output_dir, "options.json"), 'w') as f:
        json.dump(options, f, indent=2)
        
    # Initialize job state
    processing_jobs[bag_name] = {
        "status": "running",
        "progress": 0,
        "total": 0,
        "current": 0,
        "cancel_flag": False
    }
        
    # Run processing
    import threading
    thread = threading.Thread(target=run_processing, args=(bag_name, bag_path, images_dir, options))
    thread.start()
    
    return jsonify({"status": "started", "output_dir": output_dir})

@app.route('/api/check_processing', methods=['POST'])
def check_processing():
    data = request.json
    bag_name = data.get('bag_name')
    options = data.get('options', {})
    
    if not bag_name:
        return jsonify({"error": "Bag name required"}), 400
        
    folder_name = processor.get_output_folder_name(options)
    output_dir = os.path.join(PROCESSED_DIR, folder_name, bag_name)
    images_dir = os.path.join(output_dir, "images")
    
    if os.path.exists(images_dir) and os.path.isdir(images_dir):
        # Get modification time of the directory
        mtime = os.path.getmtime(images_dir)
        return jsonify({
            "exists": True,
            "timestamp": mtime,
            "path": output_dir
        })
    else:
        return jsonify({"exists": False})

@app.route('/api/processing_status/<bag_name>')
def get_processing_status(bag_name):
    job = processing_jobs.get(bag_name)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)

@app.route('/api/cancel_processing/<bag_name>', methods=['POST'])
def cancel_processing(bag_name):
    job = processing_jobs.get(bag_name)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    job['cancel_flag'] = True
    return jsonify({"status": "cancellation_requested"})

def run_processing(bag_name, bag_path, output_dir, options):
    print(f"Starting processing for {bag_path} with options {options}")
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        processing_jobs[bag_name]['status'] = "error"
        processing_jobs[bag_name]['error'] = str(e)
        return

    metadata = reader.get_metadata()
    
    # Filter for image topics
    image_topics = []
    image_count = 0
    for t in metadata.topics_with_message_count:
        if 'image' in t.topic_metadata.name:
            image_topics.append(t.topic_metadata.name)
            image_count += t.message_count
            
    if not image_topics:
        print("No image topics found")
        processing_jobs[bag_name]['status'] = "error"
        processing_jobs[bag_name]['error'] = "No image topics found"
        return
        
    processing_jobs[bag_name]['total'] = image_count
        
    storage_filter = StorageFilter(topics=image_topics)
    reader.set_filter(storage_filter)
    
    msg_type_image = get_message('sensor_msgs/msg/Image')
    msg_type_compressed = get_message('sensor_msgs/msg/CompressedImage')
    
    count = 0
    while reader.has_next():
        # Check cancellation
        if processing_jobs[bag_name]['cancel_flag']:
            print(f"Processing cancelled for {bag_name}")
            processing_jobs[bag_name]['status'] = "cancelled"
            return

        (topic, data, t) = reader.read_next()
        t_sec = (t - metadata.starting_time.nanoseconds) / 1e9
        
        try:
            cv_img = None
            if 'compressed' in topic:
                msg = deserialize_message(data, msg_type_compressed)
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                msg = deserialize_message(data, msg_type_image)
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
            if cv_img is not None:
                # Process
                processed_img = processor.process(cv_img, options)
                
                # Save
                filename = f"{t_sec:.6f}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), processed_img)
                count += 1
                
                # Update progress
                processing_jobs[bag_name]['current'] = count
                if image_count > 0:
                    processing_jobs[bag_name]['progress'] = (count / image_count) * 100
                
                if count % 100 == 0:
                    print(f"Processed {count} frames")
                    
        except Exception as e:
            print(f"Error processing frame: {e}")
            
    print(f"Finished processing {count} frames")
    processing_jobs[bag_name]['status'] = "done"
    processing_jobs[bag_name]['progress'] = 100

@app.route('/api/processed_bags')
def list_processed_bags():
    # Structure: PROCESSED_DIR / <options_name> / <bag_name> / images
    # We want to list unique processed bags available.
    # Actually, the user selects "processed bags".
    # A processed bag is defined by (Original Bag Name, Processing Options).
    # We can list them grouped by options or flat.
    # Let's return a list of objects: { bag_name, options_name, options, path }
    
    results = []
    if not os.path.exists(PROCESSED_DIR):
        return jsonify([])
        
    for options_name in os.listdir(PROCESSED_DIR):
        opt_path = os.path.join(PROCESSED_DIR, options_name)
        if not os.path.isdir(opt_path): continue
        
        for bag_name in os.listdir(opt_path):
            bag_path = os.path.join(opt_path, bag_name)
            if not os.path.isdir(bag_path): continue
            
            # Check for options.json
            options = {}
            try:
                with open(os.path.join(bag_path, "options.json"), 'r') as f:
                    options = json.load(f)
            except:
                pass
            
            # Find a thumbnail
            thumbnail_path = None
            images_dir = os.path.join(bag_path, "images")
            if os.path.exists(images_dir):
                # Get first jpg
                images = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
                if images:
                    # Construct relative path for serving
                    # Route: /api/processed_file/<path>
                    # Filepath needs to be relative to PROCESSED_DIR
                    full_img_path = os.path.join(images_dir, images[0])
                    thumbnail_path = os.path.relpath(full_img_path, PROCESSED_DIR)

            results.append({
                "bag_name": bag_name,
                "options_name": options_name,
                "options": options,
                "path": bag_path,
                "thumbnail_path": thumbnail_path
            })
            
    return jsonify(results)

generation_jobs = {}

def generate_dataset_thread(job_id, selected_bags, dataset_name, history_rate, history_duration, future_rate, future_duration):
    try:
        samples = []
        total_bags = len(selected_bags)
        
        for i, bag_info in enumerate(selected_bags):
            generation_jobs[job_id]['current'] = i
            generation_jobs[job_id]['total'] = total_bags
            generation_jobs[job_id]['progress'] = (i / total_bags) * 100
            
            bag_path = bag_info['path']
            bag_name = bag_info['bag_name']
            images_dir = os.path.join(bag_path, "images")
            
            # Load telemetry
            telemetry_path = os.path.join(PROCESSED_DIR, f"{bag_name}_telemetry.json")
            
            if not os.path.exists(telemetry_path):
                print(f"Telemetry not found for {bag_name}")
                continue
                
            with open(telemetry_path, 'r') as f:
                telemetry = json.load(f)
                
            # Sort telemetry by time
            telemetry.sort(key=lambda x: x['time'])
            telemetry_times = np.array([x['time'] for x in telemetry])
            
            # Get available images
            image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
            
            # Parse timestamps from filenames
            image_times = []
            for f in image_files:
                try:
                    t = float(os.path.splitext(f)[0])
                    image_times.append(t)
                except:
                    pass
            
            image_times = np.array(image_times)
            
            # Generate samples
            hist_interval = 1.0 / history_rate
            fut_interval = 1.0 / future_rate
            
            num_hist = int(history_duration * history_rate)
            num_fut = int(future_duration * future_rate)
            
            for curr_t in image_times:
                # Current image
                curr_img_name = f"{curr_t:.6f}.jpg"
                
                # Find history images
                history_images = []
                valid_sample = True
                
                for k in range(num_hist, 0, -1):
                    target_t = curr_t - k * hist_interval
                    idx = (np.abs(image_times - target_t)).argmin()
                    closest_t = image_times[idx]
                    
                    if abs(closest_t - target_t) > 0.1:
                        valid_sample = False
                        break
                        
                    history_images.append(os.path.join(bag_path, "images", f"{closest_t:.6f}.jpg"))
                    
                if not valid_sample:
                    continue
                    
                # Find future actions and images
                future_actions = []
                future_images = []
                
                for k in range(1, num_fut + 1):
                    target_t = curr_t + k * fut_interval
                    
                    idx = (np.abs(telemetry_times - target_t)).argmin()
                    closest_t = telemetry_times[idx]
                    
                    if abs(closest_t - target_t) > 0.1:
                        valid_sample = False
                        break
                        
                    entry = telemetry[idx]
                    future_actions.append([entry['steer'], entry['throttle']])
                    
                    img_idx = (np.abs(image_times - target_t)).argmin()
                    closest_img_t = image_times[img_idx]
                    if abs(closest_img_t - target_t) > 0.1:
                        valid_sample = False
                        break
                    
                    future_images.append(os.path.join(bag_path, "images", f"{closest_img_t:.6f}.jpg"))
                    
                if not valid_sample:
                    continue
                    
                # Add sample
                rel_curr = os.path.relpath(os.path.join(images_dir, curr_img_name), PROCESSED_DIR)
                rel_hist = [os.path.relpath(p, PROCESSED_DIR) for p in history_images]
                rel_fut = [os.path.relpath(p, PROCESSED_DIR) for p in future_images]
                
                samples.append({
                    "bag": bag_name,
                    "timestamp": curr_t,
                    "current_image": rel_curr,
                    "history_images": rel_hist,
                    "future_actions": future_actions,
                    "future_images": rel_fut
                })
                
        # Save metadata
        output_file = os.path.join(DATASETS_DIR, f"{dataset_name}.json")
        meta = {
            "root_dir": PROCESSED_DIR,
            "dataset_name": dataset_name,
            "source_bags": [b['bag_name'] for b in selected_bags], # Store source bags for duplicate checking
            "parameters": {
                "history_rate": history_rate,
                "history_duration": history_duration,
                "future_rate": future_rate,
                "future_duration": future_duration
            },
            "samples": samples
        }
        
        with open(output_file, 'w') as f:
            json.dump(meta, f, indent=2)
            
        generation_jobs[job_id]['status'] = 'done'
        generation_jobs[job_id]['progress'] = 100
        generation_jobs[job_id]['file'] = output_file
        generation_jobs[job_id]['count'] = len(samples)
        
    except Exception as e:
        print(f"Generation error: {e}")
        generation_jobs[job_id]['status'] = 'error'
        generation_jobs[job_id]['error'] = str(e)

@app.route('/api/generate_dataset', methods=['POST'])
def generate_dataset():
    data = request.json
    selected_bags = data.get('selected_bags', [])
    dataset_name = data.get('dataset_name', 'dataset')
    
    history_rate = float(data.get('history_rate', 1.0))
    history_duration = float(data.get('history_duration', 1.0))
    future_rate = float(data.get('future_rate', 1.0))
    future_duration = float(data.get('future_duration', 1.0))
    
    if not selected_bags:
        return jsonify({"error": "No bags selected"}), 400
        
    job_id = str(uuid.uuid4())
    generation_jobs[job_id] = {
        'status': 'running',
        'progress': 0,
        'current': 0,
        'total': len(selected_bags)
    }
    
    thread = threading.Thread(target=generate_dataset_thread, args=(
        job_id, selected_bags, dataset_name, history_rate, history_duration, future_rate, future_duration
    ))
    thread.start()
    
    return jsonify({"job_id": job_id})

@app.route('/api/generation_status/<job_id>')
def generation_status(job_id):
    if job_id not in generation_jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(generation_jobs[job_id])

@app.route('/api/datasets')
def list_datasets():
    if not os.path.exists(DATASETS_DIR):
        return jsonify([])
        
    datasets = []
    for f in os.listdir(DATASETS_DIR):
        if f.endswith('.json'):
            try:
                with open(os.path.join(DATASETS_DIR, f), 'r') as file:
                    data = json.load(file)
                    if 'dataset_name' in data:
                        datasets.append({
                            "dataset_name": data['dataset_name'],
                            "source_bags": data.get('source_bags', []),
                            "samples_count": len(data.get('samples', [])),
                            "parameters": data.get('parameters', {})
                        })
            except:
                pass
                
    return jsonify(datasets)

@app.route('/api/dataset/<name>', methods=['GET', 'DELETE'])
def handle_dataset(name):
    path = os.path.join(DATASETS_DIR, f"{name}.json")
    
    if request.method == 'DELETE':
        if not os.path.exists(path):
            return jsonify({"error": "Dataset not found"}), 404
        try:
            os.remove(path)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if not os.path.exists(path):
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    return jsonify(data)

@app.route('/api/processed_file/<path:filepath>')
def serve_processed_file(filepath):
    # Serve file from PROCESSED_DIR
    # Security check: ensure it's within PROCESSED_DIR
    full_path = os.path.join(PROCESSED_DIR, filepath)
    if not os.path.abspath(full_path).startswith(os.path.abspath(PROCESSED_DIR)):
         return "Access denied", 403
         
    if not os.path.exists(full_path):
        return "File not found", 404
        
    return send_file(full_path)

def main():
    rclpy.init()
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
