import os
import re
import math

def parse_lua_config(content):
    """
    Simple regex-based parser for Lua config files used in this project.
    Extracts variable assignments: name = value;
    Values can be floats, booleans (true/false), or strings.
    """
    config = {}
    
    # Regex for assignments: variable = value;
    pattern = re.compile(r'(\w+)\s*=\s*([^;\n]+);?')
    
    for match in pattern.finditer(content):
        key = match.group(1)
        value_str = match.group(2).strip()
        
        # Remove trailing comments if any (start with --)
        value_str = re.split(r'\s*--', value_str)[0].strip()
        
        # Determine type
        if value_str.lower() == 'true':
            value = True
        elif value_str.lower() == 'false':
            value = False
        elif value_str.startswith('"') and value_str.endswith('"'):
            value = value_str[1:-1]
        else:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
                
        config[key] = value
    
    return config

def load_vesc_config():
    """
    Loads VESC and Joystick configurations from the standard location.
    Returns a dictionary merged from both config files.
    """
    config_data = {}
    # Path to config files (assuming standard location)
    config_dir = os.path.expanduser("~/roboracer_ws/src/ut_automata/config")
    
    vesc_lua_path = os.path.join(config_dir, "vesc.lua")
    joystick_lua_path = os.path.join(config_dir, "joystick.lua")
    
    if os.path.exists(vesc_lua_path):
        with open(vesc_lua_path, 'r') as f:
            config_data.update(parse_lua_config(f.read()))
            
    if os.path.exists(joystick_lua_path):
        with open(joystick_lua_path, 'r') as f:
            config_data.update(parse_lua_config(f.read()))
            
    return config_data

def bezier4(t, p0, p1, p2, p3, p4):
    one_minus_t = 1.0 - t
    one_minus_t2 = one_minus_t * one_minus_t
    one_minus_t3 = one_minus_t2 * one_minus_t
    one_minus_t4 = one_minus_t3 * one_minus_t
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    
    return (one_minus_t4 * p0 +
            4.0 * one_minus_t3 * t * p1 +
            6.0 * one_minus_t2 * t2 * p2 +
            4.0 * one_minus_t * t3 * p3 +
            t4 * p4)

def bezier4_prime(t, p0, p1, p2, p3, p4):
    one_minus_t = 1.0 - t
    one_minus_t2 = one_minus_t * one_minus_t
    one_minus_t3 = one_minus_t2 * one_minus_t
    t2 = t * t
    t3 = t2 * t
    
    return (4.0 * one_minus_t3 * (p1 - p0) +
            12.0 * one_minus_t2 * t * (p2 - p1) +
            12.0 * one_minus_t * t2 * (p3 - p2) +
            4.0 * t3 * (p4 - p3))

def calculate_steering(joystick_val, config):
    try:
        xm = float(config.get('steering_curve_xm', 0.8))
        ym = float(config.get('steering_curve_ym', 0.1))
        max_angle = float(config.get('max_steering_angle', 0.425))
    except (ValueError, TypeError):
        # Fallback defaults
        xm, ym, max_angle = 0.8, 0.1, 0.425
        
    x_target = joystick_val
    
    # Initial guess for t (linear mapping from [-1, 1] to [0, 1])
    t = (x_target + 1.0) * 0.5
    
    # Newton's method
    for _ in range(10):
        x_val = bezier4(t, -1.0, -xm, 0.0, xm, 1.0)
        error = x_val - x_target
        if abs(error) < 1e-4:
            break
        dx_dt = bezier4_prime(t, -1.0, -xm, 0.0, xm, 1.0)
        if abs(dx_dt) < 1e-6:
            break
        t -= error / dx_dt
        t = max(0.0, min(1.0, t))
        
    steer_curved = bezier4(t, -1.0, -ym, 0.0, ym, 1.0)
    return steer_curved * max_angle

def calculate_curvature(steering_angle, config):
    wheelbase = float(config.get('wheelbase', 0.324))
    if steering_angle == 0 or wheelbase == 0:
        return 0.0
    return math.tan(steering_angle) / wheelbase
