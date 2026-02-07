"""
Dead Reckoning Maneuvers - Full-Featured Control Script
========================================================
A comprehensive script for optical flow-based position hold with 
advanced maneuver capabilities. Uses dead reckoning with PID controllers 
to maintain the drone's position and execute directional maneuvers.

This script provides a full-featured GUI for controlling the drone with
real-time visualization of sensor data, position tracking, and various
control modes including manual joystick control and autonomous shapes.

Features:
- Sensor Test to arm/prepare the drone
- Optical flow-based position hold
- Adjustable TRIM values, Height, and PID parameters
- Real-time position, velocity, and correction feedback
- Safety checks for battery and sensors
- NeoPixel LED control for visual feedback
- Directional maneuvers (Forward, Backward, Left, Right)
- Autonomous shape execution
- Joystick control with multiple modes
- CSV data logging for flight analysis
- Real-time matplotlib plotting

Usage:
1. Connect to the drone via UDP
2. Click "Sensor Test" to arm and verify sensors
3. Adjust parameters if needed (TRIM, Height, PID)
4. Click "Start Position Hold" to takeoff and hover
5. Use directional buttons or joystick for maneuvers
6. The drone will maintain its position automatically
7. Click "Stop" or press Enter for emergency stop

Author: Dharageswaran S
Version: 1.0
"""

import time
import threading
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.pylab import f
import numpy as np
import csv
import math
from datetime import datetime
import os
from PIL import Image, ImageTk
from collections import deque
import threading


# === CONSTANTS ===
# CRTP and NeoPixel constants
CRTP_PORT_NEOPIXEL = 0x09
NEOPIXEL_CHANNEL_SET_PIXEL = 0x00
NEOPIXEL_CHANNEL_SHOW = 0x01
NEOPIXEL_CHANNEL_CLEAR = 0x02
NEOPIXEL_CHANNEL_BLINK = 0x03
NP_SEND_RETRIES = 3
NP_PACKET_DELAY = 0.02
NP_LINK_SETUP_DELAY = 0.12


# === CONFIGURATION PARAMETERS ===
DRONE_URI = "udp://192.168.43.42"
TARGET_HEIGHT = 0.3  # Target hover height in meters
TAKEOFF_TIME = 1.0  # Time to takeoff and stabilize (reduced for steeper ramp)
HOVER_DURATION = 20.0  # How long to hover with position hold
LANDING_TIME = 0.5  # Time to land
# Debug mode - set to True to disable motors (sensors and logging still work)
DEBUG_MODE = False
# Height sensor safety check - set to False to disable emergency stop during takeoff/stabilized
ENABLE_HEIGHT_SENSOR_SAFETY = False 
# Filtering strength for velocity smoothing (0.0 = no smoothing, 1.0 = max smoothing)
VELOCITY_SMOOTHING_ALPHA = 0.85  # Default: 0.7 (previously hardcoded)
# CSV Logging - set to False to disable CSV file generation
DRONE_CSV_LOGGING = False
# Takeoff Ramp - set to False to disable smooth altitude climb (direct target height)
ENABLE_TAKEOFF_RAMP = False

# Basic trim corrections
TRIM_VX = 0.0  # Forward/backward trim correction
TRIM_VY = 0.0  # Left/right trim correction
# Battery monitoring
LOW_BATTERY_THRESHOLD = 2.9  # Low battery warning threshold in volts
# Height sensor safety
HEIGHT_SENSOR_MIN_CHANGE = (
    0.005  # Minimum height change expected during takeoff (meters)
)

# === DEAD RECKONING POSITION CONTROL PARAMETERS ===
# PID Controller Parameters
# Start here, then increase gradually
POSITION_KP = 1.0
POSITION_KI = 0.03  # Increased from 0.01 for better drift correction
POSITION_KD = 0.0
VELOCITY_KP = 0.7   # Increased from 0.5 for better damping
VELOCITY_KI = 0.01
VELOCITY_KD = 0.0
# Control limits
MAX_CORRECTION = 0.7  # Maximum control correction allowed
VELOCITY_THRESHOLD = 0.005  # Consider drone "stationary" below this velocity
DRIFT_COMPENSATION_RATE = 0.004  # Gentle pull toward zero when moving slowly
# Position integration and reset
PERIODIC_RESET_INTERVAL = 90.0  # Reset integrated position every 5 seconds
MAX_POSITION_ERROR = 2.0  # Clamp position error to prevent runaway
# Sensor parameters
SENSOR_PERIOD_MS = 10  # Motion sensor update rate
DT = SENSOR_PERIOD_MS / 1000.0
CONTROL_UPDATE_RATE = 0.02  # 50Hz control loop
# Velocity calculation constants
DEG_TO_RAD = 3.1415926535 / 180.0
# Optical flow scaling - adjust these to match your sensor/setup
OPTICAL_FLOW_SCALE = (
    4.4  # Empirical scaling factor (adjust based on real vs measured distance)
)
USE_HEIGHT_SCALING = True  # Set to False to disable height dependency

# === MANEUVER PARAMETERS ===
MANEUVER_DISTANCE = 0.5  # Default distance
MANEUVER_THRESHOLD = 0.10  # Within 5cm is close enough
WAYPOINT_TIMEOUT = 60.0  # Seconds before aborting mission
WAYPOINT_STABILIZATION_TIME = 0.5  # Seconds to hover at each corner
JOYSTICK_SENSITIVITY = 0.2

# === JOYSTICK MOMENTUM COMPENSATION ===
# When keys are released in "Hold at Current Position" mode, these parameters help prevent counter-movement
MOMENTUM_COMPENSATION_TIME = (
    0.10  # Seconds of velocity to predict stopping position (0.05-0.15 recommended)
)
SETTLING_DURATION = (
    0.1  # Time to use gentler corrections after key release (0.1-0.3 seconds)
)
SETTLING_CORRECTION_FACTOR = (
    0.5  # Correction strength during settling period (0.3-0.7, lower = gentler)
)

# === FIRMWARE PARAMETERS (Z-AXIS) ===
# Set to True to send these values to the drone on connection
ENABLE_FIRMWARE_PARAMS = False
FW_THRUST_BASE = 24000  # Default: 24000. Increase if drone feels heavy (e.g., 26000)
FW_Z_POS_KP = 1.6        # Default: 1.6. Height position gain
FW_Z_VEL_KP = 15.0       # Default: 22.0. Vertical velocity damping (stop bouncing)

# === OUTPUT WINDOW LOG ===
Output_Window = True  # Set to True to enable output log window, False to disable

# === GLOBAL VARIABLES ===
# Sensor data
current_height = 0.0
current_range_height = 0.0
motion_delta_x = 0
motion_delta_y = 0
sensor_data_ready = False
last_sensor_heartbeat = time.time()  # Track last received packet
DATA_TIMEOUT_THRESHOLD = 0.2        # Max allowed time between sensor packets (seconds)
# Log file
log_file = None
log_writer = None
# Battery voltage data
current_battery_voltage = 0.0
battery_data_ready = False
# Velocity tracking
current_vx = 0.0
current_vy = 0.0
velocity_x_history = [0.0, 0.0]
velocity_y_history = [0.0, 0.0]
# Dead reckoning position integration
integrated_position_x = 0.0
integrated_position_y = 0.0
last_integration_time = time.time()
last_reset_time = time.time()
# Control corrections
current_correction_vx = 0.0
current_correction_vy = 0.0
# PID Controller state variables
position_integral_x = 0.0
position_integral_y = 0.0
position_derivative_x = 0.0
position_derivative_y = 0.0
last_position_error_x = 0.0
last_position_error_y = 0.0
velocity_integral_x = 0.0
velocity_integral_y = 0.0
velocity_derivative_x = 0.0
velocity_derivative_y = 0.0
last_velocity_error_x = 0.0
last_velocity_error_y = 0.0
# Flight state
flight_phase = "IDLE"
flight_active = False
sensor_test_active = False  # New variable for sensor test state
scf_instance = None
position_integration_enabled = False
# Maneuver state
maneuver_active = False
target_position_x = 0.0
target_position_y = 0.0
# Shape maneuver state
shape_active = False
shape_waypoints = []
shape_index = 0
waypoint_start_time = 0.0
# Data history for plotting
max_history_points = 300  # Increased for better visualization
time_history = deque(maxlen=max_history_points)
velocity_x_history_plot = deque(maxlen=max_history_points)
velocity_y_history_plot = deque(maxlen=max_history_points)
position_x_history = deque(maxlen=max_history_points)
position_y_history = deque(maxlen=max_history_points)
correction_vx_history = deque(maxlen=max_history_points)
correction_vy_history = deque(maxlen=max_history_points)
height_history = deque(maxlen=max_history_points)
range_height_history = deque(maxlen=max_history_points)

# Complete trajectory history (never trimmed for data, but downsampled for UI)
MAX_PLOT_TRAJECTORY_POINTS = 5000  # Limit points drawn on 2D plot to prevent lag
complete_trajectory_x = []
complete_trajectory_y = []
key_release_points = []  # Store (x, y) tuples of release points
start_time = None
neo_controller = None
data_lock = threading.Lock()  # Protect shared data structures
# Debug counter for motion callback
debug_counter = 0

# Global logger function - can be set to GUI's log_to_output method
global_logger = None


def set_global_logger(logger_func):
    """Set the global logger function to redirect messages to GUI output window"""
    global global_logger
    global_logger = logger_func


def log_message(message):
    """Log a message using the global logger if available, otherwise print to console"""
    if global_logger is not None:
        try:
            global_logger(message)
        except Exception:
            # Fallback to print if logger fails
            print(message)
    else:
        print(message)


def check_link_safety(cf, logger=None):
    """
    Check if the Crazyflie is still connected and sensor data is fresh.
    Returns True if safe, False otherwise.
    """
    global flight_active, sensor_test_active

    # 1. Connection check
    if not cf.is_connected():
        if logger:
            logger("CRITICAL: Crazyflie disconnected!")
        return False

    # 2. Sensor heartbeat check (only if not in debug mode)
    if not DEBUG_MODE and sensor_data_ready:
        elapsed_since_last_data = time.time() - last_sensor_heartbeat
        if elapsed_since_last_data > DATA_TIMEOUT_THRESHOLD:
            if logger:
                logger(
                    f"CRITICAL: Sensor data timeout! ({elapsed_since_last_data:.2f}s delay)"
                )
            return False

    return True


# === HELPER FUNCTIONS ===
# Inline robust CRTP send + NeoPixel helpers (adapted from neopixel_control.py)
def _send_crtp_with_fallback(cf, port, channel, payload: bytes):
    header = ((port & 0x0F) << 4) | (channel & 0x0F)

    class _PacketObj:
        def __init__(self, header, data: bytes):
            self.header = header
            self.data = data
            try:
                self.datat = tuple(data)
            except Exception:
                self.datat = tuple()

        def is_data_size_valid(self):
            return len(self.data) <= 30

        @property
        def size(self):
            return len(self.data)

        def raw(self):
            return bytes([self.header]) + self.data

    pkt_obj = _PacketObj(header, payload)

    # 1) Crazyflie.send_packet if available
    try:
        send_fn = getattr(cf, "send_packet", None)
        if callable(send_fn):
            try:
                send_fn(pkt_obj)
                return
            except Exception:
                pass
    except Exception:
        pass

    # 2) low-level link object: _link or link
    try:
        link = getattr(cf, "_link", None) or getattr(cf, "link", None)
        if link is not None:
            if hasattr(link, "sendPacket"):
                try:
                    link.sendPacket(pkt_obj)
                    return
                except Exception:
                    pass
            if hasattr(link, "send_packet"):
                try:
                    link.send_packet(pkt_obj)
                    return
                except Exception:
                    pass
    except Exception:
        pass

    # 3) cflib.crtp.send_packet fallback (object or raw bytes)
    try:
        import cflib.crtp as _crtp

        sendp = getattr(_crtp, "send_packet", None)
        if callable(sendp):
            try:
                sendp(pkt_obj)
                return
            except Exception:
                try:
                    sendp(bytes([pkt_obj.header]) + pkt_obj.data)
                    return
                except Exception:
                    pass
    except Exception:
        pass

    raise RuntimeError(
        "Unable to send CRTP packet: no send method available on Crazyflie instance"
    )


# NeoPixel utility wrappers that reuse the same Crazyflie link
def np_set_pixel(cf, index, r, g, b):
    _send_crtp_with_fallback(
        cf, CRTP_PORT_NEOPIXEL, NEOPIXEL_CHANNEL_SET_PIXEL, bytes([index, r, g, b])
    )


def np_show(cf):
    _send_crtp_with_fallback(cf, CRTP_PORT_NEOPIXEL, NEOPIXEL_CHANNEL_SHOW, b"")


def np_clear(cf):
    _send_crtp_with_fallback(cf, CRTP_PORT_NEOPIXEL, NEOPIXEL_CHANNEL_CLEAR, b"")


def np_start_blink(cf, on_ms=500, off_ms=500):
    data = bytes(
        [1, (on_ms >> 8) & 0xFF, on_ms & 0xFF, (off_ms >> 8) & 0xFF, off_ms & 0xFF]
    )
    _send_crtp_with_fallback(cf, CRTP_PORT_NEOPIXEL, NEOPIXEL_CHANNEL_BLINK, data)


def np_stop_blink(cf):
    data = bytes([0, 0, 0, 0, 0])
    _send_crtp_with_fallback(cf, CRTP_PORT_NEOPIXEL, NEOPIXEL_CHANNEL_BLINK, data)


def np_set_all(cf, r, g, b):
    """Set all NeoPixels to same RGB using SET_ALL CRTP channel."""
    # The firmware uses the special broadcast index 0xFF on the SET_PIXEL
    # channel to indicate "set all". That avoids adding a channel value
    # beyond the 2-bit channel field in the CRTP packet.
    _send_crtp_with_fallback(
        cf,
        CRTP_PORT_NEOPIXEL,
        NEOPIXEL_CHANNEL_SET_PIXEL,
        bytes([0xFF, r & 0xFF, g & 0xFF, b & 0xFF]),
    )


def _try_send_with_retries(cf, fn, *args, retries=NP_SEND_RETRIES, logger=None):
    """Call a np_* function with retries and small inter-packet delay.
    fn is expected to be a function taking (cf, *args).
    Returns True on success, False on failure.
    """
    last_exc = None
    fn_name = getattr(fn, "__name__", repr(fn))
    for attempt in range(1, retries + 1):
        try:
            fn(cf, *args)
            return True
        except Exception as e:
            last_exc = e
            if logger:
                logger(f"[NeoPixel] Attempt {attempt} failed: {e}")
            time.sleep(NP_PACKET_DELAY)
    if logger:
        logger(f"[NeoPixel] Failed after {retries} attempts: {last_exc}")
    return False


def calculate_velocity(delta_value, altitude):
    """Convert optical flow delta to linear velocity"""
    if altitude <= 0:
        return 0.0
    if USE_HEIGHT_SCALING:
        # Original height-dependent calculation
        velocity_constant = (5.4 * DEG_TO_RAD) / (30.0 * DT)
        velocity = delta_value * altitude * velocity_constant
    else:
        # Simplified calculation without height dependency
        # Using empirical scaling factor
        velocity = delta_value * OPTICAL_FLOW_SCALE * DT
    return velocity


def smooth_velocity(new_velocity, history):
    """Simple 2-point smoothing filter with adjustable alpha"""
    history[1] = history[0]
    history[0] = new_velocity
    alpha = VELOCITY_SMOOTHING_ALPHA  # Use the global variable
    smoothed = (history[0] * alpha) + (history[1] * (1 - alpha))
    if abs(smoothed) < VELOCITY_THRESHOLD:
        smoothed = 0.0
    return smoothed


def integrate_position(vx, vy, dt):
    """Dead reckoning: integrate velocity to position"""
    global integrated_position_x, integrated_position_y
    if dt <= 0 or dt > 0.1:
        return
    # Simple integration
    integrated_position_x += vx * dt
    integrated_position_y += vy * dt
    # Apply drift compensation when moving slowly
    velocity_magnitude = (vx * vx + vy * vy) ** 0.5
    if velocity_magnitude < VELOCITY_THRESHOLD * 2:
        integrated_position_x -= integrated_position_x * DRIFT_COMPENSATION_RATE * dt
        integrated_position_y -= integrated_position_y * DRIFT_COMPENSATION_RATE * dt
    # Clamp position error
    integrated_position_x = max(
        -MAX_POSITION_ERROR, min(MAX_POSITION_ERROR, integrated_position_x)
    )
    integrated_position_y = max(
        -MAX_POSITION_ERROR, min(MAX_POSITION_ERROR, integrated_position_y)
    )


def periodic_position_reset():
    """Reset integrated position every few seconds"""
    global integrated_position_x, integrated_position_y, last_reset_time
    current_time = time.time()
    if current_time - last_reset_time >= PERIODIC_RESET_INTERVAL:
        integrated_position_x = 0.0
        integrated_position_y = 0.0
        last_reset_time = current_time
        return True
    return False



def apply_firmware_parameters(cf, logger=None):
    """
    Apply custom vertical PID and thrust parameters to the drone's brain.
    Only takes effect if ENABLE_FIRMWARE_PARAMS is True.
    """
    if not ENABLE_FIRMWARE_PARAMS:
        return
        
    try:
        if logger: logger("Applying custom firmware parameters (Z-Axis/Thrust)...")
        # Set parameters as strings since cflib expects that/it's safer for radio transport
        cf.param.set_value('posCtlPid.thrustBase', str(FW_THRUST_BASE))
        cf.param.set_value('posCtlPid.zKp', str(FW_Z_POS_KP))
        cf.param.set_value('velCtlPid.vzKp', str(FW_Z_VEL_KP))
        
        # Brief wait to ensure the drone processed the parameters
        time.sleep(0.2)
        
        # Verify the most important one
        actual_thrust = cf.param.get_value('posCtlPid.thrustBase')
        if logger: logger(f"Firmware configured: thrustBase={actual_thrust}, zKp={FW_Z_POS_KP}, vzKp={FW_Z_VEL_KP}")
        
    except Exception as e:
        if logger: logger(f"WARNING: Failed to set firmware parameters: {str(e)}")


def reset_position_tracking(reset_integrals=True):
    """Reset integrated position tracking to prevent sensor drift"""
    global integrated_position_x, integrated_position_y, last_integration_time, last_reset_time, position_integration_enabled
    global position_integral_x, position_integral_y, velocity_integral_x, velocity_integral_y
    global last_position_error_x, last_position_error_y, last_velocity_error_x, last_velocity_error_y
    global last_sensor_heartbeat
    integrated_position_x = 0.0
    integrated_position_y = 0.0
    last_integration_time = time.time()
    last_sensor_heartbeat = time.time()
    last_reset_time = time.time()
    position_integration_enabled = True
    
    if reset_integrals:
        # Reset PID state - Use this only at the very start of a flight
        position_integral_x = 0.0
        position_integral_y = 0.0
        velocity_integral_x = 0.0
        velocity_integral_y = 0.0
        last_position_error_x = 0.0
        last_position_error_y = 0.0
        last_velocity_error_x = 0.0
        last_velocity_error_y = 0.0


def calculate_position_hold_corrections(target_x=None, target_y=None):
    """Calculate control corrections using PID controllers"""
    global current_correction_vx, current_correction_vy
    global position_integral_x, position_integral_y, position_derivative_x, position_derivative_y
    global last_position_error_x, last_position_error_y
    global velocity_integral_x, velocity_integral_y, velocity_derivative_x, velocity_derivative_y
    global last_velocity_error_x, last_velocity_error_y
    global maneuver_active, target_position_x, target_position_y

    if not sensor_data_ready or current_height <= 0:
        current_correction_vx = 0.0
        current_correction_vy = 0.0
        return 0.0, 0.0

    # Fallback to global targets if not provided explicitly
    if target_x is None: target_x = target_position_x
    if target_y is None: target_y = target_position_y

    # Calculate position errors (negative because we want to correct toward target)
    position_error_x = -(integrated_position_x - target_x)
    position_error_y = -(integrated_position_y - target_y)

    # Calculate velocity errors (negative because we want to dampen velocity)
    velocity_error_x = -current_vx
    velocity_error_y = -current_vy

    # Position PID Controller
    # Proportional
    position_p_x = position_error_x * POSITION_KP
    position_p_y = position_error_y * POSITION_KP
    # Integral (with anti-windup)
    position_integral_x += position_error_x * CONTROL_UPDATE_RATE
    position_integral_y += position_error_y * CONTROL_UPDATE_RATE
    # Anti-windup: limit integral term
    position_integral_x = max(-0.1, min(0.1, position_integral_x))
    position_integral_y = max(-0.1, min(0.1, position_integral_y))
    position_i_x = position_integral_x * POSITION_KI
    position_i_y = position_integral_y * POSITION_KI
    # Derivative
    position_derivative_x = (
        position_error_x - last_position_error_x
    ) / CONTROL_UPDATE_RATE
    position_derivative_y = (
        position_error_y - last_position_error_y
    ) / CONTROL_UPDATE_RATE
    position_d_x = position_derivative_x * POSITION_KD
    position_d_y = position_derivative_y * POSITION_KD
    # Store current errors for next iteration
    last_position_error_x = position_error_x
    last_position_error_y = position_error_y

    # Velocity PID Controller
    # Proportional
    velocity_p_x = velocity_error_x * VELOCITY_KP
    velocity_p_y = velocity_error_y * VELOCITY_KP
    # Integral (with anti-windup)
    velocity_integral_x += velocity_error_x * CONTROL_UPDATE_RATE
    velocity_integral_y += velocity_error_y * CONTROL_UPDATE_RATE
    # Anti-windup: limit integral term
    velocity_integral_x = max(-0.05, min(0.05, velocity_integral_x))
    velocity_integral_y = max(-0.05, min(0.05, velocity_integral_y))
    velocity_i_x = velocity_integral_x * VELOCITY_KI
    velocity_i_y = velocity_integral_y * VELOCITY_KI
    # Derivative
    velocity_derivative_x = (
        velocity_error_x - last_velocity_error_x
    ) / CONTROL_UPDATE_RATE
    velocity_derivative_y = (
        velocity_error_y - last_velocity_error_y
    ) / CONTROL_UPDATE_RATE
    velocity_d_x = velocity_derivative_x * VELOCITY_KD
    velocity_d_y = velocity_derivative_y * VELOCITY_KD
    # Store current errors for next iteration
    last_velocity_error_x = velocity_error_x
    last_velocity_error_y = velocity_error_y

    # Combine PID outputs
    position_correction_vx = position_p_x + position_i_x + position_d_x
    position_correction_vy = position_p_y + position_i_y + position_d_y
    velocity_correction_vx = velocity_p_x + velocity_i_x + velocity_d_x
    velocity_correction_vy = velocity_p_y + velocity_i_y + velocity_d_y

    # Combine position and velocity corrections
    total_vx = position_correction_vx + velocity_correction_vx
    total_vy = position_correction_vy + velocity_correction_vy

    # Apply approach damping when close to target to reduce overshoot
    distance_to_target = (
        (integrated_position_x - target_x) ** 2
        + (integrated_position_y - target_y) ** 2
    ) ** 0.5
    
    if distance_to_target < 0.1:  # Simple 10cm damping zone
        velocity_magnitude = (current_vx**2 + current_vy**2) ** 0.5
        if velocity_magnitude > 0.05:
            total_vx *= 0.8
            total_vy *= 0.8

    # Apply limits
    total_vx = max(-MAX_CORRECTION, min(MAX_CORRECTION, total_vx))
    total_vy = max(-MAX_CORRECTION, min(MAX_CORRECTION, total_vy))

    # Store for GUI display
    current_correction_vx = total_vx
    current_correction_vy = total_vy
    return total_vx, total_vy


def update_history():
    """Update data history for plotting (Thread-Safe)"""
    global start_time
    if start_time is None:
        start_time = time.time()
    current_time = time.time() - start_time
    
    with data_lock:
        # Add new data points (deques handle trimming automatically)
        time_history.append(current_time)
        velocity_x_history_plot.append(current_vx)
        velocity_y_history_plot.append(current_vy)
        position_x_history.append(integrated_position_x)
        position_y_history.append(integrated_position_y)
        correction_vx_history.append(current_correction_vx)
        correction_vy_history.append(current_correction_vy)
        height_history.append(current_height)
        range_height_history.append(current_range_height)
        
        # Add to complete trajectory
        complete_trajectory_x.append(integrated_position_x)
        complete_trajectory_y.append(integrated_position_y)


def motion_callback(timestamp, data, logconf):
    """Motion sensor data callback"""
    global current_height, current_range_height, motion_delta_x, motion_delta_y, sensor_data_ready
    global current_vx, current_vy, last_integration_time, last_sensor_heartbeat
    global debug_counter

    # Update heartbeat immediately
    last_sensor_heartbeat = time.time()

    # Get sensor data
    current_height = data.get("stateEstimate.z", 0)
    # Get raw range and convert to meters (mm -> m)
    raw_range_mm = data.get("range.zrange", 0)
    current_range_height = raw_range_mm / 1000.0 if raw_range_mm else 0.0
    motion_delta_x = data.get("motion.deltaX", 0)
    motion_delta_y = data.get("motion.deltaY", 0)
    sensor_data_ready = True

    # Calculate velocities
    raw_velocity_x = calculate_velocity(motion_delta_x, current_height)
    raw_velocity_y = calculate_velocity(motion_delta_y, current_height)

    # Debug output every 100 callbacks (reduce console spam)
    # debug_counter += 1
    # if debug_counter % 100 == 0 and (
    #     abs(motion_delta_x) > 0 or abs(motion_delta_y) > 0
    # ):
    #     print(
    #         f"Sensor Debug - Height: {current_height:.3f}m, "
    #         f"Raw Motion: X={motion_delta_x}, Y={motion_delta_y}, "
    #         f"Velocities: X={raw_velocity_x:.4f}, Y={raw_velocity_y:.4f}"
    #     )

    # Apply smoothing
    current_vx = smooth_velocity(raw_velocity_x, velocity_x_history)
    current_vy = smooth_velocity(raw_velocity_y, velocity_y_history)

    # Dead reckoning position integration (only when enabled)
    current_time = time.time()
    dt = current_time - last_integration_time
    if 0.001 <= dt <= 0.1 and position_integration_enabled:
        integrate_position(current_vx, current_vy, dt)
    last_integration_time = current_time

    # Update history for GUI
    update_history()


def battery_callback(timestamp, data, logconf):
    """Battery voltage data callback"""
    global current_battery_voltage, battery_data_ready
    # Get battery voltage
    current_battery_voltage = data.get("pm.vbat", 0.0)
    battery_data_ready = True


def setup_logging(cf, logger=None):
    """Setup motion sensor and battery voltage logging"""
    log_motion = LogConfig(name="Motion", period_in_ms=SENSOR_PERIOD_MS)
    log_battery = LogConfig(
        name="Battery", period_in_ms=500
    )  # Check battery every 500ms

    try:
        toc = cf.log.toc.toc
        # Setup motion logging
        motion_variables = [
            ("motion.deltaX", "int16_t"),
            ("motion.deltaY", "int16_t"),
            ("stateEstimate.z", "float"),
            ("range.zrange", "uint16_t"),
        ]
        added_motion_vars = []
        for var_name, var_type in motion_variables:
            group, name = var_name.split(".")
            if group in toc and name in toc[group]:
                try:
                    log_motion.add_variable(var_name, var_type)
                    added_motion_vars.append(var_name)
                except Exception as e:
                    if logger:
                        logger(f"Failed to add motion variable {var_name}: {e}")
                    else:
                        # Use global logger to redirect to output window
                        log_message(f"Failed to add motion variable {var_name}: {e}")
            else:
                if logger:
                    logger(f"Motion variable not found: {var_name}")
                else:
                    # Use global logger to redirect to output window
                    log_message(f"Motion variable not found: {var_name}")

        if len(added_motion_vars) < 2:
            if logger:
                logger("ERROR: Not enough motion variables found!")
            else:
                # Use global logger to redirect to output window
                log_message("ERROR: Not enough motion variables found!")
            return None, None

        # Setup battery logging
        battery_variables = [("pm.vbat", "float")]
        added_battery_vars = []
        for var_name, var_type in battery_variables:
            group, name = var_name.split(".")
            if group in toc and name in toc[group]:
                try:
                    log_battery.add_variable(var_name, var_type)
                    added_battery_vars.append(var_name)
                    if logger:
                        logger(f"Added battery variable: {var_name}")
                    else:
                        # Use global logger to redirect to output window
                        log_message(f"Added battery variable: {var_name}")
                except Exception as e:
                    if logger:
                        logger(f"Failed to add battery variable {var_name}: {e}")
                    else:
                        # Use global logger to redirect to output window
                        log_message(f"Failed to add battery variable {var_name}: {e}")
            else:
                if logger:
                    logger(f"Battery variable not found: {var_name}")
                else:
                    # Use global logger to redirect to output window
                    log_message(f"Battery variable not found: {var_name}")

        # Setup callbacks
        log_motion.data_received_cb.add_callback(motion_callback)
        if len(added_battery_vars) > 0:
            log_battery.data_received_cb.add_callback(battery_callback)

        # Add configurations
        cf.log.add_config(log_motion)
        if len(added_battery_vars) > 0:
            cf.log.add_config(log_battery)

        time.sleep(0.5)

        # Validate configurations
        if not log_motion.valid:
            if logger:
                logger("ERROR: Motion log configuration invalid!")
            else:
                # Use global logger to redirect to output window
                log_message("ERROR: Motion log configuration invalid!")
            return None, None
        if len(added_battery_vars) > 0 and not log_battery.valid:
            if logger:
                logger("WARNING: Battery log configuration invalid!")
            else:
                # Use global logger to redirect to output window
                log_message("WARNING: Battery log configuration invalid!")
            # Continue without battery logging
            log_battery = None

        # Start logging
        log_motion.start()
        if log_battery:
            log_battery.start()

        time.sleep(0.5)
        if logger:
            logger(
                f"Logging started - Motion: {len(added_motion_vars)} vars, Battery: {len(added_battery_vars)} vars"
            )
        else:
            # Use global logger to redirect to output window
            log_message(
                f"Logging started - Motion: {len(added_motion_vars)} vars, Battery: {len(added_battery_vars)} vars"
            )
        return log_motion, log_battery

    except Exception as e:
        error_msg = f"Logging setup failed: {str(e)}"
        if logger:
            logger(error_msg)
        else:
            # Use global logger to redirect to output window
            log_message(error_msg)
        raise Exception(error_msg)


def init_csv_logging(logger=None):
    """Initialize CSV logging for position and height"""
    global log_file, log_writer
    if not DRONE_CSV_LOGGING:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"drone_flight_log_{timestamp}.csv"
    log_file = open(log_filename, mode="w", newline="")
    log_writer = csv.writer(log_file)
    # Write header
    log_writer.writerow(
        [
            "Timestamp (s)",
            "Integrated Position X (m)",
            "Integrated Position Y (m)",
            "Height (m)",
            "Range (m)",
            "Velocity X (m/s)",
            "Velocity Y (m/s)",
            "Correction VX",
            "Correction VY",
        ]
    )
    if logger:
        logger(f"Logging to CSV: {log_filename}")
    else:
        # Use global logger to redirect to output window
        log_message(f"Logging to CSV: {log_filename}")


def log_to_csv():
    """Log current state to CSV if logging is active"""
    global log_writer, start_time
    if not DRONE_CSV_LOGGING or log_writer is None or start_time is None:
        return
    elapsed = time.time() - start_time
    log_writer.writerow(
        [
            f"{elapsed:.3f}",
            f"{integrated_position_x:.6f}",
            f"{integrated_position_y:.6f}",
            f"{current_height:.6f}",
            f"{current_range_height:.6f}",
            f"{current_vx:.6f}",
            f"{current_vy:.6f}",
            f"{current_correction_vx:.6f}",
            f"{current_correction_vy:.6f}",
        ]
    )


def close_csv_logging(logger=None):
    """Close CSV log file"""
    global log_file
    if log_file:
        log_file.close()
        log_file = None
        if logger:
            logger("CSV log closed.")
        else:
            # Use global logger to redirect to output window
            log_message("CSV log closed.")


class DeadReckoningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Dead Reckoning Position Hold with Maneuvers - Real-Time Monitor"
        )
        self.root.geometry("1400x950")  # Increased height to accommodate new controls

        # Flight control variables
        self.flight_thread = None
        self.flight_running = False
        self.sensor_test_thread = None  # New thread variable for sensor test
        self.sensor_test_running = False  # New flag for sensor test running state
        self.joystick_thread = None  # Thread for joystick control
        self.joystick_active = False  # Flag for joystick control active
        self.joystick_keys = {
            "w": False,
            "a": False,
            "s": False,
            "d": False,
        }  # Joystick key states

        # Output window visibility control
        self.show_output_window = True  # Boolean to enable/disable output window

        # Joystick position hold mode
        self.joystick_hold_at_origin = (
            False  # False = hold at current position, True = return to origin
        )

        # Track which keys are currently pressed to prevent duplicate logging
        self.key_pressed_flags = {
            "w": False,
            "a": False,
            "s": False,
            "d": False,
        }
        self._joystick_debug_counter = 0  # Counter for throttling debug logs

        self.create_ui()
        self.setup_plots()

        # Set up global logging to redirect fallback messages to the output window
        set_global_logger(self.log_to_output)

        # NeoPixel state (lazy connect). We store a Crazyflie instance here
        # and a flag indicating whether this GUI owns the link (so we can
        # close it when appropriate). Reuse global SyncCrazyflie when present.
        self.neo_cf = None
        self._neo_owns_link = False
        self.blinking = False
        self.low_battery_blinking = False  # Flag for low battery blink
        # Persistent last color for NeoPixels. This is used so blinking and
        # static color are independent and stopping blink does not lose the
        # previously-set color. Initialize from the UI defaults.
        try:
            r = int(self.rgb_r_var.get())
            g = int(self.rgb_g_var.get())
            b = int(self.rgb_b_var.get())
        except Exception:
            r, g, b = 255, 255, 255
        self.neo_last_color = (r, g, b)

        # Start animation
        self.anim = animation.FuncAnimation(
            self.fig, self.update_plots, interval=100, cache_frame_data=False
        )

        # Bind keyboard events for joystick control
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.root.focus_set()  # Ensure root window has focus for key events

    def create_ui(self):
        """Create the user interface"""
        # Control panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Load and display logo in top right
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(script_dir, "litewing_logo.png")
            if os.path.exists(logo_path):
                pil_img = Image.open(logo_path)
                # Resize logo to a reasonable size (e.g., height of 40px)
                aspect_ratio = pil_img.width / pil_img.height
                new_height = 60
                new_width = int(new_height * aspect_ratio)
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(pil_img)
                self.logo_label = tk.Label(control_frame, image=self.logo_img)
                self.logo_label.pack(side=tk.RIGHT, padx=10)
        except Exception as e:
            print(f"Error loading logo: {e}")


        # Flight control buttons
        self.start_button = tk.Button(
            control_frame,
            text="Start Flight",
            command=self.start_flight,
            bg="green",
            fg="white",
            font=("Arial", 12),
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        # Sensor Test button - starts a separate sensor test thread (non-flight)
        self.sensor_test_button = tk.Button(
            control_frame,
            text="Sensor Test",
            command=self.start_sensor_test,
            bg="lightblue",
            fg="black",
            font=("Arial", 12),
        )
        self.sensor_test_button.pack(side=tk.LEFT, padx=10)

        # Reset Battery button - resets battery voltage reading
        self.reset_battery_button = tk.Button(
            control_frame,
            text="Reset Battery",
            command=self.reset_battery,
            bg="orange",
            fg="black",
            font=("Arial", 11),
        )
        self.reset_battery_button.pack(side=tk.LEFT, padx=5)

        # Create a frame for the checkboxes to stack them vertically
        checkboxes_frame = tk.Frame(control_frame)
        checkboxes_frame.pack(side=tk.LEFT, padx=(10, 0))

        # Enable logging checkbox for sensor test
        self.enable_sensor_logging_var = tk.BooleanVar(value=False)
        self.enable_sensor_logging_check = tk.Checkbutton(
            checkboxes_frame,
            text="Log Sensor Test",
            variable=self.enable_sensor_logging_var,
        )
        self.enable_sensor_logging_check.pack(side=tk.TOP, anchor=tk.W)

        # Enable debug mode checkbox (stacked below sensor test in same column)
        self.enable_debug_mode_var = tk.BooleanVar(value=DEBUG_MODE)
        self.enable_debug_mode_check = tk.Checkbutton(
            checkboxes_frame,
            text="Enable Debug Mode",
            variable=self.enable_debug_mode_var,
            command=self.toggle_debug_mode,
        )
        self.enable_debug_mode_check.pack(side=tk.TOP, anchor=tk.W)
        
        # Primary CSV logging toggle
        self.enable_csv_logging_var = tk.BooleanVar(value=DRONE_CSV_LOGGING)
        self.enable_csv_logging_check = tk.Checkbutton(
            checkboxes_frame,
            text="Enable CSV Logging",
            variable=self.enable_csv_logging_var,
            command=self.toggle_csv_logging,
        )
        self.enable_csv_logging_check.pack(side=tk.TOP, anchor=tk.W)

        # Blink NeoPixel button - New button
        self.blink_button = tk.Button(
            control_frame,
            text="Blink LEDs",
            command=self.toggle_blink,
            bg="yellow",
            state=tk.DISABLED,
            fg="black",
            font=("Arial", 12),
        )
        self.blink_button.pack(side=tk.LEFT, padx=10)

        self.clear_leds_button = tk.Button(
            control_frame,
            text="Clear LEDs",
            command=self.clear_leds,
            bg="lightgrey",
            state=tk.DISABLED,
            fg="black",
            font=("Arial", 11),
        )
        self.clear_leds_button.pack(side=tk.LEFT, padx=6)

        # RGB controls (R, G, B spinboxes) and Set Color button
        rgb_frame = tk.Frame(control_frame)
        rgb_frame.pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(rgb_frame, text="R:").pack(side=tk.LEFT)
        self.rgb_r_var = tk.StringVar(value="255")
        self.rgb_r_spin = tk.Spinbox(
            rgb_frame, from_=0, to=255, width=4, textvariable=self.rgb_r_var
        )
        self.rgb_r_spin.pack(side=tk.LEFT)
        tk.Label(rgb_frame, text="G:").pack(side=tk.LEFT, padx=(6, 0))
        self.rgb_g_var = tk.StringVar(value="255")
        self.rgb_g_spin = tk.Spinbox(
            rgb_frame, from_=0, to=255, width=4, textvariable=self.rgb_g_var
        )
        self.rgb_g_spin.pack(side=tk.LEFT)
        tk.Label(rgb_frame, text="B:").pack(side=tk.LEFT, padx=(6, 0))
        self.rgb_b_var = tk.StringVar(value="255")
        self.rgb_b_spin = tk.Spinbox(
            rgb_frame, from_=0, to=255, width=4, textvariable=self.rgb_b_var
        )
        self.rgb_b_spin.pack(side=tk.LEFT)

        self.set_color_button = tk.Button(
            control_frame,
            text="Set Color",
            command=self.set_color_from_ui,
            bg="lightgrey",
            state=tk.DISABLED,
            fg="black",
            font=("Arial", 11),
        )
        self.set_color_button.pack(side=tk.LEFT, padx=6)

        # Flight status
        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_label = tk.Label(
            control_frame,
            textvariable=self.status_var,
            font=("Arial", 12, "bold"),
            fg="blue",
        )
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Main frame for layout
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left side - Parameters
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        # Flight Control Parameters
        runtime_frame = tk.LabelFrame(
            left_frame, text="Flight Control Parameters", padx=10, pady=10
        )
        runtime_frame.pack(fill=tk.X, pady=5)

        # Create runtime controls
        self.create_runtime_controls(runtime_frame)

        # Maneuver Controls
        maneuver_frame = tk.LabelFrame(left_frame, text="Controls", padx=10, pady=10)
        maneuver_frame.pack(fill=tk.X, pady=5)

        # Create maneuver controls
        self.create_maneuver_controls(maneuver_frame)

        # PID Tuning Controls
        pid_frame = tk.LabelFrame(
            left_frame, text="PID Tuning Controls", padx=10, pady=10
        )
        pid_frame.pack(fill=tk.X, pady=5)

        # Create PID tuning controls
        self.create_pid_controls(pid_frame)

        # Right side - Values and Plots
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Real-time values display (Top of right side)
        values_frame = tk.LabelFrame(
            right_frame, text="Real-Time Values", padx=10, pady=10
        )
        values_frame.pack(fill=tk.X, pady=5)

        # Create value displays in a grid
        self.create_value_displays(values_frame)

        # Matplotlib figure (Bottom of right side, expanded)
        self.fig = Figure(figsize=(12, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_maneuver_controls(self, parent):
        """Create maneuver control buttons in compact layout"""
        # Main container frame for both maneuver and joystick controls
        main_controls_frame = tk.Frame(parent)
        main_controls_frame.pack(fill=tk.X, pady=5)

        # Left side - Maneuver Controls (more compact)
        maneuver_frame = tk.LabelFrame(
            main_controls_frame, text="Maneuver Controls", padx=5, pady=5
        )
        maneuver_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))

        # Maneuver distance control (compact)
        distance_frame = tk.Frame(maneuver_frame)
        distance_frame.pack(fill=tk.X, pady=2)
        tk.Label(distance_frame, text="Distance (m):", width=12).pack(side=tk.LEFT)
        self.maneuver_distance_var = tk.StringVar(value=str(MANEUVER_DISTANCE))
        self.maneuver_distance_entry = tk.Entry(
            distance_frame, textvariable=self.maneuver_distance_var, width=6
        )
        self.maneuver_distance_entry.pack(side=tk.LEFT, padx=2)

        # Joystick layout frame (compact 3x3 grid)
        joystick_frame = tk.Frame(maneuver_frame)
        joystick_frame.pack(pady=5)

        # Create a 3x3 grid for joystick layout
        for i in range(3):
            joystick_frame.grid_rowconfigure(i, weight=1)
            joystick_frame.grid_columnconfigure(i, weight=1)

        # Forward button (top center) - smaller
        self.forward_button = tk.Button(
            joystick_frame,
            text="↑\nForward",
            command=self.maneuver_forward,
            bg="blue",
            fg="white",
            font=("Arial", 9),
            width=6,
            height=2,
        )
        self.forward_button.grid(row=0, column=1, padx=2, pady=2)

        # Left button (middle left) - smaller
        self.left_button = tk.Button(
            joystick_frame,
            text="→\nRight",
            command=self.maneuver_right,
            bg="blue",
            fg="white",
            font=("Arial", 9),
            width=6,
            height=2,
        )
        self.left_button.grid(row=1, column=2, padx=2, pady=2)

        # Stop button (center) - smaller
        self.stop_button = tk.Button(
            joystick_frame,
            text="STOP",
            command=self.stop_maneuver,
            bg="red",
            fg="white",
            font=("Arial", 10, "bold"),
            width=6,
            height=2,
        )
        self.stop_button.grid(row=1, column=1, padx=2, pady=2)

        # Right button (middle right) - smaller
        self.right_button = tk.Button(
            joystick_frame,
            text="←\nLeft",
            command=self.maneuver_left,
            bg="blue",
            fg="white",
            font=("Arial", 9),
            width=6,
            height=2,
        )
        self.right_button.grid(row=1, column=0, padx=2, pady=2)

        # Backward button (bottom center) - smaller
        self.backward_button = tk.Button(
            joystick_frame,
            text="↓\nBackward",
            command=self.maneuver_backward,
            bg="blue",
            fg="white",
            font=("Arial", 9),
            width=6,
            height=2,
        )
        self.backward_button.grid(row=2, column=1, padx=2, pady=2)

        # Shape maneuver buttons (compact horizontal layout)
        shape_frame = tk.Frame(maneuver_frame)
        shape_frame.pack(pady=3)
        tk.Label(shape_frame, text="Shapes:").pack(side=tk.LEFT, padx=(0, 3))
        self.square_button = tk.Button(
            shape_frame,
            text="Square",
            command=self.maneuver_square,
            bg="purple",
            fg="white",
            font=("Arial", 9),
            width=6,
        )
        self.square_button.pack(side=tk.LEFT, padx=1)

        # Hop-style landing controls removed for simplification

        # Right side - Joystick Controls (more compact)
        joystick_control_frame = tk.LabelFrame(
            main_controls_frame, text="Joystick Control", padx=5, pady=5
        )
        joystick_control_frame.pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(3, 0)
        )

        # Joystick sensitivity control (compact)
        sensitivity_frame = tk.Frame(joystick_control_frame)
        sensitivity_frame.pack(fill=tk.X, pady=2)
        tk.Label(sensitivity_frame, text="Sensitivity:", width=10).pack(side=tk.LEFT)
        self.joystick_sensitivity_var = tk.StringVar(value=str(JOYSTICK_SENSITIVITY))
        self.joystick_sensitivity_entry = tk.Entry(
            sensitivity_frame, textvariable=self.joystick_sensitivity_var, width=5
        )
        self.joystick_sensitivity_entry.pack(side=tk.LEFT, padx=2)
        tk.Label(
            sensitivity_frame, text="(0.1-2.0)", font=("Arial", 7), fg="gray"
        ).pack(side=tk.LEFT)

        # Joystick button layout frame (compact 3x3 grid)
        joystick_buttons_frame = tk.Frame(joystick_control_frame)
        joystick_buttons_frame.pack(pady=5)

        # Create a 3x3 grid for joystick layout
        for i in range(3):
            joystick_buttons_frame.grid_rowconfigure(i, weight=1)
            joystick_buttons_frame.grid_columnconfigure(i, weight=1)

        # Forward button (top center) - smaller
        self.joystick_forward_button = tk.Button(
            joystick_buttons_frame,
            text="↑\nW",
            bg="green",
            fg="white",
            font=("Arial", 9),
            width=5,
            height=2,
        )
        self.joystick_forward_button.grid(row=0, column=1, padx=2, pady=2)
        self.joystick_forward_button.bind(
            "<ButtonPress>", lambda e: self.start_continuous_movement("w")
        )
        self.joystick_forward_button.bind(
            "<ButtonRelease>", lambda e: self.stop_continuous_movement("w")
        )

        # Left button (middle left) - smaller
        self.joystick_left_button = tk.Button(
            joystick_buttons_frame,
            text="←\nA",
            bg="green",
            fg="white",
            font=("Arial", 9),
            width=5,
            height=2,
        )
        self.joystick_left_button.grid(row=1, column=0, padx=2, pady=2)
        self.joystick_left_button.bind(
            "<ButtonPress>", lambda e: self.start_continuous_movement("a")
        )
        self.joystick_left_button.bind(
            "<ButtonRelease>", lambda e: self.stop_continuous_movement("a")
        )

        # Stop button (center) - smaller
        self.joystick_stop_button = tk.Button(
            joystick_buttons_frame,
            text="STOP",
            command=self.stop_joystick_control,
            bg="red",
            fg="white",
            font=("Arial", 10, "bold"),
            width=5,
            height=2,
        )
        self.joystick_stop_button.grid(row=1, column=1, padx=2, pady=2)

        # Right button (middle right) - smaller
        self.joystick_right_button = tk.Button(
            joystick_buttons_frame,
            text="→\nD",
            bg="green",
            fg="white",
            font=("Arial", 9),
            width=5,
            height=2,
        )
        self.joystick_right_button.grid(row=1, column=2, padx=2, pady=2)
        self.joystick_right_button.bind(
            "<ButtonPress>", lambda e: self.start_continuous_movement("d")
        )
        self.joystick_right_button.bind(
            "<ButtonRelease>", lambda e: self.stop_continuous_movement("d")
        )

        # Backward button (bottom center) - smaller
        self.joystick_backward_button = tk.Button(
            joystick_buttons_frame,
            text="↓\nS",
            bg="green",
            fg="white",
            font=("Arial", 9),
            width=5,
            height=2,
        )
        self.joystick_backward_button.grid(row=2, column=1, padx=2, pady=2)
        self.joystick_backward_button.bind(
            "<ButtonPress>", lambda e: self.start_continuous_movement("s")
        )
        self.joystick_backward_button.bind(
            "<ButtonRelease>", lambda e: self.stop_continuous_movement("s")
        )

        # Joystick control buttons (centered start button only)
        control_buttons_frame = tk.Frame(joystick_control_frame)
        control_buttons_frame.pack(fill=tk.X, pady=3)

        self.start_joystick_button = tk.Button(
            control_buttons_frame,
            text="Start Joystick",
            command=self.start_joystick_control,
            bg="green",
            fg="white",
            font=("Arial", 9, "bold"),
            width=15,
        )
        self.start_joystick_button.pack(expand=True)

        # Joystick position hold mode checkbox
        position_hold_frame = tk.Frame(joystick_control_frame)
        position_hold_frame.pack(fill=tk.X, pady=3)

        self.joystick_hold_origin_var = tk.BooleanVar(
            value=self.joystick_hold_at_origin
        )
        self.joystick_hold_origin_check = tk.Checkbutton(
            position_hold_frame,
            text="Hold at Origin (uncheck to hold at current position)",
            variable=self.joystick_hold_origin_var,
            command=self.toggle_joystick_hold_mode,
            font=("Arial", 8),
        )
        self.joystick_hold_origin_check.pack()

        # Joystick status display (compact)
        self.joystick_status_var = tk.StringVar(value="Joystick: INACTIVE")
        tk.Label(
            joystick_control_frame,
            textvariable=self.joystick_status_var,
            font=("Arial", 9, "bold"),
            fg="blue",
        ).pack(pady=2)

    def maneuver_forward(self):
        """Execute forward maneuver"""
        try:
            distance = float(self.maneuver_distance_var.get())
            # For directional buttons, we want exactly 'distance' from 'here'
            # Reset position but keep learned auto-trim (integrals)
            reset_position_tracking(reset_integrals=False) 
            new_target_x = 0.0
            new_target_y = distance
            
            self.start_maneuver(new_target_x, new_target_y)
            self.log_to_output(f"Maneuver: Forward {distance:.2f}m")
        except ValueError:
            self.status_var.set("Status: Invalid maneuver distance")

    def maneuver_backward(self):
        """Execute backward maneuver"""
        try:
            distance = float(self.maneuver_distance_var.get())
            # Reset position but keep learned auto-trim (integrals)
            reset_position_tracking(reset_integrals=False)
            new_target_x = 0.0
            new_target_y = -distance
            
            self.start_maneuver(new_target_x, new_target_y)
            self.log_to_output(f"Maneuver: Backward {distance:.2f}m")
        except ValueError:
            self.status_var.set("Status: Invalid maneuver distance")

    def maneuver_left(self):
        """Execute left maneuver"""
        try:
            distance = float(self.maneuver_distance_var.get())
            # Reset position but keep learned auto-trim (integrals)
            reset_position_tracking(reset_integrals=False)
            new_target_x = distance
            new_target_y = 0.0
            
            self.start_maneuver(new_target_x, new_target_y)
            self.log_to_output(f"Maneuver: Left {distance:.2f}m")
        except ValueError:
            self.status_var.set("Status: Invalid maneuver distance")

    def maneuver_right(self):
        """Execute right maneuver"""
        try:
            distance = float(self.maneuver_distance_var.get())
            # Reset position but keep learned auto-trim (integrals)
            reset_position_tracking(reset_integrals=False)
            new_target_x = -distance
            new_target_y = 0.0
            
            self.start_maneuver(new_target_x, new_target_y)
            self.log_to_output(f"Maneuver: Right {distance:.2f}m")
        except ValueError:
            self.status_var.set("Status: Invalid maneuver distance")

    def stop_maneuver(self):
        """Stop the current maneuver and flight"""
        global maneuver_active, target_position_x, target_position_y, flight_active
        maneuver_active = False
        flight_active = False
        target_position_x = integrated_position_x
        target_position_y = integrated_position_y
        self.log_to_output("Maneuver and flight stopped")
        self.log_to_output("Maneuver stopped")

    def maneuver_square(self):
        """Execute square maneuver"""
        try:
            distance = float(self.maneuver_distance_var.get())
            # 1. Reset origin to "Here" first, but keep the learned auto-trim (integrals)
            reset_position_tracking(reset_integrals=False) 
            # 2. Calculate waypoints for square pattern (Right -> Forward -> Left -> Home)
            # +X=Left, -X=Right, +Y=Forward, -Y=Backward
            waypoints = [
                (-distance, 0.0),        # Stage 1: Move Right
                (-distance, distance),   # Stage 2: Move Forward
                (0.0, distance),        # Stage 3: Move Left
                (0.0, 0.0)              # Stage 4: Back Home
            ]
            self.start_shape_maneuver(waypoints)
            self.log_to_output(f"Maneuver: Square {distance:.2f}m initiated")
        except ValueError:
            self.status_var.set("Status: Invalid maneuver distance")

    def calculate_square_waypoints(self, distance):
        """Calculate waypoints for square pattern (deprecated - logic moved to maneuver_square)"""
        return []

    def start_shape_maneuver(self, waypoints):
        """Start a shape maneuver with waypoints or update current flight"""
        global shape_waypoints, shape_index, shape_active, maneuver_active, target_position_x, target_position_y, waypoint_start_time
        
        # If already flying, just update the waypoints and activate shape
        if self.flight_running:
            shape_waypoints = waypoints
            shape_index = 0
            shape_active = True
            maneuver_active = True
            target_position_x, target_position_y = shape_waypoints[0]
            waypoint_start_time = time.time()
            self.log_to_output(f"Shape maneuver updated mid-flight: {len(waypoints)} waypoints")
            return

        if not self.flight_running and not self.sensor_test_running:
            # Battery safety check
            if (
                current_battery_voltage > 0
                and current_battery_voltage < LOW_BATTERY_THRESHOLD
            ):
                self.status_var.set(
                    f"Status: Battery too low ({current_battery_voltage:.2f}V)! Cannot start maneuver."
                )
                return
            elif current_battery_voltage == 0.0:
                print("WARNING: Battery voltage unknown")

            # SENSOR SAFETY CHECK
            if not sensor_data_ready:
                self.status_var.set(
                    "Status: Sensor data not ready! Wait for height & motion readings."
                )
                return

            if current_height <= 0.0:
                self.status_var.set(
                    "Status: Invalid height reading! Ensure drone is powered and sensors active."
                )
                return

            # Set shape maneuver parameters
            shape_waypoints = waypoints
            shape_index = 0
            shape_active = True
            maneuver_active = True
            target_position_x, target_position_y = shape_waypoints[0]
            waypoint_start_time = time.time()  # Initialize timeout tracking

            # Proceed
            self.flight_running = True
            self.start_button.config(
                text="Stop Flight", command=self.emergency_stop, bg="red"
            )
            self.status_var.set(
                f"Status: Starting Shape Maneuver ({len(waypoints)} points) - Stabilizing height first..."
            )
            self.flight_thread = threading.Thread(target=self.flight_controller_thread)
            self.flight_thread.daemon = True
            self.flight_thread.start()

            # Log to output window
            self.log_to_output(f"Shape maneuver started: {len(waypoints)} waypoints")
        elif self.sensor_test_running:
            self.log_to_output(
                "Cannot start Shape Maneuver while Sensor Test is active."
            )
            self.status_var.set(
                "Status: Sensor Test Active - Cannot Start Shape Maneuver"
            )

    def start_maneuver(self, delta_x, delta_y):
        """Start a maneuver flight or update current flight"""
        global maneuver_active, target_position_x, target_position_y, waypoint_start_time
        
        # If already flying, just update the target and activate maneuver
        if self.flight_running:
            maneuver_active = True
            target_position_x = delta_x
            target_position_y = delta_y
            waypoint_start_time = time.time()
            self.log_to_output(f"Maneuver updated mid-flight: ({delta_x:.2f}, {delta_y:.2f})")
            return

        if not self.flight_running and not self.sensor_test_running:
            # Battery safety check
            if (
                current_battery_voltage > 0
                and current_battery_voltage < LOW_BATTERY_THRESHOLD
            ):
                self.status_var.set(
                    f"Status: Battery too low ({current_battery_voltage:.2f}V)! Cannot start maneuver."
                )
                return
            elif current_battery_voltage == 0.0:
                print("WARNING: Battery voltage unknown")

            # SENSOR SAFETY CHECK
            if not sensor_data_ready:
                self.status_var.set(
                    "Status: Sensor data not ready! Wait for height & motion readings."
                )
                return

            if current_height <= 0.0:
                self.status_var.set(
                    "Status: Invalid height reading! Ensure drone is powered and sensors active."
                )
                return

            # Set maneuver parameters
            maneuver_active = True
            target_position_x = delta_x
            target_position_y = delta_y
            waypoint_start_time = time.time()  # Reset timer for single maneuver

            # Proceed
            self.flight_running = True
            self.start_button.config(
                text="Stop Flight", command=self.emergency_stop, bg="red"
            )
            self.status_var.set(
                f"Status: Starting Maneuver ({delta_x:.2f}, {delta_y:.2f}) - Stabilizing height first..."
            )
            self.flight_thread = threading.Thread(target=self.flight_controller_thread)
            self.flight_thread.daemon = True
            self.flight_thread.start()

            # Log to output window
            self.log_to_output(f"Maneuver started: ({delta_x:.2f}, {delta_y:.2f})")
        elif self.sensor_test_running:
            self.log_to_output("Cannot start Maneuver while Sensor Test is active.")
            self.status_var.set("Status: Sensor Test Active - Cannot Start Maneuver")

    def create_value_displays(self, parent):
        """Create real-time value display widgets"""
        # Main container with left and right columns
        main_container = tk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left column - Value displays
        left_column = tk.Frame(main_container)
        left_column.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        # Row 1: Basic values
        row1 = tk.Frame(left_column)
        row1.pack(fill=tk.X, pady=2)
        self.height_var = tk.StringVar(value="Height: 0.000m")
        self.phase_var = tk.StringVar(value="Phase: IDLE")
        self.battery_var = tk.StringVar(value="Battery: 0.00V")
        tk.Label(
            row1, textvariable=self.height_var, font=("Arial", 11, "bold"), fg="blue"
        ).pack(side=tk.LEFT, padx=10)
        self.range_var = tk.StringVar(value="Range: 0.000m")
        tk.Label(
            row1, textvariable=self.range_var, font=("Arial", 11, "bold"), fg="cyan4"
        ).pack(side=tk.LEFT, padx=10)
        tk.Label(
            row1, textvariable=self.phase_var, font=("Arial", 11, "bold"), fg="red"
        ).pack(side=tk.LEFT, padx=10)
        tk.Label(
            row1, textvariable=self.battery_var, font=("Arial", 11, "bold"), fg="orange"
        ).pack(side=tk.LEFT, padx=20)

        # Row 2: Velocities
        row2 = tk.Frame(left_column)
        row2.pack(fill=tk.X, pady=2)
        self.vx_var = tk.StringVar(value="VX: 0.000 m/s")
        self.vy_var = tk.StringVar(value="VY: 0.000 m/s")
        tk.Label(row2, textvariable=self.vx_var, font=("Arial", 11)).pack(
            side=tk.LEFT, padx=20
        )
        tk.Label(row2, textvariable=self.vy_var, font=("Arial", 11)).pack(
            side=tk.LEFT, padx=20
        )

        # Row 3: Integrated positions
        row3 = tk.Frame(left_column)
        row3.pack(fill=tk.X, pady=2)
        self.pos_x_var = tk.StringVar(value="Position X: 0.000m")
        self.pos_y_var = tk.StringVar(value="Position Y: 0.000m")
        tk.Label(
            row3,
            textvariable=self.pos_x_var,
            font=("Arial", 11, "bold"),
            fg="darkgreen",
        ).pack(side=tk.LEFT, padx=20)
        tk.Label(
            row3,
            textvariable=self.pos_y_var,
            font=("Arial", 11, "bold"),
            fg="darkgreen",
        ).pack(side=tk.LEFT, padx=20)

        # Row 4: Control corrections
        row4 = tk.Frame(left_column)
        row4.pack(fill=tk.X, pady=2)
        self.corr_vx_var = tk.StringVar(value="Correction VX: 0.000")
        self.corr_vy_var = tk.StringVar(value="Correction VY: 0.000")
        tk.Label(
            row4, textvariable=self.corr_vx_var, font=("Arial", 11), fg="red"
        ).pack(side=tk.LEFT, padx=20)
        tk.Label(
            row4, textvariable=self.corr_vy_var, font=("Arial", 11), fg="red"
        ).pack(side=tk.LEFT, padx=20)

        # Control buttons row
        button_row = tk.Frame(left_column)
        button_row.pack(fill=tk.X, pady=10)

        self.apply_all_button = tk.Button(
            button_row,
            text="Apply All Values",
            command=self.apply_all_values,
            bg="green",
            fg="black",
            font=("Arial", 10, "bold"),
        )
        self.apply_all_button.pack(side=tk.LEFT, padx=5)

        self.reset_all_button = tk.Button(
            button_row,
            text="Reset to Default",
            command=self.reset_all_values,
            bg="orange",
            fg="black",
            font=("Arial", 10),
        )
        self.reset_all_button.pack(side=tk.LEFT, padx=5)

        self.clear_graphs_button = tk.Button(
            button_row,
            text="Clear Graphs",
            command=self.clear_graphs,
            bg="blue",
            fg="black",
            font=("Arial", 10),
        )
        self.clear_graphs_button.pack(side=tk.LEFT, padx=5)

        # Right column - Output window (only if enabled)
        if self.show_output_window:
            right_column = tk.Frame(main_container)
            right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

            # Output window label
            output_label = tk.Label(
                right_column, text="Output Log", font=("Arial", 10, "bold")
            )
            output_label.pack(anchor=tk.W, pady=(0, 5))

            # Output text widget with scrollbar
            output_frame = tk.Frame(right_column)
            output_frame.pack(fill=tk.BOTH, expand=True)

            self.output_text = tk.Text(
                output_frame,
                height=8,
                width=40,
                font=("Courier", 9),
                bg="white",
                fg="green",
                wrap=tk.WORD,
            )
            output_scrollbar = tk.Scrollbar(
                output_frame, command=self.output_text.yview
            )
            self.output_text.config(yscrollcommand=output_scrollbar.set)

            self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Output control buttons frame
            output_buttons_frame = tk.Frame(right_column)
            output_buttons_frame.pack(anchor=tk.W, pady=(5, 0))

            # Clear output button
            clear_output_button = tk.Button(
                output_buttons_frame,
                text="Clear Output",
                command=self.clear_output,
                bg="gray",
                fg="black",
                font=("Arial", 9),
            )
            clear_output_button.pack(side=tk.LEFT, padx=(0, 5))

            # Save output log button
            save_output_button = tk.Button(
                output_buttons_frame,
                text="Save",
                command=self.save_output_log,
                bg="blue",
                fg="white",
                font=("Arial", 9),
            )
            save_output_button.pack(side=tk.LEFT)
        else:
            # If output window is disabled, set output_text to None
            self.output_text = None

    # --- NEW FUNCTION: create_runtime_controls ---
    def create_runtime_controls(self, parent):
        """Create runtime adjustable parameter controls"""
        # Main frame for runtime controls
        runtime_main_frame = tk.Frame(parent)
        runtime_main_frame.pack(fill=tk.X, pady=2)

        # Left column
        left_col = tk.Frame(runtime_main_frame)
        left_col.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=(0, 5))

        # Right column
        right_col = tk.Frame(runtime_main_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, expand=True, padx=(5, 0))

        # Left column parameters
        # Target Height
        target_height_frame = tk.Frame(left_col)
        target_height_frame.pack(fill=tk.X, pady=2)
        tk.Label(target_height_frame, text="Target Height (m):", width=18).pack(
            side=tk.LEFT
        )
        self.target_height_var = tk.StringVar(value=str(TARGET_HEIGHT))
        self.target_height_entry = tk.Entry(
            target_height_frame, textvariable=self.target_height_var, width=8
        )
        self.target_height_entry.pack(side=tk.LEFT, padx=5)

        # Takeoff Time
        takeoff_time_frame = tk.Frame(left_col)
        takeoff_time_frame.pack(fill=tk.X, pady=2)
        tk.Label(takeoff_time_frame, text="Takeoff Time (s):", width=18).pack(
            side=tk.LEFT
        )
        self.takeoff_time_var = tk.StringVar(value=str(TAKEOFF_TIME))
        self.takeoff_time_entry = tk.Entry(
            takeoff_time_frame, textvariable=self.takeoff_time_var, width=8
        )
        self.takeoff_time_entry.pack(side=tk.LEFT, padx=5)

        # Hover Duration
        hover_duration_frame = tk.Frame(left_col)
        hover_duration_frame.pack(fill=tk.X, pady=2)
        tk.Label(hover_duration_frame, text="Hover Duration (s):", width=18).pack(
            side=tk.LEFT
        )
        self.hover_duration_var = tk.StringVar(value=str(HOVER_DURATION))
        self.hover_duration_entry = tk.Entry(
            hover_duration_frame, textvariable=self.hover_duration_var, width=8
        )
        self.hover_duration_entry.pack(side=tk.LEFT, padx=5)

        # Landing Time
        landing_time_frame = tk.Frame(left_col)
        landing_time_frame.pack(fill=tk.X, pady=2)
        tk.Label(landing_time_frame, text="Landing Time (s):", width=18).pack(
            side=tk.LEFT
        )
        self.landing_time_var = tk.StringVar(value=str(LANDING_TIME))
        self.landing_time_entry = tk.Entry(
            landing_time_frame, textvariable=self.landing_time_var, width=8
        )
        self.landing_time_entry.pack(side=tk.LEFT, padx=5)

        # Velocity Smoothing Alpha
        vel_smooth_alpha_frame = tk.Frame(left_col)
        vel_smooth_alpha_frame.pack(fill=tk.X, pady=2)
        tk.Label(vel_smooth_alpha_frame, text="Velocity Smoothing α:", width=18).pack(
            side=tk.LEFT
        )
        self.vel_smooth_alpha_var = tk.StringVar(value=str(VELOCITY_SMOOTHING_ALPHA))
        self.vel_smooth_alpha_entry = tk.Entry(
            vel_smooth_alpha_frame, textvariable=self.vel_smooth_alpha_var, width=8
        )
        self.vel_smooth_alpha_entry.pack(side=tk.LEFT, padx=5)

        # Right column parameters
        # Max Correction
        max_corr_frame = tk.Frame(right_col)
        max_corr_frame.pack(fill=tk.X, pady=2)
        tk.Label(max_corr_frame, text="Max Correction:", width=18).pack(side=tk.LEFT)
        self.max_corr_var = tk.StringVar(value=str(MAX_CORRECTION))
        self.max_corr_entry = tk.Entry(
            max_corr_frame, textvariable=self.max_corr_var, width=8
        )
        self.max_corr_entry.pack(side=tk.LEFT, padx=5)

        # Velocity Threshold
        vel_thresh_frame = tk.Frame(right_col)
        vel_thresh_frame.pack(fill=tk.X, pady=2)
        tk.Label(vel_thresh_frame, text="Velocity Threshold (m/s):", width=18).pack(
            side=tk.LEFT
        )
        self.vel_thresh_var = tk.StringVar(value=str(VELOCITY_THRESHOLD))
        self.vel_thresh_entry = tk.Entry(
            vel_thresh_frame, textvariable=self.vel_thresh_var, width=8
        )
        self.vel_thresh_entry.pack(side=tk.LEFT, padx=5)

        # Drift Compensation Rate
        drift_rate_frame = tk.Frame(right_col)
        drift_rate_frame.pack(fill=tk.X, pady=2)
        tk.Label(drift_rate_frame, text="Drift Compensation Rate:", width=18).pack(
            side=tk.LEFT
        )
        self.drift_rate_var = tk.StringVar(value=str(DRIFT_COMPENSATION_RATE))
        self.drift_rate_entry = tk.Entry(
            drift_rate_frame, textvariable=self.drift_rate_var, width=8
        )
        self.drift_rate_entry.pack(side=tk.LEFT, padx=5)

        # Reset Interval
        reset_int_frame = tk.Frame(right_col)
        reset_int_frame.pack(fill=tk.X, pady=2)
        tk.Label(reset_int_frame, text="Reset Interval (s):", width=18).pack(
            side=tk.LEFT
        )
        self.reset_int_var = tk.StringVar(value=str(PERIODIC_RESET_INTERVAL))
        self.reset_int_entry = tk.Entry(
            reset_int_frame, textvariable=self.reset_int_var, width=8
        )
        self.reset_int_entry.pack(side=tk.LEFT, padx=5)

        # Max Position Error
        max_pos_err_frame = tk.Frame(right_col)
        max_pos_err_frame.pack(fill=tk.X, pady=2)
        tk.Label(max_pos_err_frame, text="Max Position Error (m):", width=18).pack(
            side=tk.LEFT
        )
        self.max_pos_err_var = tk.StringVar(value=str(MAX_POSITION_ERROR))
        self.max_pos_err_entry = tk.Entry(
            max_pos_err_frame, textvariable=self.max_pos_err_var, width=8
        )
        self.max_pos_err_entry.pack(side=tk.LEFT, padx=5)

    # --- END NEW FUNCTION ---

    def toggle_blink(self):
        """Toggle NeoPixel LED blinking on/off"""
        # Use inline np_* helpers; reuse existing SyncCrazyflie when present
        try:
            if not self.blinking:
                # Start blinking: update stored color from UI, set it, then start blink.
                try:
                    r = int(self.rgb_r_var.get())
                    g = int(self.rgb_g_var.get())
                    b = int(self.rgb_b_var.get())
                except Exception:
                    r, g, b = self.neo_last_color

                # Clamp and store
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                self.neo_last_color = (r, g, b)

                cf = (
                    getattr(scf_instance, "cf", None)
                    if scf_instance is not None
                    else None
                )
                if cf is not None:
                    _try_send_with_retries(
                        cf, np_set_all, r, g, b, logger=self.log_to_output
                    )
                    _try_send_with_retries(
                        cf, np_start_blink, 500, 500, logger=self.log_to_output
                    )
                else:
                    tmp_cf = Crazyflie(rw_cache="./cache")
                    try:
                        with SyncCrazyflie(DRONE_URI, cf=tmp_cf) as scf:
                            tmp = getattr(scf, "cf", tmp_cf)
                            time.sleep(NP_LINK_SETUP_DELAY)
                            _try_send_with_retries(
                                tmp, np_set_all, r, g, b, logger=self.log_to_output
                            )
                            _try_send_with_retries(
                                tmp, np_start_blink, 500, 500, logger=self.log_to_output
                            )
                    except Exception as e:
                        print(f"NeoPixel blink start error: {e}")
                        raise

                self.blinking = True
                self.low_battery_blinking = False
                self.blink_button.config(text="Stop Blinking", bg="orange")
                self.status_var.set("Status: LEDs blinking...")
            else:
                # Stop blinking
                cf = (
                    getattr(scf_instance, "cf", None)
                    if scf_instance is not None
                    else None
                )

                # Stop blinking but restore last color (do not clear stored RGB)
                if cf is not None:
                    _try_send_with_retries(cf, np_stop_blink, logger=self.log_to_output)
                    # Restore last color so stop is non-destructive
                    _try_send_with_retries(
                        cf, np_set_all, *self.neo_last_color, logger=self.log_to_output
                    )
                else:
                    tmp_cf = Crazyflie(rw_cache="./cache")
                    try:
                        with SyncCrazyflie(DRONE_URI, cf=tmp_cf) as scf:
                            tmp = getattr(scf, "cf", tmp_cf)
                            time.sleep(NP_LINK_SETUP_DELAY)
                            _try_send_with_retries(
                                tmp, np_stop_blink, logger=self.log_to_output
                            )
                            _try_send_with_retries(
                                tmp,
                                np_set_all,
                                *self.neo_last_color,
                                logger=self.log_to_output,
                            )
                    except Exception:
                        pass

                self.blinking = False
                self.blink_button.config(text="Blink LEDs", bg="yellow")
                self.status_var.set("Status: LEDs stopped (color preserved)")
        except Exception as e:
            self.status_var.set(f"Status: NeoPixel error - {str(e)}")
            print(f"NeoPixel error: {e}")
            self.log_to_output(f"NeoPixel Error: {str(e)}")

    def toggle_debug_mode(self):
        """Toggle debug mode on/off"""
        global DEBUG_MODE
        DEBUG_MODE = self.enable_debug_mode_var.get()
        mode_text = "ENABLED" if DEBUG_MODE else "DISABLED"
        self.status_var.set(f"Status: Debug Mode {mode_text}")
        self.log_to_output(
            f"Debug Mode {mode_text} - {'Motor commands will be skipped' if DEBUG_MODE else 'Normal flight operations'}"
        )
        print(f"Debug Mode: {mode_text}")

    def set_static_mode(self):
        """Make the current LED color static (stop blinking but keep color)."""
        try:
            # Always set static color (independent of whether blinking was active).
            try:
                r = int(self.rgb_r_var.get())
                g = int(self.rgb_g_var.get())
                b = int(self.rgb_b_var.get())
            except Exception:
                r, g, b = self.neo_last_color

            # Clamp and store the color so future actions reuse it
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            self.neo_last_color = (r, g, b)

            cf = getattr(scf_instance, "cf", None) if scf_instance is not None else None
            if cf is not None:
                # Stop blink if active and set color
                if self.blinking:
                    _try_send_with_retries(cf, np_stop_blink, logger=self.log_to_output)
                _try_send_with_retries(
                    cf, np_set_all, r, g, b, logger=self.log_to_output
                )
            else:
                tmp_cf = Crazyflie(rw_cache="./cache")
                try:
                    with SyncCrazyflie(DRONE_URI, cf=tmp_cf) as scf:
                        tmp = getattr(scf, "cf", tmp_cf)
                        time.sleep(NP_LINK_SETUP_DELAY)
                        if self.blinking:
                            _try_send_with_retries(tmp, np_stop_blink)
                        _try_send_with_retries(tmp, np_set_all, r, g, b)
                except Exception as e:
                    print(f"NeoPixel error setting static: {e}")
                    return

            self.blinking = False
            self.low_battery_blinking = False
            self.blink_button.config(text="Blink LEDs", bg="yellow")
            self.status_var.set("Status: LEDs set to static mode")
        except Exception as e:
            self.status_var.set(f"Status: NeoPixel error - {str(e)}")
            print(f"NeoPixel error: {e}")
            self.log_to_output(f"NeoPixel Error: {str(e)}")

    def toggle_csv_logging(self):
        """Toggle CSV logging state"""
        global DRONE_CSV_LOGGING
        DRONE_CSV_LOGGING = self.enable_csv_logging_var.get()
        status = "ENABLED" if DRONE_CSV_LOGGING else "DISABLED"
        self.log_to_output(f"CSV Logging {status}")

    def set_leds_color(self, r, g, b):
        """Set a stable color on the NeoPixels (stops blinking first)."""
        try:
            # Update stored color so future actions reuse it
            try:
                r = int(r)
                g = int(g)
                b = int(b)
            except Exception:
                pass
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            self.neo_last_color = (r, g, b)

            # If currently blinking, stop it first; then set the color (use set_all)
            cf = getattr(scf_instance, "cf", None) if scf_instance is not None else None
            if cf is not None:
                if self.blinking:
                    _try_send_with_retries(cf, np_stop_blink, logger=self.log_to_output)
                _try_send_with_retries(
                    cf, np_set_all, r, g, b, logger=self.log_to_output
                )
                _try_send_with_retries(cf, np_show, logger=self.log_to_output)
            else:
                tmp_cf = Crazyflie(rw_cache="./cache")
                try:
                    with SyncCrazyflie(DRONE_URI, cf=tmp_cf) as scf:
                        tmp = getattr(scf, "cf", tmp_cf)
                        time.sleep(NP_LINK_SETUP_DELAY)
                        if self.blinking:
                            _try_send_with_retries(
                                tmp, np_stop_blink, logger=self.log_to_output
                            )
                        _try_send_with_retries(
                            tmp, np_set_all, r, g, b, logger=self.log_to_output
                        )
                        _try_send_with_retries(tmp, np_show, logger=self.log_to_output)
                except Exception as e:
                    self.status_var.set(f"Status: NeoPixel error - {str(e)}")
                    print(f"NeoPixel error: {e}")
                    return

            self.blinking = False
            self.low_battery_blinking = False
            self.blink_button.config(text="Blink LEDs", bg="yellow")

            self.status_var.set("Status: LEDs set to color")
        except Exception as e:
            self.status_var.set(f"Status: NeoPixel error - {str(e)}")
            print(f"NeoPixel set color error: {e}")
            self.log_to_output(f"NeoPixel Error: {str(e)}")

    def set_color_from_ui(self):
        """Read R,G,B from UI controls and set LEDs accordingly."""
        try:
            r = int(self.rgb_r_var.get())
            g = int(self.rgb_g_var.get())
            b = int(self.rgb_b_var.get())
        except Exception:
            self.status_var.set("Status: Invalid RGB values")
            return

        # Clamp values
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        self.set_leds_color(r, g, b)

    def clear_leds(self):
        """Clear all NeoPixels (set to off)."""
        try:
            cf = getattr(scf_instance, "cf", None) if scf_instance is not None else None
            if cf is not None:
                _try_send_with_retries(cf, np_stop_blink, logger=self.log_to_output)
                _try_send_with_retries(cf, np_clear, logger=self.log_to_output)
            else:
                tmp_cf = Crazyflie(rw_cache="./cache")
                try:
                    with SyncCrazyflie(DRONE_URI, cf=tmp_cf) as scf:
                        tmp = getattr(scf, "cf", tmp_cf)
                        time.sleep(NP_LINK_SETUP_DELAY)
                        _try_send_with_retries(
                            tmp, np_stop_blink, logger=self.log_to_output
                        )
                        _try_send_with_retries(tmp, np_clear, logger=self.log_to_output)
                except Exception:
                    pass

            # Persist the cleared state as the last known color
            self.neo_last_color = (0, 0, 0)
            self.blinking = False
            self.low_battery_blinking = False
            self.blink_button.config(text="Blink LEDs", bg="yellow")
            self.status_var.set("Status: LEDs cleared")
        except Exception as e:
            self.status_var.set(f"Status: NeoPixel error - {str(e)}")
            print(f"NeoPixel clear error: {e}")
            self.log_to_output(f"NeoPixel Error: {str(e)}")

    def low_battery_blink_start(self):
        """Start blinking red LEDs for low battery alert"""
        if not self.blinking:
            try:
                cf = (
                    getattr(scf_instance, "cf", None)
                    if scf_instance is not None
                    else None
                )
                if cf is not None:
                    _try_send_with_retries(
                        cf, np_set_all, 255, 0, 0, logger=self.log_to_output
                    )
                    _try_send_with_retries(
                        cf, np_start_blink, 500, 500, logger=self.log_to_output
                    )
                else:
                    tmp_cf = Crazyflie(rw_cache="./cache")
                    try:
                        with SyncCrazyflie(DRONE_URI, cf=tmp_cf) as scf:
                            tmp = getattr(scf, "cf", tmp_cf)
                            time.sleep(NP_LINK_SETUP_DELAY)
                            _try_send_with_retries(
                                tmp, np_set_all, 255, 0, 0, logger=self.log_to_output
                            )
                            _try_send_with_retries(
                                tmp, np_start_blink, 500, 500, logger=self.log_to_output
                            )
                    except Exception as e:
                        print(f"Low battery blink start error: {e}")
                        return
                self.blinking = True
                self.low_battery_blinking = True
                self.status_var.set("Status: Low battery - LEDs blinking red")
            except Exception as e:
                self.status_var.set(f"Status: NeoPixel error - {str(e)}")
                print(f"NeoPixel low battery blink error: {e}")
                self.log_to_output(f"NeoPixel Error: {str(e)}")

    def low_battery_blink_stop(self):
        """Stop low battery blinking and clear LEDs"""
        if self.blinking and self.low_battery_blinking:
            try:
                cf = (
                    getattr(scf_instance, "cf", None)
                    if scf_instance is not None
                    else None
                )
                if cf is not None:
                    _try_send_with_retries(cf, np_stop_blink, logger=self.log_to_output)
                    _try_send_with_retries(cf, np_clear, logger=self.log_to_output)
                else:
                    tmp_cf = Crazyflie(rw_cache="./cache")
                    try:
                        with SyncCrazyflie(DRONE_URI, cf=tmp_cf) as scf:
                            tmp = getattr(scf, "cf", tmp_cf)
                            time.sleep(NP_LINK_SETUP_DELAY)
                            _try_send_with_retries(
                                tmp, np_stop_blink, logger=self.log_to_output
                            )
                            _try_send_with_retries(
                                tmp, np_clear, logger=self.log_to_output
                            )
                    except Exception:
                        pass
                self.blinking = False
                self.low_battery_blinking = False
                self.status_var.set("Status: Battery OK - LEDs cleared")
            except Exception as e:
                self.status_var.set(f"Status: NeoPixel error - {str(e)}")
                print(f"NeoPixel low battery stop error: {e}")
                self.log_to_output(f"NeoPixel Error: {str(e)}")

    def create_pid_controls(self, parent):
        """Create PID tuning input controls with TRIM controls - compact layout"""
        # Main control frame - horizontal layout for PID and TRIM sections
        main_control_frame = tk.Frame(parent)
        main_control_frame.pack(fill=tk.X, pady=2)

        # Left side - PID Controls
        pid_frame = tk.LabelFrame(
            main_control_frame, text="PID Controls", padx=5, pady=5
        )
        pid_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Position PID Controls (horizontal layout)
        pos_frame = tk.LabelFrame(pid_frame, text="Position PID", padx=5, pady=2)
        pos_frame.pack(fill=tk.X, pady=2)
        pos_row = tk.Frame(pos_frame)
        pos_row.pack(fill=tk.X)
        # Position Kp
        tk.Label(pos_row, text="Kp:", width=3).pack(side=tk.LEFT)
        self.pos_kp_var = tk.StringVar(value=str(POSITION_KP))
        self.pos_kp_entry = tk.Entry(pos_row, textvariable=self.pos_kp_var, width=6)
        self.pos_kp_entry.pack(side=tk.LEFT, padx=2)
        # Position Ki
        tk.Label(pos_row, text="Ki:", width=3).pack(side=tk.LEFT, padx=(5, 0))
        self.pos_ki_var = tk.StringVar(value=str(POSITION_KI))
        self.pos_ki_entry = tk.Entry(pos_row, textvariable=self.pos_ki_var, width=6)
        self.pos_ki_entry.pack(side=tk.LEFT, padx=2)
        # Position Kd
        tk.Label(pos_row, text="Kd:", width=3).pack(side=tk.LEFT, padx=(5, 0))
        self.pos_kd_var = tk.StringVar(value=str(POSITION_KD))
        self.pos_kd_entry = tk.Entry(pos_row, textvariable=self.pos_kd_var, width=6)
        self.pos_kd_entry.pack(side=tk.LEFT, padx=2)

        # Velocity PID Controls (horizontal layout)
        vel_frame = tk.LabelFrame(pid_frame, text="Velocity PID", padx=5, pady=2)
        vel_frame.pack(fill=tk.X, pady=2)
        vel_row = tk.Frame(vel_frame)
        vel_row.pack(fill=tk.X)
        # Velocity Kp
        tk.Label(vel_row, text="Kp:", width=3).pack(side=tk.LEFT)
        self.vel_kp_var = tk.StringVar(value=str(VELOCITY_KP))
        self.vel_kp_entry = tk.Entry(vel_row, textvariable=self.vel_kp_var, width=6)
        self.vel_kp_entry.pack(side=tk.LEFT, padx=2)
        # Velocity Ki
        tk.Label(vel_row, text="Ki:", width=3).pack(side=tk.LEFT, padx=(5, 0))
        self.vel_ki_var = tk.StringVar(value=str(VELOCITY_KI))
        self.vel_ki_entry = tk.Entry(vel_row, textvariable=self.vel_ki_var, width=6)
        self.vel_ki_entry.pack(side=tk.LEFT, padx=2)
        # Velocity Kd
        tk.Label(vel_row, text="Kd:", width=3).pack(side=tk.LEFT, padx=(5, 0))
        self.vel_kd_var = tk.StringVar(value=str(VELOCITY_KD))
        self.vel_kd_entry = tk.Entry(vel_row, textvariable=self.vel_kd_var, width=6)
        self.vel_kd_entry.pack(side=tk.LEFT, padx=2)

        # Right side - TRIM Controls
        trim_frame = tk.LabelFrame(
            main_control_frame, text="TRIM Controls", padx=5, pady=5
        )
        trim_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))

        # TRIM VX Control
        trim_vx_frame = tk.Frame(trim_frame)
        trim_vx_frame.pack(fill=tk.X, pady=5)
        tk.Label(trim_vx_frame, text="TRIM VX:", width=10).pack(side=tk.LEFT)
        self.trim_vx_var = tk.StringVar(value=str(TRIM_VX))
        self.trim_vx_entry = tk.Entry(
            trim_vx_frame, textvariable=self.trim_vx_var, width=8
        )
        self.trim_vx_entry.pack(side=tk.LEFT, padx=2)

        # TRIM VY Control
        trim_vy_frame = tk.Frame(trim_frame)
        trim_vy_frame.pack(fill=tk.X, pady=5)
        tk.Label(trim_vy_frame, text="TRIM VY:", width=10).pack(side=tk.LEFT)
        self.trim_vy_var = tk.StringVar(value=str(TRIM_VY))
        self.trim_vy_entry = tk.Entry(
            trim_vy_frame, textvariable=self.trim_vy_var, width=8
        )
        self.trim_vy_entry.pack(side=tk.LEFT, padx=2)

        # Helper labels for TRIM values
        trim_help_frame = tk.Frame(trim_frame)
        trim_help_frame.pack(fill=tk.X, pady=2)
        tk.Label(
            trim_help_frame,
            text="(Forward/Back, Left/Right)",
            font=("Arial", 8),
            fg="gray",
        ).pack()

        # Optical Flow Scaling Controls
        scale_frame = tk.LabelFrame(
            main_control_frame, text="Optical Flow Scaling", padx=5, pady=5
        )
        scale_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))

        # Scaling factor control
        scale_factor_frame = tk.Frame(scale_frame)
        scale_factor_frame.pack(fill=tk.X, pady=5)
        tk.Label(scale_factor_frame, text="Scale Factor:", width=12).pack(side=tk.LEFT)
        self.scale_factor_var = tk.StringVar(value=str(OPTICAL_FLOW_SCALE))
        self.scale_factor_entry = tk.Entry(
            scale_factor_frame, textvariable=self.scale_factor_var, width=8
        )
        self.scale_factor_entry.pack(side=tk.LEFT, padx=2)

        # Height scaling checkbox
        height_scale_frame = tk.Frame(scale_frame)
        height_scale_frame.pack(fill=tk.X, pady=5)
        self.height_scaling_var = tk.BooleanVar(value=USE_HEIGHT_SCALING)
        self.height_scaling_check = tk.Checkbutton(
            height_scale_frame,
            text="Use Height Scaling",
            variable=self.height_scaling_var,
        )
        self.height_scaling_check.pack()

        # Helper text for scaling
        scale_help_frame = tk.Frame(scale_frame)
        scale_help_frame.pack(fill=tk.X, pady=2)
        tk.Label(
            scale_help_frame,
            text="(Increase if trajectory too small)",
            font=("Arial", 8),
            fg="gray",
        ).pack()

    def setup_plots(self):
        """Setup matplotlib plots"""
        # Create 2x2 subplot layout
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Velocities
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Integrated Position (2D plot)
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Control Corrections
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Height

        # Velocities plot
        self.ax1.set_title("Velocities (m/s)", fontsize=12)
        self.ax1.set_ylabel("Velocity (m/s)")
        self.ax1.grid(True, alpha=0.3)
        (self.line_vx,) = self.ax1.plot([], [], "b-", linewidth=2, label="VX")
        (self.line_vy,) = self.ax1.plot([], [], "r-", linewidth=2, label="VY")
        self.ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)

        # 2D Position plot
        self.ax2.set_title("Integrated Position", fontsize=12)
        self.ax2.set_xlabel("X Position (m)")
        self.ax2.set_ylabel("Y Position (m)")
        self.ax2.set_aspect("equal")
        self.ax2.grid(True, alpha=0.3)
        (self.line_pos,) = self.ax2.plot(
            [], [], "purple", linewidth=2, alpha=0.7, label="Trajectory"
        )
        (self.current_pos,) = self.ax2.plot([], [], "ro", markersize=8, label="Current")
        self.ax2.plot(
            0,
            0,
            "ko",
            markersize=10,
            markerfacecolor="yellow",
            markeredgecolor="black",
            label="Origin",
        )
        (self.release_points_scatter,) = self.ax2.plot(
            [],
            [],
            "x",
            markersize=10,
            markeredgewidth=2,
            color="orange",
            label="Release Pts",
        )
        # Move legend outside to the right to prevent shadowing the trajectory
        self.ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=9)

        # Control corrections plot
        # self.ax3.set_title("Control Corrections", fontsize=12)
        self.ax3.text(
            0.5,
            0.95,
            "Control Corrections",
            transform=self.ax3.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )
        self.ax3.set_ylabel("Correction")
        self.ax3.grid(True, alpha=0.3)
        (self.line_corr_vx,) = self.ax3.plot([], [], "g-", linewidth=2, label="Corr VX")
        (self.line_corr_vy,) = self.ax3.plot([], [], "m-", linewidth=2, label="Corr VY")
        self.ax3.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)

        # Height plot
        # self.ax4.set_title("Height", fontsize=12)
        self.ax4.text(
            0.5,
            0.95,
            "Height",
            transform=self.ax4.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )
        self.ax4.set_xlabel("Time (s)")
        self.ax4.set_ylabel("Height (m)")
        self.ax4.grid(True, alpha=0.3)
        (self.line_height,) = self.ax4.plot(
            [], [], "orange", linewidth=2, label="Height"
        )
        (self.line_range,) = self.ax4.plot(
            [], [], "c--", linewidth=1, alpha=0.8, label="Range (ToF)"
        )
        self.ax4.axhline(
            y=TARGET_HEIGHT, color="red", linestyle="--", alpha=0.7, label="Target"
        )
        self.ax4.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)

        self.fig.tight_layout()
        # Increased right margin and horizontal spacing for legends
        self.fig.subplots_adjust(left=0.07, right=0.88, top=0.94, bottom=0.06, wspace=0.45, hspace=0.3)


    def update_plots(self, frame):
        """Update all plots with new data (highly optimized and thread-safe)"""
        if not time_history:
            return []

        # Local snapshots for thread-safe plotting
        with data_lock:
            # Convert deques to numpy arrays for faster processing
            t_hist = np.array(time_history)
            vx_hist = np.array(velocity_x_history_plot)
            vy_hist = np.array(velocity_y_history_plot)
            h_hist = np.array(height_history)
            r_hist = np.array(range_height_history)
            corr_vx_hist = np.array(correction_vx_history)
            corr_vy_hist = np.array(correction_vy_history)
            
            # Trajectory downsampling for long flights
            total_traj_pts = len(complete_trajectory_x)
            if total_traj_pts > MAX_PLOT_TRAJECTORY_POINTS:
                step = total_traj_pts // MAX_PLOT_TRAJECTORY_POINTS
                traj_x = np.array(complete_trajectory_x[::step])
                traj_y = np.array(complete_trajectory_y[::step])
            else:
                traj_x = np.array(complete_trajectory_x)
                traj_y = np.array(complete_trajectory_y)
            
            # Current values for UI
            cur_h = current_height
            cur_range = current_range_height
            cur_vx = current_vx
            cur_vy = current_vy
            cur_pos_x = integrated_position_x
            cur_pos_y = integrated_position_y
            cur_corr_vx = current_correction_vx
            cur_corr_vy = current_correction_vy
            
            # Key release points snapshot
            rel_points = list(key_release_points)

        # Check for stale sensor data (visual only)
        is_stale = False
        if not DEBUG_MODE and sensor_data_ready:
            if time.time() - last_sensor_heartbeat > 1.0:
                is_stale = True

        # Update real-time value displays
        stale_msg = " (STALE!)" if is_stale else ""
        self.height_var.set(f"Height: {cur_h:.3f}m{stale_msg}")
        self.range_var.set(f"Range: {cur_range:.3f}m{stale_msg}")
        self.phase_var.set(f"Phase: {flight_phase}")
        
        if current_battery_voltage > 0:
            status = " (LOW!)" if current_battery_voltage < LOW_BATTERY_THRESHOLD else ""
            self.battery_var.set(f"Battery: {current_battery_voltage:.2f}V{status}")
        else:
            self.battery_var.set("Battery: N/A")
            
        self.vx_var.set(f"VX: {cur_vx:.3f} m/s{stale_msg}")
        self.vy_var.set(f"VY: {cur_vy:.3f} m/s{stale_msg}")
        self.pos_x_var.set(f"Position X: {cur_pos_x:.3f}m{stale_msg}")
        self.pos_y_var.set(f"Position Y: {cur_pos_y:.3f}m{stale_msg}")
        self.corr_vx_var.set(f"Correction VX: {cur_corr_vx:.3f}")
        self.corr_vy_var.set(f"Correction VY: {cur_corr_vy:.3f}")

        # Low battery blinking (visual only)
        if 0 < current_battery_voltage <= 3.3:
            if not self.blinking: self.low_battery_blink_start()
        elif current_battery_voltage > 3.3 and self.low_battery_blinking:
            self.low_battery_blink_stop()

        # Update plots
        try:
            # 1. Velocities
            self.line_vx.set_data(t_hist, vx_hist)
            self.line_vy.set_data(t_hist, vy_hist)

            # 2. 2D Position
            if len(traj_x) > 0:
                self.line_pos.set_data(-traj_x, traj_y)
                self.current_pos.set_data([-cur_pos_x], [cur_pos_y])
                
                # Update release points scatter
                if rel_points:
                    rx = [-p[0] for p in rel_points]
                    ry = [p[1] for p in rel_points]
                    self.release_points_scatter.set_data(rx, ry)
                else:
                    self.release_points_scatter.set_data([], [])

            # 3. Control corrections
            self.line_corr_vx.set_data(t_hist, corr_vx_hist)
            self.line_corr_vy.set_data(t_hist, corr_vy_hist)

            # 4. Height
            self.line_height.set_data(t_hist, h_hist)
            self.line_range.set_data(t_hist, r_hist)

            # --- Smart Axis Scaling ---
            if len(t_hist) > 1:
                t_min, t_max = t_hist[0], t_hist[-1]
                
                # Time axis (X)
                cur_xmin, cur_xmax = self.ax1.get_xlim()
                if t_max > cur_xmax - (cur_xmax-cur_xmin)*0.1 or t_min < cur_xmin:
                    margin = (t_max - t_min) * 0.1
                    for ax in [self.ax1, self.ax3, self.ax4]:
                        ax.set_xlim(t_min, t_max + margin)

                # Velocity Y-axis
                v_all = np.concatenate([vx_hist, vy_hist])
                v_min, v_max = v_all.min(), v_all.max()
                v_margin = max((v_max - v_min) * 0.1, 0.05)
                self.ax1.set_ylim(v_min - v_margin, v_max + v_margin)

                # Position plot (Centered on trajectory)
                if len(traj_x) > 0:
                    px_min, px_max = -traj_x.max(), -traj_x.min()
                    py_min, py_max = traj_y.min(), traj_y.max()
                    range_x, range_y = px_max - px_min, py_max - py_min
                    max_r = max(range_x, range_y, 0.1)
                    cx, cy = (px_min + px_max) / 2, (py_min + py_max) / 2
                    m = max_r * 0.7
                    self.ax2.set_xlim(cx - m, cx + m)
                    self.ax2.set_ylim(cy - m, cy + m)

                # Corrections Y-axis
                c_all = np.concatenate([corr_vx_hist, corr_vy_hist])
                c_min, c_max = c_all.min(), c_all.max()
                c_margin = max((c_max - c_min) * 0.1, 0.01)
                self.ax3.set_ylim(c_min - c_margin, c_max + c_margin)

                # Height Y-axis
                h_all = np.concatenate([h_hist, r_hist])
                h_min, h_max = h_all.min(), h_all.max()
                h_margin = max((h_max - h_min) * 0.1, 0.05)
                self.ax4.set_ylim(h_min - h_margin, h_max + h_margin)

        except Exception as e:
            pass # Plotting error (usually frame sync)

        return []

    # --- NEW FUNCTION: apply_runtime_values ---
    def apply_runtime_values(self):
        """Apply runtime adjustable values from GUI inputs"""
        global TARGET_HEIGHT, TAKEOFF_TIME, HOVER_DURATION, LANDING_TIME, VELOCITY_SMOOTHING_ALPHA, MAX_CORRECTION
        global VELOCITY_THRESHOLD, DRIFT_COMPENSATION_RATE, PERIODIC_RESET_INTERVAL, MAX_POSITION_ERROR

        try:
            # Get values from GUI
            new_target_height = float(self.target_height_var.get())
            new_takeoff_time = float(self.takeoff_time_var.get())
            new_hover_duration = float(self.hover_duration_var.get())
            new_landing_time = float(self.landing_time_var.get())
            new_vel_smooth_alpha = float(self.vel_smooth_alpha_var.get())
            new_max_corr = float(self.max_corr_var.get())
            new_vel_thresh = float(self.vel_thresh_var.get())
            new_drift_rate = float(self.drift_rate_var.get())
            new_reset_int = float(self.reset_int_var.get())
            new_max_pos_err = float(self.max_pos_err_var.get())

            # Validate values (optional, add checks as needed)
            if new_takeoff_time < 0 or new_hover_duration < 0 or new_landing_time < 0:
                raise ValueError("Time values cannot be negative.")
            if new_vel_smooth_alpha < 0 or new_vel_smooth_alpha > 1.0:
                raise ValueError("Smoothing alpha must be between 0 and 1.")
            if new_max_corr < 0:
                raise ValueError("Max correction cannot be negative.")
            if new_vel_thresh < 0:
                raise ValueError("Velocity threshold cannot be negative.")
            if new_drift_rate < 0:
                raise ValueError("Drift compensation rate cannot be negative.")
            if new_reset_int <= 0:
                raise ValueError("Reset interval must be positive.")
            if new_max_pos_err <= 0:
                raise ValueError("Max position error must be positive.")

            # Apply values to global variables
            TARGET_HEIGHT = new_target_height
            TAKEOFF_TIME = new_takeoff_time
            HOVER_DURATION = new_hover_duration
            LANDING_TIME = new_landing_time
            VELOCITY_SMOOTHING_ALPHA = new_vel_smooth_alpha
            MAX_CORRECTION = new_max_corr
            VELOCITY_THRESHOLD = new_vel_thresh
            DRIFT_COMPENSATION_RATE = new_drift_rate
            PERIODIC_RESET_INTERVAL = new_reset_int
            MAX_POSITION_ERROR = new_max_pos_err

            self.log_to_output(f"Runtime Values Applied:")
            self.log_to_output(f"  Target Height: {TARGET_HEIGHT}")
            self.log_to_output(f"  Takeoff Time: {TAKEOFF_TIME}")
            self.log_to_output(f"  Hover Duration: {HOVER_DURATION}")
            self.log_to_output(f"  Landing Time: {LANDING_TIME}")
            self.log_to_output(
                f"  Velocity Smoothing Alpha: {VELOCITY_SMOOTHING_ALPHA}"
            )
            self.log_to_output(f"  Max Correction: {MAX_CORRECTION}")
            self.log_to_output(f"  Velocity Threshold: {VELOCITY_THRESHOLD}")
            self.log_to_output(f"  Drift Compensation Rate: {DRIFT_COMPENSATION_RATE}")
            self.log_to_output(f"  Reset Interval: {PERIODIC_RESET_INTERVAL}")
            self.log_to_output(f"  Max Position Error: {MAX_POSITION_ERROR}")


            # Log to output window
            self.log_to_output("Runtime values applied successfully")

        except ValueError as e:
            self.log_to_output(f"Error applying runtime values: {e}")
            self.log_to_output("Please enter valid numbers for all runtime parameters.")

    # --- END NEW FUNCTION ---

    def apply_pid_values(self):
        """Apply PID values from GUI inputs"""
        global POSITION_KP, POSITION_KI, POSITION_KD, VELOCITY_KP, VELOCITY_KI, VELOCITY_KD
        try:
            # Get values from GUI
            POSITION_KP = float(self.pos_kp_var.get())
            POSITION_KI = float(self.pos_ki_var.get())
            POSITION_KD = float(self.pos_kd_var.get())
            VELOCITY_KP = float(self.vel_kp_var.get())
            VELOCITY_KI = float(self.vel_ki_var.get())
            VELOCITY_KD = float(self.vel_kd_var.get())

            # Reset PID state when applying new values
            global position_integral_x, position_integral_y, velocity_integral_x, velocity_integral_y
            global last_position_error_x, last_position_error_y, last_velocity_error_x, last_velocity_error_y
            position_integral_x = 0.0
            position_integral_y = 0.0
            velocity_integral_x = 0.0
            velocity_integral_y = 0.0
            last_position_error_x = 0.0
            last_position_error_y = 0.0
            last_velocity_error_x = 0.0
            last_velocity_error_y = 0.0

            self.log_to_output(f"PID Values Applied:")
            self.log_to_output(
                f"Position: Kp={POSITION_KP}, Ki={POSITION_KI}, Kd={POSITION_KD}"
            )
            self.log_to_output(
                f"Velocity: Kp={VELOCITY_KP}, Ki={VELOCITY_KI}, Kd={VELOCITY_KD}"
            )

            # Log to output window
            self.log_to_output("PID values applied successfully")
        except ValueError as e:
            self.log_to_output(f"Error applying PID values: {e}")
            self.log_to_output("Please enter valid numbers")

    def reset_pid_values(self):
        """Reset PID values to default"""
        # Default values (from your current settings)
        self.pos_kp_var.set("1.2")
        self.pos_ki_var.set("0.00")
        self.pos_kd_var.set("0.0")
        self.vel_kp_var.set("1.2")
        self.vel_ki_var.set("0.0")
        self.vel_kd_var.set("0.0")
        # Apply the reset values
        self.apply_pid_values()
        self.log_to_output("PID values reset to default")

    def apply_all_values(self):
        """Apply PID, TRIM, Optical Flow scaling, and Runtime values from GUI inputs"""
        # First apply runtime values
        self.apply_runtime_values()
        # Then apply PID values
        self.apply_pid_values()
        # Then apply TRIM values
        global TRIM_VX, TRIM_VY, OPTICAL_FLOW_SCALE, USE_HEIGHT_SCALING
        try:
            TRIM_VX = float(self.trim_vx_var.get())
            TRIM_VY = float(self.trim_vy_var.get())
            self.log_to_output(f"TRIM Values Applied: VX={TRIM_VX}, VY={TRIM_VY}")
        except ValueError as e:
            self.log_to_output(f"Error applying TRIM values: {e}")
            self.log_to_output("Please enter valid numbers for TRIM values")

        # Apply Optical Flow scaling values
        try:
            OPTICAL_FLOW_SCALE = float(self.scale_factor_var.get())
            USE_HEIGHT_SCALING = self.height_scaling_var.get()
            self.log_to_output(
                f"Optical Flow Scaling Applied: Scale={OPTICAL_FLOW_SCALE}, Height Scaling={USE_HEIGHT_SCALING}"
            )
        except ValueError as e:
            self.log_to_output(f"Error applying Optical Flow scaling: {e}")
            self.log_to_output("Please enter valid numbers for scaling factor")

        # Apply Joystick sensitivity
        try:
            global JOYSTICK_SENSITIVITY
            JOYSTICK_SENSITIVITY = float(self.joystick_sensitivity_var.get())
            if JOYSTICK_SENSITIVITY < 0.1 or JOYSTICK_SENSITIVITY > 2.0:
                raise ValueError("Sensitivity must be between 0.1 and 2.0")
            self.log_to_output(f"Joystick Sensitivity Applied: {JOYSTICK_SENSITIVITY}")
        except ValueError as e:
            self.log_to_output(f"Error applying joystick sensitivity: {e}")
            self.log_to_output("Please enter a value between 0.1 and 2.0")

    def reset_all_values(self):
        """Reset PID, TRIM, and Optical Flow scaling values to default"""
        # Reset PID values
        self.reset_pid_values()
        # Reset TRIM values to current defaults
        self.trim_vx_var.set("0.1")
        self.trim_vy_var.set("-0.02")
        # Reset Optical Flow scaling values
        self.scale_factor_var.set("3.7")
        self.height_scaling_var.set(False)
        # Reset Runtime values to defaults
        self.target_height_var.set("0.2")
        self.takeoff_time_var.set("1.0")
        self.hover_duration_var.set("30.0")
        self.landing_time_var.set("0.5")
        self.vel_smooth_alpha_var.set("0.8")
        self.max_corr_var.set("0.1")
        self.vel_thresh_var.set("0.005")
        self.drift_rate_var.set("0.003")
        self.reset_int_var.set("30.0")
        self.max_pos_err_var.set("2.0")
        # Reset Joystick sensitivity
        self.joystick_sensitivity_var.set("0.5")
        # Apply all values
        global TRIM_VX, TRIM_VY, OPTICAL_FLOW_SCALE, USE_HEIGHT_SCALING
        global TARGET_HEIGHT, TAKEOFF_TIME, HOVER_DURATION, LANDING_TIME, VELOCITY_SMOOTHING_ALPHA, MAX_CORRECTION
        global VELOCITY_THRESHOLD, DRIFT_COMPENSATION_RATE, PERIODIC_RESET_INTERVAL, MAX_POSITION_ERROR

        global JOYSTICK_SENSITIVITY

        TRIM_VX = 0.1
        TRIM_VY = -0.02
        OPTICAL_FLOW_SCALE = 3.7
        USE_HEIGHT_SCALING = False
        TARGET_HEIGHT = 0.2
        TAKEOFF_TIME = 1.0
        HOVER_DURATION = 30.0
        LANDING_TIME = 0.5
        VELOCITY_SMOOTHING_ALPHA = 0.8
        MAX_CORRECTION = 0.1
        VELOCITY_THRESHOLD = 0.005
        DRIFT_COMPENSATION_RATE = 0.003
        PERIODIC_RESET_INTERVAL = 30.0
        MAX_POSITION_ERROR = 2.0
        JOYSTICK_SENSITIVITY = 0.5
        self.log_to_output("All values reset to default")

    def toggle_output_window(self):
        """Toggle the output window visibility"""
        self.show_output_window = self.show_output_window_var.get()
        # Show message that change will take effect on restart
        self.status_var.set("Status: Output window setting changed - restart required")
        self.log_to_output(
            "Output window visibility changed - restart the application to apply"
        )

    def clear_output(self):
        """Clear the output log window"""
        if hasattr(self, "output_text") and self.output_text is not None:
            self.output_text.delete(1.0, tk.END)
            self.log_to_output("Output window cleared")

    def save_output_log(self):
        """Save the output log to a file"""
        if hasattr(self, "output_text") and self.output_text is not None:
            try:
                # Get all text from the output window
                log_content = self.output_text.get(1.0, tk.END).strip()

                if not log_content:
                    self.status_var.set("Status: No log content to save")
                    return

                # Create filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"drone_output_log_{timestamp}.txt"

                # Save to file
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(log_content)

                self.status_var.set(f"Status: Log saved to {filename}")
                self.log_to_output(f"Output log saved to: {filename}")

            except Exception as e:
                error_msg = f"Failed to save log: {str(e)}"
                self.status_var.set(f"Status: {error_msg}")
                self.log_to_output(error_msg)
        else:
            self.status_var.set("Status: Output window not available")

    def log_to_output(self, message):
        """Log a message to the output window (Thread-Safe)"""
        # Check if we are on the main thread
        if threading.current_thread() is threading.main_thread():
            self._log_internal(message)
        else:
            # Schedule update on main thread
            self.root.after(0, lambda: self._log_internal(message))

    def _log_internal(self, message):
        """Internal logging method intended to run on main thread"""
        if hasattr(self, "output_text") and self.output_text is not None:
            timestamp = time.strftime("%H:%M:%S")
            try:
                self.output_text.insert(tk.END, f"[{timestamp}] {message}\n")
                self.output_text.see(tk.END)  # Auto-scroll to bottom
            except Exception:
                pass  # Ignore tkinter errors if window is destroying

    def update_status(self, message):
        """Thread-safe status update"""
        self.root.after(0, lambda: self.status_var.set(message))

    def update_button(self, button, **kwargs):
        """Thread-safe button configuration update"""
        self.root.after(0, lambda: button.config(**kwargs))

    def safe_set_var(self, variable, value):
        """Thread-safe variable set"""
        self.root.after(0, lambda: variable.set(value))

    def clear_graphs(self):
        """Clear all graph data and reset plotting"""
        global time_history, velocity_x_history_plot, velocity_y_history_plot
        global position_x_history, position_y_history, correction_vx_history, correction_vy_history
        global height_history, complete_trajectory_x, complete_trajectory_y, start_time

        # Clear all history arrays
        time_history.clear()
        velocity_x_history_plot.clear()
        velocity_y_history_plot.clear()
        position_x_history.clear()
        position_y_history.clear()
        correction_vx_history.clear()
        correction_vy_history.clear()
        height_history.clear()
        range_height_history.clear()
        complete_trajectory_x.clear()
        complete_trajectory_x.clear()
        complete_trajectory_y.clear()
        key_release_points.clear()

        # Reset start time
        start_time = None

        # Clear plot lines
        if hasattr(self, "line_vx"):
            self.line_vx.set_data([], [])
        if hasattr(self, "line_vy"):
            self.line_vy.set_data([], [])
        if hasattr(self, "line_pos"):
            self.line_pos.set_data([], [])
        if hasattr(self, "release_points_scatter"):
            self.release_points_scatter.set_data([], [])
        if hasattr(self, "line_corr_vx"):
            self.line_corr_vx.set_data([], [])
        if hasattr(self, "line_corr_vy"):
            self.line_corr_vy.set_data([], [])
        if hasattr(self, "line_height"):
            self.line_height.set_data([], [])
        if hasattr(self, "line_range"):
            self.line_range.set_data([], [])

        # Redraw the plots
        if hasattr(self, "canvas"):
            self.canvas.draw()

        self.log_to_output("Graphs cleared")

    def start_continuous_movement(self, key):
        """Start continuous movement for GUI buttons"""
        if not self.joystick_active:
            return

        key = key.lower()
        if key in ["w", "a", "s", "d"]:
            # Set joystick key for control thread
            self.joystick_keys[key] = True

            # Update status
            active_keys = [k.upper() for k, v in self.joystick_keys.items() if v]
            self.joystick_status_var.set(f"Joystick: ACTIVE ({','.join(active_keys)})")

            # Log button press
            self.log_to_output(
                f"Continuous movement: {self._key_to_direction(key)} started"
            )

    def stop_continuous_movement(self, key):
        """Stop continuous movement for GUI buttons"""
        if not self.joystick_active:
            return

        key = key.lower()
        if key in ["w", "a", "s", "d"]:
            # Clear joystick key for control thread
            self.joystick_keys[key] = False

            # Update status
            active_keys = [k.upper() for k, v in self.joystick_keys.items() if v]
            if active_keys:
                self.joystick_status_var.set(
                    f"Joystick: ACTIVE ({','.join(active_keys)})"
                )
            else:
                self.joystick_status_var.set("Joystick: ACTIVE")

            # Log button release with hold mode info when all keys released
            if not any(self.joystick_keys.values()):
                mode_info = (
                    " - Returning to origin"
                    if self.joystick_hold_at_origin
                    else " - Holding current position"
                )
                self.log_to_output(
                    f"Continuous movement: {self._key_to_direction(key)} stopped{mode_info}"
                )
            else:
                self.log_to_output(
                    f"Continuous movement: {self._key_to_direction(key)} stopped"
                )

    def _key_to_direction(self, key):
        """Convert key to direction name"""
        directions = {
            "w": "Forward (W)",
            "a": "Left (A)",
            "s": "Backward (S)",
            "d": "Right (D)",
        }
        return directions.get(key, key.upper())

    def toggle_joystick_hold_mode(self):
        """Toggle joystick position hold mode between origin and current position"""
        self.joystick_hold_at_origin = self.joystick_hold_origin_var.get()

        if self.joystick_hold_at_origin:
            mode_text = "Origin mode: Drone will return to takeoff point when releasing joystick"
        else:
            mode_text = "Current position mode: Drone will hold at the position where joystick is released"

        self.log_to_output(f"Joystick hold mode changed: {mode_text}")
        self.log_to_output(
            f"Joystick hold mode: {'Hold at Origin' if self.joystick_hold_at_origin else 'Hold at Current Position'}"
        )

    def start_sensor_test(self):  # New function for sensor test
        """Start the sensor test in a separate thread"""
        if (
            not self.sensor_test_running and not self.flight_running
        ):  # Prevent starting if flight is active
            self.sensor_test_running = True
            self.sensor_test_button.config(
                text="Stop Sensor Test", command=self.stop_sensor_test, bg="red"
            )
            self.status_var.set("Status: Starting Sensor Test...")
            self.sensor_test_thread = threading.Thread(
                target=self.sensor_test_controller_thread
            )
            self.sensor_test_thread.daemon = True
            self.sensor_test_thread.start()
        elif self.flight_running:
            print("Cannot start Sensor Test while Flight is active.")
            self.status_var.set("Status: Flight Active - Cannot Test Sensors")

    def stop_sensor_test(self):
        """Stop the sensor test"""
        if self.sensor_test_running:
            global sensor_test_active
            sensor_test_active = False
            self.sensor_test_running = False
            if self.sensor_test_thread and self.sensor_test_thread.is_alive():
                self.sensor_test_thread.join(timeout=2.0)
            self.status_var.set("Status: Sensor test stopped")
            self.sensor_test_button.config(
                text="Sensor Test", command=self.start_sensor_test, bg="lightblue"
            )

    def reset_battery(self):
        """Reset battery voltage reading to allow new flights with fresh batteries"""
        global current_battery_voltage, battery_data_ready
        current_battery_voltage = 0.0
        battery_data_ready = False
        self.battery_var.set("Battery: RESET - Waiting for new reading")
        self.status_var.set(
            "Status: Battery voltage reset. New reading will update shortly."
        )
        self.log_to_output("Battery voltage reset to 0.0V")

    def sensor_test_controller_thread(self):  # New thread function for sensor test
        """Sensor test controller running in separate thread"""
        global flight_phase, sensor_test_active, scf_instance
        global integrated_position_x, integrated_position_y, last_integration_time, last_reset_time
        global position_integration_enabled  # Need to access this to enable integration
        global current_battery_voltage, battery_data_ready  # Reset battery on new connection

        # Clear previous run data at start of new sensor test
        self.root.after(0, lambda: self.clear_output())
        self.root.after(0, lambda: self.clear_graphs())

        # Reset position immediately to update GUI (fix reset delay)
        integrated_position_x = 0.0
        integrated_position_y = 0.0
        last_integration_time = time.time()
        # Reset PID state as well
        position_integral_x = 0.0
        position_integral_y = 0.0

        sensor_test_active = True
        flight_phase = "SENSOR_TEST"  # Update phase

        # Reset battery voltage for new connection
        current_battery_voltage = 0.0
        battery_data_ready = False

        cflib.crtp.init_drivers()
        cf = Crazyflie(rw_cache="./cache")
        log_motion = None
        log_battery = None

        try:
            with SyncCrazyflie(DRONE_URI, cf=cf) as scf:
                scf_instance = scf
                # Enable NeoPixel controls now that a Crazyflie link is established
                try:
                    self.update_button(self.blink_button, state=tk.NORMAL)
                    self.update_button(self.clear_leds_button, state=tk.NORMAL)
                    self.update_button(self.set_color_button, state=tk.NORMAL)
                except Exception:
                    pass
                # Setup logging (same as flight)
                log_motion, log_battery = setup_logging(cf)
                use_position_hold = log_motion is not None
                if use_position_hold:
                    time.sleep(1.0)

                # Initialize flight parameters (skip if in debug mode, but logging still happens)
                if not DEBUG_MODE:
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.1)
                    cf.param.set_value("commander.enHighLevel", "1")
                    time.sleep(0.5)
                else:
                    self.log_to_output(
                        "DEBUG MODE: Skipping flight initialization for sensor test"
                    )

                # Enable position integration for sensor test
                reset_position_tracking()

                # Run sensor test loop (no motor commands)
                flight_phase = "SENSOR_TEST"
                start_time = time.time()
                if self.enable_sensor_logging_var.get():
                    init_csv_logging(logger=self.log_to_output)
                while sensor_test_active:  # Continue while sensor test is active
                    # Safety check: link and sensor stale
                    if not check_link_safety(cf, logger=self.log_to_output):
                        sensor_test_active = False
                        break

                    # Calculate corrections (they will be 0 if PID params are 0, but still updates internal state)
                    if use_position_hold and sensor_data_ready:
                        motion_vx, motion_vy = calculate_position_hold_corrections()
                        # Check for periodic reset
                        if periodic_position_reset():
                            flight_phase = "SENSOR_TEST (RESET)"
                        else:
                            flight_phase = "SENSOR_TEST"
                    time.sleep(CONTROL_UPDATE_RATE)  # Maintain control loop rate
                    if self.enable_sensor_logging_var.get():
                        log_to_csv()
        except Exception as e:
            flight_phase = "ERROR"
            self.log_to_output(f"Sensor Test Error: {str(e)}")
        finally:
            # Stop logging
            if self.enable_sensor_logging_var.get():
                close_csv_logging(logger=self.log_to_output)
            if log_motion:
                try:
                    log_motion.stop()
                except:
                    pass
            if log_battery:
                try:
                    log_battery.stop()
                except:
                    pass
            # Disable NeoPixel controls when sensor test stops
            # Disable NeoPixel controls when sensor test stops
            try:
                self.update_button(self.blink_button, state=tk.DISABLED)
                self.update_button(self.clear_leds_button, state=tk.DISABLED)
                self.update_button(self.set_color_button, state=tk.DISABLED)
            except Exception:
                pass
            sensor_test_active = False
            flight_phase = "IDLE"
            self.sensor_test_running = False
            self.update_button(
                self.sensor_test_button,
                text="Sensor Test",
                command=self.start_sensor_test,
                bg="lightblue",
            )
            self.update_status("Status: Sensor Test Stopped")

    def start_flight(self):
        """Start the flight in a separate thread with battery and sensor safety checks"""
        if not self.flight_running and not self.sensor_test_running:
            # Battery safety check
            if (
                current_battery_voltage > 0
                and current_battery_voltage < LOW_BATTERY_THRESHOLD
            ):
                self.status_var.set(
                    f"Status: Battery too low ({current_battery_voltage:.2f}V)! Cannot start flight."
                )
                self.log_to_output(
                    f"SAFETY: Flight blocked - Battery voltage {current_battery_voltage:.2f}V is below 3.5V minimum"
                )
                return
            elif current_battery_voltage == 0.0:
                self.log_to_output("WARNING: Battery voltage unknown")

            # SENSOR SAFETY CHECK: ensure height and motion data are flowing
            if not sensor_data_ready:
                self.status_var.set(
                    "Status: Sensor data not ready! Wait for height & motion readings."
                )
                self.log_to_output(
                    "SAFETY: Flight blocked - No sensor data received yet."
                )
                return

            if current_height <= 0.0:  # e.g., drone on ground or invalid
                self.status_var.set(
                    "Status: Invalid height reading! Ensure drone is powered and sensors active."
                )
                self.log_to_output(
                    "SAFETY: Flight blocked - Height too low or invalid:",
                    current_height,
                )
                return

            # Proceed
            self.flight_running = True
            self.start_button.config(
                text="Stop Flight", command=self.emergency_stop, bg="red"
            )
            self.status_var.set("Status: Starting Flight...")
            self.flight_thread = threading.Thread(target=self.flight_controller_thread)
            self.flight_thread.daemon = True
            self.flight_thread.start()

            # Log to output window
            self.log_to_output("Flight started")
        elif self.sensor_test_running:
            self.log_to_output("Cannot start Flight while Sensor Test is active.")
            self.status_var.set("Status: Sensor Test Active - Cannot Start Flight")

    def emergency_stop(self):
        """Emergency stop the flight or sensor test"""
        global flight_active, sensor_test_active
        flight_active = False
        sensor_test_active = False  # Stop sensor test as well
        self.flight_running = False
        self.sensor_test_running = False  # Reset sensor test flag
        self.start_button.config(
            text="Start Flight", command=self.start_flight, bg="green"
        )
        self.sensor_test_button.config(
            text="Sensor Test", command=self.start_sensor_test, bg="lightblue"
        )

        # Also stop joystick control if active
        if self.joystick_active:
            self.joystick_active = False
            global maneuver_active
            maneuver_active = False

            # Clear all joystick keys
            self.joystick_keys = {"w": False, "a": False, "s": False, "d": False}
            self.key_pressed_flags = {"w": False, "a": False, "s": False, "d": False}

            # Update joystick status
            self.joystick_status_var.set("Joystick: EMERGENCY STOP")

            # Update UI buttons
            self.start_joystick_button.config(state=tk.NORMAL)

            # Log joystick stop
            self.log_to_output("Joystick control emergency stopped")

        self.status_var.set("Status: Emergency Stopped")

    def flight_controller_thread(self):
        """Flight controller running in separate thread"""
        global flight_phase, flight_active, scf_instance
        global integrated_position_x, integrated_position_y, last_integration_time, last_reset_time
        global maneuver_active, target_position_x, target_position_y
        global shape_active, shape_waypoints, shape_index, waypoint_start_time
        global current_battery_voltage, battery_data_ready  # Reset battery on new connection
        global position_integral_x, position_integral_y  # PID integral terms
        global waypoint_start_time  # Ensure timer is reset

        # Clear previous run data at start of new flight
        self.root.after(0, lambda: self.clear_output())
        self.root.after(0, lambda: self.clear_graphs())

        # Reset position immediately to update GUI (fix reset delay)
        integrated_position_x = 0.0
        integrated_position_y = 0.0
        # Implicitly reset target position to origin for standard flight if no maneuver is active
        # This allows maneuvers to be started from IDLE while preserving their target coordinates
        if not maneuver_active:
            target_position_x = 0.0
            target_position_y = 0.0
        last_integration_time = time.time()
        # Reset PID state as well
        position_integral_x = 0.0
        position_integral_y = 0.0
        waypoint_start_time = time.time()  # Reset timer at start of flight thread

        cflib.crtp.init_drivers()
        cf = Crazyflie(rw_cache="./cache")
        log_motion = None
        log_battery = None

        # Reset battery voltage for new connection
        current_battery_voltage = 0.0
        battery_data_ready = False

        try:
            flight_phase = "CONNECTING"
            with SyncCrazyflie(DRONE_URI, cf=cf) as scf:
                scf_instance = scf
                flight_active = True
                
                # Apply firmware parameters right after connection
                apply_firmware_parameters(cf, logger=self.log_to_output)

                # Setup logging
                flight_phase = "SETUP"
                log_motion, log_battery = setup_logging(cf, logger=self.log_to_output)
                use_position_hold = log_motion is not None
                if use_position_hold:
                    time.sleep(1.0)

                # Reset position tracking immediately after logging setup for faster initialization
                reset_position_tracking()

                # SAFETY CHECK: Verify position integration reset was successful before takeoff
                if (
                    not position_integration_enabled
                    or integrated_position_x != 0.0
                    or integrated_position_y != 0.0
                ):
                    error_msg = f"SAFETY: Position integration reset failed! Integration enabled: {position_integration_enabled}, Position: ({integrated_position_x:.3f}, {integrated_position_y:.3f}). Cannot start flight."
                    self.log_to_output(error_msg)
                    flight_phase = "SAFETY_ERROR"
                    raise Exception(error_msg)

                # Initialize flight (skip if in debug mode)
                if not DEBUG_MODE:
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.1)
                    cf.param.set_value("commander.enHighLevel", "1")
                    time.sleep(0.5)
                else:
                    self.log_to_output("DEBUG MODE: Skipping flight initialization")

                # Takeoff (position integration enabled from start for safety)
                flight_phase = "TAKEOFF"
                if DEBUG_MODE:
                    self.log_to_output("DEBUG MODE: Simulating takeoff phase")
                start_time = time.time()
                init_csv_logging(logger=self.log_to_output)

                # Height sensor validation during takeoff
                takeoff_height_start = current_height
                height_sensor_min_change = HEIGHT_SENSOR_MIN_CHANGE

                while time.time() - start_time < TAKEOFF_TIME and flight_active:
                    # Safety check: link and sensor stale
                    if not check_link_safety(cf, logger=self.log_to_output):
                        flight_active = False
                        break

                    # Calculate elapsed time for takeoff ramp
                    elapsed_takeoff_time = time.time() - start_time

                    if not DEBUG_MODE:
                        # Enable control corrections during takeoff if height is sufficient (> 5cm)
                        # This prevents drift during the 1.5s takeoff phase
                        if use_position_hold and sensor_data_ready and current_height > 0.04:
                            # Hold at origin (0,0) during takeoff regardless of maneuver target
                            motion_vx, motion_vy = calculate_position_hold_corrections(0.0, 0.0)
                        else:
                            motion_vx, motion_vy = 0.0, 0.0
                        
                        # Apply corrections (TRIM + PID output)
                        total_vx = TRIM_VX + motion_vy
                        total_vy = TRIM_VY + motion_vx
                        
                        # SMOOTH TAKEOFF RAMP: Gradually increase height target to minimize bouncing
                        if ENABLE_TAKEOFF_RAMP:
                            # Calculate progress (0.0 to 1.0)
                            takeoff_progress = min(1.0, elapsed_takeoff_time / TAKEOFF_TIME)
                            # Ramp from current ground height to target height
                            command_height = takeoff_height_start + (TARGET_HEIGHT - takeoff_height_start) * takeoff_progress
                        else:
                            # Direct target height (leveraging firmware's internal ramp or jumping to height)
                            command_height = TARGET_HEIGHT
                        
                        cf.commander.send_hover_setpoint(
                            total_vx, total_vy, 0, command_height
                        )

                    log_to_csv()
                    time.sleep(CONTROL_UPDATE_RATE)

                # POST-TAKEOFF SAFETY CHECK: Verify height sensor worked during full takeoff period
                takeoff_duration = time.time() - start_time
                final_height_change = current_height - takeoff_height_start

                if ENABLE_HEIGHT_SENSOR_SAFETY and final_height_change < height_sensor_min_change and not DEBUG_MODE:
                    # Height sensor appears stuck despite full takeoff thrust period - emergency stop
                    emergency_msg = (
                        f"EMERGENCY STOP: Height sensor failure detected! "
                        f"Height stuck at {current_height:.3f}m (change: {final_height_change:.3f}m) "
                        f"after full {takeoff_duration:.1f}s takeoff to {TARGET_HEIGHT:.3f}m target. "
                        f"Expected >{height_sensor_min_change:.3f}m change with commanded thrust."
                    )
                    self.log_to_output(emergency_msg)
                    flight_phase = "EMERGENCY_HEIGHT_SENSOR"
                    # Emergency stop motors
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    raise Exception(emergency_msg)

                # Height stabilization phase - wait for drone to stabilize at target height
                flight_phase = "STABILIZING"
                if DEBUG_MODE:
                    self.log_to_output("DEBUG MODE: Simulating stabilization phase")
                stabilization_start = time.time()
                stabilization_duration = 3.0  # 3 seconds to stabilize

                # Height sensor validation during stabilization
                stabilization_height_check_start = time.time()
                stabilization_height_timeout = 2.0  # seconds - if height doesn't stabilize within this time, check sensor

                while (
                    time.time() - stabilization_start < stabilization_duration
                    and flight_active
                ):
                    # Safety check: link and sensor stale
                    if not check_link_safety(cf, logger=self.log_to_output):
                        flight_active = False
                        break

                    # HEIGHT SENSOR SAFETY CHECK: Detect stuck/frozen height sensor during stabilization
                    stabilization_elapsed = (
                        time.time() - stabilization_height_check_start
                    )

                    # After 2 seconds of stabilization, check if height is reasonably close to target
                    if (
                        ENABLE_HEIGHT_SENSOR_SAFETY and stabilization_elapsed > stabilization_height_timeout
                        and abs(current_height - TARGET_HEIGHT)
                        > 0.3  # Allow 30cm tolerance
                        and not DEBUG_MODE
                    ):
                        # Height sensor may be stuck - the drone should be at target height but isn't
                        emergency_msg = (
                            f"EMERGENCY STOP: Height sensor failure during stabilization! "
                            f"Height: {current_height:.3f}m, Target: {TARGET_HEIGHT:.3f}m "
                            f"(error: {abs(current_height - TARGET_HEIGHT):.3f}m) after {stabilization_elapsed:.1f}s. "
                            f"Height sensor appears stuck or inaccurate."
                        )
                        self.log_to_output(emergency_msg)
                        flight_phase = "EMERGENCY_HEIGHT_SENSOR_STABILIZATION"
                        # Emergency stop motors
                        cf.commander.send_setpoint(0, 0, 0, 0)
                        raise Exception(emergency_msg)

                    if use_position_hold and sensor_data_ready:
                        # Hold at origin (0,0) during stabilization phase
                        motion_vx, motion_vy = calculate_position_hold_corrections(0.0, 0.0)
                    else:
                        motion_vx, motion_vy = 0.0, 0.0
                    log_to_csv()
                    # Apply corrections (note: axes swapped)
                    total_vx = TRIM_VX + motion_vy
                    total_vy = TRIM_VY + motion_vx
                    if not DEBUG_MODE:
                        cf.commander.send_hover_setpoint(
                            total_vx, total_vy, 0, TARGET_HEIGHT
                        )
                    time.sleep(CONTROL_UPDATE_RATE)

                # Position hold hover or maneuver - Main Loop
                hover_start_time = None
                waypoint_arrival_time = None  # Track when we reach a corner
                
                while flight_active:
                    if maneuver_active:
                        flight_phase = "MISSION"
                        if not check_link_safety(cf, logger=self.log_to_output):
                            flight_active = False
                            break

                        if use_position_hold and sensor_data_ready:
                            motion_vx, motion_vy = calculate_position_hold_corrections()
                            
                            distance_to_target = ((integrated_position_x - target_position_x)**2 + 
                                               (integrated_position_y - target_position_y)**2)**0.5
                            
                            # Log progress every 0.5s
                            if int(time.time() * 2) % 4 == 0:
                                self.log_to_output(f"MISSION: Dist={distance_to_target:.3f}m, Target=({target_position_x:.2f}, {target_position_y:.2f})")

                            # Check completion and handle stabilization
                            if distance_to_target < MANEUVER_THRESHOLD:
                                if waypoint_arrival_time is None:
                                    waypoint_arrival_time = time.time()
                                    self.log_to_output(f"Waypoint {shape_index if shape_active else ''} reached! Stabilizing for {WAYPOINT_STABILIZATION_TIME}s...")
                                
                                # Wait for stabilization time
                                if time.time() - waypoint_arrival_time >= WAYPOINT_STABILIZATION_TIME:
                                    waypoint_arrival_time = None # Reset for next waypoint
                                    
                                    if shape_active and shape_index < len(shape_waypoints) - 1:
                                        shape_index += 1
                                        target_position_x, target_position_y = shape_waypoints[shape_index]
                                        self.log_to_output(f"Moving to next waypoint: ({target_position_x:.2f}, {target_position_y:.2f})")
                                        # Do NOT reset integrals for next leg to preserve auto-trim balance
                                        # reset_position_tracking(reset_integrals=False)
                                    else:
                                        self.log_to_output(f"Mission target reached at ({integrated_position_x:.2f}, {integrated_position_y:.2f})!")
                                        maneuver_active = False
                                        shape_active = False
                                        # When mission ends, we'll enter HOVER mode in next iteration
                                        hover_start_time = time.time() # Start hover timer
                            else:
                                # Still moving toward target, reset arrival time if we drift away
                                waypoint_arrival_time = None 
                        else:
                            motion_vx, motion_vy = 0.0, 0.0

                        total_vx = TRIM_VX + motion_vy
                        total_vy = TRIM_VY + motion_vx
                        if not DEBUG_MODE:
                            cf.commander.send_hover_setpoint(total_vx, total_vy, 0, TARGET_HEIGHT)
                        log_to_csv()
                        time.sleep(CONTROL_UPDATE_RATE)

                    else:
                        # HOVER Mode
                        if hover_start_time is None:
                            hover_start_time = time.time()
                        
                        elapsed_hover = time.time() - hover_start_time
                        if elapsed_hover >= HOVER_DURATION:
                            self.log_to_output(f"Hover duration ({HOVER_DURATION}s) completed.")
                            break # Exit main loop to land
                            
                        flight_phase = "HOVER"
                        if use_position_hold and sensor_data_ready:
                            motion_vx, motion_vy = calculate_position_hold_corrections()
                            # Check for periodic reset
                            if periodic_position_reset():
                                flight_phase = "HOVER (RESET)"
                        else:
                            motion_vx, motion_vy = 0.0, 0.0
                        
                        log_to_csv()
                        total_vx = TRIM_VX + motion_vy
                        total_vy = TRIM_VY + motion_vx
                        if not DEBUG_MODE:
                            cf.commander.send_hover_setpoint(total_vx, total_vy, 0, TARGET_HEIGHT)
                        time.sleep(CONTROL_UPDATE_RATE)

                # Landing with stabilization
                flight_phase = "LANDING"
                if DEBUG_MODE:
                    self.log_to_output("DEBUG MODE: Simulating landing phase")
                start_time = time.time()
                while (
                    time.time() - start_time < LANDING_TIME and flight_active
                ):
                    if use_position_hold and sensor_data_ready:
                        motion_vx, motion_vy = calculate_position_hold_corrections()
                    else:
                        motion_vx, motion_vy = 0.0, 0.0
                    
                    # Apply corrections even during landing to prevent drift
                    total_vx = TRIM_VX + motion_vy
                    total_vy = TRIM_VY + motion_vx
                    
                    if not DEBUG_MODE:
                        # Command height 0.0 to descend while maintaining horizontal position
                        cf.commander.send_hover_setpoint(total_vx, total_vy, 0, 0.0)
                    log_to_csv()
                    time.sleep(CONTROL_UPDATE_RATE)

                # Stop motors
                if not DEBUG_MODE:
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.1)  # Brief wait to ensure stop command is processed
                flight_phase = "COMPLETE"

        except Exception as e:
            error_msg = str(e)
            flight_phase = "ERROR"
            # Log the error message content to output
            self.log_to_output(f"Flight Error: {error_msg}")
        finally:
            # Stop logging
            close_csv_logging(logger=self.log_to_output)
            if log_motion:
                try:
                    log_motion.stop()
                except:
                    pass
            if log_battery:
                try:
                    log_battery.stop()
                except:
                    pass
            flight_active = False
            self.flight_running = False
            self.update_button(
                self.start_button,
                text="Start Flight",
                command=self.start_flight,
                bg="green",
            )
            if flight_phase != "COMPLETE":
                self.update_status(f"Status: {flight_phase}")
            else:
                self.update_status("Status: Flight Complete")

    def start_joystick_control(self):
        """Start joystick control using keyboard input"""
        if (
            not self.joystick_active
            and not self.flight_running
            and not self.sensor_test_running
        ):
            try:
                # Get sensitivity value
                JOYSTICK_SENSITIVITY = float(self.joystick_sensitivity_var.get())
                if JOYSTICK_SENSITIVITY < 0.1 or JOYSTICK_SENSITIVITY > 2.0:
                    raise ValueError("Sensitivity must be between 0.1 and 2.0")

                # Battery safety check
                if (
                    current_battery_voltage > 0
                    and current_battery_voltage < LOW_BATTERY_THRESHOLD
                ):
                    self.status_var.set(
                        f"Status: Battery too low ({current_battery_voltage:.2f}V)! Cannot start joystick control."
                    )
                    return
                elif current_battery_voltage == 0.0:
                    self.log_to_output("WARNING: Battery voltage unknown")

                # SENSOR SAFETY CHECK
                if not sensor_data_ready:
                    self.status_var.set(
                        "Status: Sensor data not ready! Wait for height & motion readings."
                    )
                    return

                if current_height <= 0.0:
                    self.status_var.set(
                        "Status: Invalid height reading! Ensure drone is powered and sensors active."
                    )
                    return

                # Start joystick control
                self.joystick_active = True
                self.start_joystick_button.config(state=tk.DISABLED)
                # self.stop_joystick_button.config(state=tk.NORMAL)  # Removed duplicate stop button
                self.joystick_status_var.set("Joystick: ACTIVE")
                self.status_var.set(
                    "Status: Joystick Control Starting - Stabilizing height first..."
                )

                # Log to output window
                self.log_to_output("Joystick control started")

                # Initialize joystick target position based on hold mode
                global target_position_x, target_position_y, maneuver_active

                if self.joystick_hold_at_origin:
                    # Hold at Origin mode: Set target to origin (0, 0)
                    target_position_x = 0.0
                    target_position_y = 0.0
                    self.log_to_output(
                        "Joystick mode: Hold at Origin - will return to takeoff point when released"
                    )
                else:
                    # Hold at Current Position mode: Set target to current position
                    target_position_x = integrated_position_x
                    target_position_y = integrated_position_y
                    self.log_to_output(
                        "Joystick mode: Hold at Current Position - will maintain position when released"
                    )

                maneuver_active = False  # Disable maneuver damping for sharper joystick response

                # Force focus to the window for key events
                self.root.focus_force()

                # Start joystick control thread
                self.joystick_thread = threading.Thread(
                    target=self.joystick_control_thread
                )
                self.joystick_thread.daemon = True
                self.joystick_thread.start()

            except ValueError as e:
                self.status_var.set(f"Status: Invalid sensitivity value - {str(e)}")
        elif self.flight_running:
            self.status_var.set("Status: Cannot start joystick while flight is active")
        elif self.sensor_test_running:
            self.status_var.set(
                "Status: Cannot start joystick while sensor test is active"
            )

    def stop_joystick_control(self):
        """Stop joystick control"""
        if self.joystick_active:
            self.joystick_active = False
            global maneuver_active
            maneuver_active = False

            # Wait for thread to finish
            if self.joystick_thread and self.joystick_thread.is_alive():
                self.joystick_thread.join(timeout=1.0)

            # Update UI
            self.start_joystick_button.config(state=tk.NORMAL)
            # self.stop_joystick_button.config(state=tk.DISABLED)  # Removed duplicate stop button
            self.joystick_status_var.set("Joystick: INACTIVE")
            self.status_var.set("Status: Joystick Control Stopped")

            # Log to output window
            self.log_to_output("Joystick control stopped")

    def joystick_control_thread(self):
        """Joystick control thread that handles position updates"""
        global target_position_x, target_position_y, maneuver_active, flight_active
        global current_battery_voltage, battery_data_ready  # Reset battery on new connection
        global integrated_position_x, integrated_position_y  # Need access to current position
        global last_integration_time, last_reset_time  # Need to reset integration timing
        global position_integration_enabled  # Control position integration
        global position_integral_x, position_integral_y, velocity_integral_x, velocity_integral_y
        global last_position_error_x, last_position_error_y, last_velocity_error_x, last_velocity_error_y

        # Clear previous run data at start of new joystick control
        self.root.after(0, lambda: self.clear_output())
        self.root.after(0, lambda: self.clear_graphs())

        # Reset position immediately to update GUI (fix reset delay)
        integrated_position_x = 0.0
        integrated_position_y = 0.0
        last_integration_time = time.time()
        # Reset PID state as well
        position_integral_x = 0.0
        position_integral_y = 0.0

        cflib.crtp.init_drivers()
        cf = Crazyflie(rw_cache="./cache")
        log_motion = None
        log_battery = None

        # Reset battery voltage for new connection
        current_battery_voltage = 0.0
        battery_data_ready = False

        try:
            with SyncCrazyflie(DRONE_URI, cf=cf) as scf:
                scf_instance = scf
                flight_active = True
                
                # Apply firmware parameters right after connection
                apply_firmware_parameters(cf, logger=self.log_to_output)

                # Setup logging
                log_motion, log_battery = setup_logging(cf, logger=self.log_to_output)
                use_position_hold = log_motion is not None
                if use_position_hold:
                    time.sleep(1.0)

                # Reset position tracking immediately after logging setup for faster initialization
                reset_position_tracking()

                # SAFETY CHECK: Verify position integration reset was successful before takeoff
                if (
                    not position_integration_enabled
                    or integrated_position_x != 0.0
                    or integrated_position_y != 0.0
                ):
                    error_msg = f"SAFETY: Position integration reset failed! Integration enabled: {position_integration_enabled}, Position: ({integrated_position_x:.3f}, {integrated_position_y:.3f}). Cannot start joystick control."
                    self.log_to_output(error_msg)
                    flight_phase = "JOYSTICK_SAFETY_ERROR"
                    raise Exception(error_msg)

                # Initialize flight
                if not DEBUG_MODE:
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.1)
                    cf.param.set_value("commander.enHighLevel", "1")
                    time.sleep(0.5)
                else:
                    self.log_to_output(
                        "DEBUG MODE: Skipping flight initialization for joystick control"
                    )

                # Takeoff
                flight_phase = "JOYSTICK_TAKEOFF"
                start_time = time.time()
                init_csv_logging(logger=self.log_to_output)

                # Height sensor validation during takeoff
                takeoff_height_start = current_height
                height_sensor_min_change = HEIGHT_SENSOR_MIN_CHANGE

                while time.time() - start_time < TAKEOFF_TIME and self.joystick_active:
                    # Safety check: link and sensor stale
                    if not check_link_safety(cf, logger=self.log_to_output):
                        self.joystick_active = False
                        break

                    if not DEBUG_MODE:
                        # Enable control corrections during takeoff if height is sufficient (> 5cm)
                        if use_position_hold and sensor_data_ready and current_height > 0.05:
                            # Hold at origin (0,0) during joystick takeoff
                            motion_vx, motion_vy = calculate_position_hold_corrections(0.0, 0.0)
                        else:
                            motion_vx, motion_vy = 0.0, 0.0
                        
                        total_vx = TRIM_VX + motion_vy
                        total_vy = TRIM_VY + motion_vx

                        cf.commander.send_hover_setpoint(
                            total_vx, total_vy, 0, TARGET_HEIGHT
                        )
                    log_to_csv()
                    time.sleep(0.01)

                # POST-TAKEOFF SAFETY CHECK: Verify height sensor worked during full takeoff period
                takeoff_duration = time.time() - start_time
                final_height_change = current_height - takeoff_height_start

                if final_height_change < height_sensor_min_change and not DEBUG_MODE:
                    # Height sensor appears stuck despite full takeoff thrust period - emergency stop
                    emergency_msg = (
                        f"EMERGENCY STOP: Height sensor failure detected! "
                        f"Height stuck at {current_height:.3f}m (change: {final_height_change:.3f}m) "
                        f"after full {takeoff_duration:.1f}s takeoff to {TARGET_HEIGHT:.3f}m target. "
                        f"Expected >{height_sensor_min_change:.3f}m change with commanded thrust."
                    )
                    self.log_to_output(emergency_msg)
                    flight_phase = "JOYSTICK_EMERGENCY_HEIGHT_SENSOR"
                    # Emergency stop motors
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    raise Exception(emergency_msg)

                # Height stabilization phase - wait for drone to stabilize at target height
                flight_phase = "JOYSTICK_STABILIZING"
                stabilization_start = time.time()
                stabilization_duration = 3.0  # 3 seconds to stabilize

                # Height sensor validation during stabilization
                stabilization_height_check_start = time.time()
                stabilization_height_timeout = 2.0  # seconds - if height doesn't stabilize within this time, check sensor

                while (
                    time.time() - stabilization_start < stabilization_duration
                    and self.joystick_active
                ):
                    # Safety check: link and sensor stale
                    if not check_link_safety(cf, logger=self.log_to_output):
                        self.joystick_active = False
                        break

                    # HEIGHT SENSOR SAFETY CHECK: Detect stuck/frozen height sensor during stabilization
                    stabilization_elapsed = (
                        time.time() - stabilization_height_check_start
                    )

                    # After 2 seconds of stabilization, check if height is reasonably close to target
                    if (
                        ENABLE_HEIGHT_SENSOR_SAFETY and stabilization_elapsed > stabilization_height_timeout
                        and abs(current_height - TARGET_HEIGHT)
                        > 0.3  # Allow 30cm tolerance
                        and not DEBUG_MODE
                    ):
                        # Height sensor may be stuck - the drone should be at target height but isn't
                        emergency_msg = (
                            f"EMERGENCY STOP: Height sensor failure during stabilization! "
                            f"Height: {current_height:.3f}m, Target: {TARGET_HEIGHT:.3f}m "
                            f"(error: {abs(current_height - TARGET_HEIGHT):.3f}m) after {stabilization_elapsed:.1f}s. "
                            f"Height sensor appears stuck or inaccurate."
                        )
                        self.log_to_output(emergency_msg)
                        flight_phase = "JOYSTICK_EMERGENCY_HEIGHT_SENSOR_STABILIZATION"
                        # Emergency stop motors
                        cf.commander.send_setpoint(0, 0, 0, 0)
                        raise Exception(emergency_msg)

                    if use_position_hold and sensor_data_ready:
                        # Hold at origin (0,0) during joystick stabilization
                        motion_vx, motion_vy = calculate_position_hold_corrections(0.0, 0.0)
                    else:
                        motion_vx, motion_vy = 0.0, 0.0
                    log_to_csv()
                    # Apply corrections (note: axes swapped)
                    total_vx = TRIM_VX + motion_vy
                    total_vy = TRIM_VY + motion_vx
                    if not DEBUG_MODE:
                        cf.commander.send_hover_setpoint(
                            total_vx, total_vy, 0, TARGET_HEIGHT
                        )
                    time.sleep(CONTROL_UPDATE_RATE)

                # Joystick control loop
                flight_phase = "JOYSTICK_CONTROL"

                # Previous time for dt calculation
                last_loop_time = time.time()
                
                # Variables for smooth key release handling and visualization
                keys_were_pressed = False
                key_release_time = 0.0
                self._joystick_debug_counter = 0  # Reset debug counter for new session

                while self.joystick_active:
                    # Safety check: link and sensor stale
                    if not check_link_safety(cf, logger=self.log_to_output):
                        self.joystick_active = False
                        break

                    current_time_loop = time.time()
                    dt = current_time_loop - last_loop_time
                    last_loop_time = current_time_loop
                    if dt > 0.1: dt = 0.1 # Cap dt to prevent jumps on lag

                    # 1. Get Joystick Input
                    sensitivity = float(self.joystick_sensitivity_var.get())
                    joystick_vx = 0.0
                    joystick_vy = 0.0

                    if self.joystick_keys["w"]:  # Forward (positive Y)
                        joystick_vy += sensitivity
                    if self.joystick_keys["s"]:  # Backward (negative Y)
                        joystick_vy -= sensitivity
                    if self.joystick_keys["a"]:  # Left (positive X)
                        joystick_vx += sensitivity
                    if self.joystick_keys["d"]:  # Right (negative X)
                        joystick_vx -= sensitivity

                    any_keys_pressed = any(self.joystick_keys.values())

                    # Check key release for visualization
                    if keys_were_pressed and not any_keys_pressed:
                         key_release_points.append((integrated_position_x, integrated_position_y))
                         self.log_to_output(f"Keys released - Holding target at ({target_position_x:.2f}, {target_position_y:.2f})")
                    keys_were_pressed = any_keys_pressed


                    # 2. Update Target Position (Target Integration)
                    if use_position_hold and sensor_data_ready:
                        # Move target based on joystick input (Closed-Loop Velocity Control)
                        if not self.joystick_hold_at_origin: # "Hold at Current" Mode (Flying Mode)
                            target_position_x += joystick_vx * dt
                            target_position_y += joystick_vy * dt
                            
                            # Clamp Target to prevent integral windup if drone is stuck or lagging
                            dx = target_position_x - integrated_position_x
                            dy = target_position_y - integrated_position_y
                            if abs(dx) > MAX_POSITION_ERROR:
                                target_position_x = integrated_position_x + math.copysign(MAX_POSITION_ERROR, dx)
                            if abs(dy) > MAX_POSITION_ERROR:
                                target_position_y = integrated_position_y + math.copysign(MAX_POSITION_ERROR, dy)
                                
                        else: # "Hold at Origin" Mode (Spring Mode)
                            target_position_x = 0.0
                            target_position_y = 0.0

                        # 3. Calculate PID Corrections (Always Active)
                        motion_vx, motion_vy = calculate_position_hold_corrections()
                    else:
                        motion_vx, motion_vy = 0.0, 0.0

                    # 4. Combine Outputs (Feedforward + Feedback)
                    # Feedforward: Joystick input (immediate response)
                    # Feedback: PID output (correction for drift/error)
                    
                    # Store for CSV logging
                    current_correction_vx = motion_vx
                    current_correction_vy = motion_vy
                    
                    log_to_csv()
                    
                    # Apply corrections (note: axes swapped for Crazyflie coordinate system)
                    # Mapping: Global Y (Forward) -> Drone X (Forward) ?
                    # Wait, calculating 'total_vx' (Command 1) and 'total_vy' (Command 2)
                    
                    # Original Code Logic Preservation: 
                    # total_vx = TRIM + motion_vy (which was joystick_vy)
                    # total_vy = TRIM + motion_vx (which was joystick_vx)
                    
                    # New Logic:
                    # We want to add Feedforward Joystick Terms to the same channels.
                    # If motion_vy maps to Channel 1 (VX), then joystick_vy should also map to Channel 1 (VX).
                    
                    total_vx = TRIM_VX + motion_vy + joystick_vy
                    total_vy = TRIM_VY + motion_vx + joystick_vx
                    
                    if not DEBUG_MODE:
                        cf.commander.send_hover_setpoint(
                            total_vx, total_vy, 0, TARGET_HEIGHT
                        )
                    
                    time.sleep(CONTROL_UPDATE_RATE)

                    # Debug logging
                    if self._joystick_debug_counter % 50 == 0 and any_keys_pressed:
                         self.log_to_output(
                            f"Joy:({joystick_vx:.2f},{joystick_vy:.2f}) "
                            f"Tgt:({target_position_x:.2f},{target_position_y:.2f}) "
                            f"Pos:({integrated_position_x:.2f},{integrated_position_y:.2f})"
                        )
                    self._joystick_debug_counter += 1

                # Landing when joystick control stops
                flight_phase = "JOYSTICK_LANDING"
                start_time = time.time()
                while time.time() - start_time < LANDING_TIME and flight_active:
                    if not DEBUG_MODE:
                        cf.commander.send_hover_setpoint(TRIM_VX, TRIM_VY, 0, 0)
                    log_to_csv()
                    time.sleep(0.01)

                # Stop motors
                if not DEBUG_MODE:
                    cf.commander.send_setpoint(0, 0, 0, 0)
                flight_phase = "JOYSTICK_COMPLETE"

        except Exception as e:
            error_msg = str(e)
            flight_phase = "JOYSTICK_ERROR"
            # Log the error message content to output
            self.log_to_output(f"Joystick Error: {error_msg}")
        finally:
            # Stop logging
            close_csv_logging(logger=self.log_to_output)
            if log_motion:
                try:
                    log_motion.stop()
                except:
                    pass
            if log_battery:
                try:
                    log_battery.stop()
                except:
                    pass
            flight_active = False
            maneuver_active = False
            self.joystick_active = False
            # Update UI in main thread
            self.root.after(
                0, lambda: self.start_joystick_button.config(state=tk.NORMAL)
            )
            # self.root.after(  # Removed duplicate stop button
            #     0, lambda: self.stop_joystick_button.config(state=tk.DISABLED)
            # )
            self.root.after(
                0, lambda: self.joystick_status_var.set("Joystick: INACTIVE")
            )
            self.root.after(
                0, lambda: self.status_var.set("Status: Joystick Control Stopped")
            )

    def on_key_press(self, event):
        """Handle key press events for joystick control"""
        # Emergency stop with Enter or Escape key (works anytime)
        if event.keysym in ("Return", "KP_Enter", "Escape"):
            self.emergency_stop()
            self.log_to_output(
                f"EMERGENCY STOP: {event.keysym} key pressed - all systems stopped"
            )
            return

        if not self.joystick_active:
            return

        key = event.char.lower()
        if key in ["w", "a", "s", "d"]:
            # Only process if this is a new key press (not a repeat)
            if not self.key_pressed_flags[key]:
                self.key_pressed_flags[key] = True

                # Update joystick keys for the control thread
                self.joystick_keys[key] = True

                # Update joystick status to show active keys
                active_keys = [k.upper() for k, v in self.joystick_keys.items() if v]
                if active_keys:
                    self.joystick_status_var.set(
                        f"Joystick: ACTIVE ({','.join(active_keys)})"
                    )
                else:
                    self.joystick_status_var.set("Joystick: ACTIVE")

                # Log continuous movement start (only once per press)
                self.log_to_output(
                    f"Continuous movement: {self._key_to_direction(key)} started"
                )
            else:
                # Key is already pressed, just ensure joystick_keys is set
                self.joystick_keys[key] = True

    def on_key_release(self, event):
        """Handle key release events for joystick control"""
        if not self.joystick_active:
            return

        key = event.char.lower()
        if key in ["w", "a", "s", "d"]:
            # Only process if this key was actually pressed
            if self.key_pressed_flags[key]:
                self.key_pressed_flags[key] = False

                # Clear joystick keys for the control thread
                self.joystick_keys[key] = False

                # Update joystick status to show remaining active keys
                active_keys = [k.upper() for k, v in self.joystick_keys.items() if v]
                if active_keys:
                    self.joystick_status_var.set(
                        f"Joystick: ACTIVE ({','.join(active_keys)})"
                    )
                else:
                    self.joystick_status_var.set("Joystick: ACTIVE")

                # Log continuous movement stop with hold mode info
                # Only show hold mode info when all keys are released
                if not any(self.joystick_keys.values()):
                    mode_info = (
                        " - Returning to origin"
                        if self.joystick_hold_at_origin
                        else " - Holding current position"
                    )
                    self.log_to_output(
                        f"Continuous movement: {self._key_to_direction(key)} stopped{mode_info}"
                    )
                else:
                    self.log_to_output(
                        f"Continuous movement: {self._key_to_direction(key)} stopped"
                    )
            else:
                # Key wasn't pressed, just ensure joystick_keys is cleared
                self.joystick_keys[key] = False

    def simulate_key_press(self, key):
        """Simulate a key press for joystick control (used by GUI buttons)"""
        if not self.joystick_active:
            return

        key = key.lower()
        if key in ["w", "a", "s", "d"]:
            # Temporarily set the key as pressed
            self.joystick_keys[key] = True
            # Update status
            active_keys = [k.upper() for k, v in self.joystick_keys.items() if v]
            self.joystick_status_var.set(f"Joystick: ACTIVE ({','.join(active_keys)})")
            # Schedule key release after a short delay (200ms)
            self.root.after(200, lambda: self.simulate_key_release(key))

            # Log simulated key press to output window
            self.log_to_output(f"Joystick button pressed: {key.upper()}")

    def simulate_key_release(self, key):
        """Simulate a key release for joystick control"""
        if not self.joystick_active:
            return

        key = key.lower()
        if key in ["w", "a", "s", "d"]:
            self.joystick_keys[key] = False
            # Update status
            active_keys = [k.upper() for k, v in self.joystick_keys.items() if v]
            if active_keys:
                self.joystick_status_var.set(
                    f"Joystick: ACTIVE ({','.join(active_keys)})"
                )
            else:
                self.joystick_status_var.set("Joystick: ACTIVE")

            # Log simulated key release to output window
            self.log_to_output(f"Joystick button released: {key.upper()}")


def main():
    # Initialize CRTP drivers once at program start so GUI callbacks
    # can open Crazyflie links later (required by some cflib versions).
    try:
        cflib.crtp.init_drivers()
        print("cflib.crtp drivers initialized")
    except Exception as e:
        print(f"Warning: cflib.crtp.init_drivers() failed: {e}")

    root = tk.Tk()
    app = DeadReckoningGUI(root)

    def on_closing():
        global flight_active, sensor_test_active
        flight_active = False
        sensor_test_active = False
        # Try to stop/clear NeoPixel using existing Crazyflie link if available
        try:
            cf = getattr(scf_instance, "cf", None)
            if cf is not None:
                try:
                    np_stop_blink(cf)
                    np_clear(cf)
                except Exception:
                    pass
        except Exception:
            pass
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
