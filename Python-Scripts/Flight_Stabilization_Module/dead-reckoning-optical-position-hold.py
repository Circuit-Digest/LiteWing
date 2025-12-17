"""
Dead Reckoning Optical Position Hold - Standalone Script
=========================================================
A simplified standalone script for optical flow-based position hold.
Uses dead reckoning with PID controllers to maintain the drone's position
after takeoff.

This script is designed for learners to understand the core concepts
of position hold using optical flow sensors without the complexity
of the full master script.

Features:
- Sensor Test to arm/prepare the drone
- Optical flow-based position hold
- Adjustable TRIM values, Height, and PID parameters
- Real-time position, velocity, and correction feedback
- Safety checks for battery and sensors

Usage:
1. Connect to the drone via UDP
2. Click "Sensor Test" to arm and verify sensors
3. Adjust parameters if needed
4. Click "Start Position Hold" to takeoff and hover
5. The drone will maintain its position automatically
6. Click "Stop" or press Enter for emergency stop

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
import csv
from datetime import datetime

# === CONFIGURATION PARAMETERS ===
DRONE_URI = "udp://192.168.43.42"
TARGET_HEIGHT = 0.3  # Target hover height in meters
TAKEOFF_TIME = 0.5  # Time to takeoff and stabilize
HOVER_DURATION = 60.0  # How long to hover with position hold (seconds)
LANDING_TIME = 0.5  # Time to land
DEBUG_MODE = False  # Set to True to disable motors (sensors and logging still work)

# Velocity and control parameters
VELOCITY_SMOOTHING_ALPHA = 0.9  # Filtering strength for velocity smoothing
VELOCITY_THRESHOLD = 0.005  # Consider drone "stationary" below this velocity
CONTROL_UPDATE_RATE = 0.02  # 50Hz control loop
SENSOR_PERIOD_MS = 10  # Motion sensor update rate
DT = SENSOR_PERIOD_MS / 1000.0

# Basic trim corrections (adjustable via UI)
TRIM_VX = 0.0  # Forward/backward trim correction
TRIM_VY = 0.0  # Left/right trim correction

# Battery monitoring
LOW_BATTERY_THRESHOLD = 2.9  # Low battery warning threshold in volts

# Height sensor safety
HEIGHT_SENSOR_MIN_CHANGE = 0.015  # Minimum height change expected during takeoff

# === DEAD RECKONING POSITION CONTROL PARAMETERS ===
# PID Controller Parameters (adjustable via UI)
POSITION_KP = 1.2
POSITION_KI = 0.0
POSITION_KD = 0.0
VELOCITY_KP = 1.2
VELOCITY_KI = 0.0
VELOCITY_KD = 0.0

# Control limits
MAX_CORRECTION = 0.1  # Maximum control correction allowed
DRIFT_COMPENSATION_RATE = 0.004  # Gentle pull toward zero when moving slowly
MAX_POSITION_ERROR = 2.0  # Clamp position error to prevent runaway
PERIODIC_RESET_INTERVAL = 90.0  # Reset integrated position periodically

# Velocity calculation constants
DEG_TO_RAD = 3.1415926535 / 180.0
OPTICAL_FLOW_SCALE = 4.4  # Empirical scaling factor
USE_HEIGHT_SCALING = True

# === GLOBAL VARIABLES ===
# Sensor data
current_height = 0.0
motion_delta_x = 0
motion_delta_y = 0
sensor_data_ready = False

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
position_integration_enabled = False

# Control corrections
current_correction_vx = 0.0
current_correction_vy = 0.0

# PID Controller state variables
position_integral_x = 0.0
position_integral_y = 0.0
last_position_error_x = 0.0
last_position_error_y = 0.0
velocity_integral_x = 0.0
velocity_integral_y = 0.0
last_velocity_error_x = 0.0
last_velocity_error_y = 0.0

# Target position for position hold
target_position_x = 0.0
target_position_y = 0.0

# Flight state
flight_phase = "IDLE"
flight_active = False
sensor_test_active = False
scf_instance = None

# Data history for plotting
max_history_points = 200
time_history = []
velocity_x_history_plot = []
velocity_y_history_plot = []
position_x_history = []
position_y_history = []
correction_vx_history = []
correction_vy_history = []
height_history = []
complete_trajectory_x = []
complete_trajectory_y = []
start_time = None

# CSV logging
log_file = None
log_writer = None


# === HELPER FUNCTIONS ===
def calculate_velocity(delta_value, altitude):
    """Convert optical flow delta to linear velocity"""
    if altitude <= 0:
        return 0.0
    if USE_HEIGHT_SCALING:
        velocity_constant = (4.4 * DEG_TO_RAD) / (30.0 * DT)
        velocity = delta_value * altitude * velocity_constant
    else:
        velocity = delta_value * OPTICAL_FLOW_SCALE * DT
    return velocity


def smooth_velocity(new_velocity, history):
    """Simple 2-point smoothing filter with adjustable alpha"""
    history[1] = history[0]
    history[0] = new_velocity
    alpha = VELOCITY_SMOOTHING_ALPHA
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
    integrated_position_x = max(-MAX_POSITION_ERROR, min(MAX_POSITION_ERROR, integrated_position_x))
    integrated_position_y = max(-MAX_POSITION_ERROR, min(MAX_POSITION_ERROR, integrated_position_y))


def periodic_position_reset():
    """Reset integrated position periodically to prevent drift accumulation"""
    global integrated_position_x, integrated_position_y, last_reset_time
    current_time = time.time()
    if current_time - last_reset_time >= PERIODIC_RESET_INTERVAL:
        integrated_position_x = 0.0
        integrated_position_y = 0.0
        last_reset_time = current_time
        return True
    return False


def reset_position_tracking():
    """Reset integrated position tracking"""
    global integrated_position_x, integrated_position_y, last_integration_time, last_reset_time
    global position_integration_enabled
    global position_integral_x, position_integral_y, velocity_integral_x, velocity_integral_y
    global last_position_error_x, last_position_error_y, last_velocity_error_x, last_velocity_error_y
    global target_position_x, target_position_y
    
    integrated_position_x = 0.0
    integrated_position_y = 0.0
    target_position_x = 0.0
    target_position_y = 0.0
    last_integration_time = time.time()
    last_reset_time = time.time()
    position_integration_enabled = True
    # Reset PID state
    position_integral_x = 0.0
    position_integral_y = 0.0
    velocity_integral_x = 0.0
    velocity_integral_y = 0.0
    last_position_error_x = 0.0
    last_position_error_y = 0.0
    last_velocity_error_x = 0.0
    last_velocity_error_y = 0.0


def calculate_position_hold_corrections():
    """Calculate control corrections using PID controllers"""
    global current_correction_vx, current_correction_vy
    global position_integral_x, position_integral_y
    global last_position_error_x, last_position_error_y
    global velocity_integral_x, velocity_integral_y
    global last_velocity_error_x, last_velocity_error_y

    if not sensor_data_ready or current_height <= 0:
        current_correction_vx = 0.0
        current_correction_vy = 0.0
        return 0.0, 0.0

    # Calculate position errors (negative because we want to correct toward target)
    position_error_x = -(integrated_position_x - target_position_x)
    position_error_y = -(integrated_position_y - target_position_y)

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
    position_integral_x = max(-0.1, min(0.1, position_integral_x))
    position_integral_y = max(-0.1, min(0.1, position_integral_y))
    position_i_x = position_integral_x * POSITION_KI
    position_i_y = position_integral_y * POSITION_KI
    # Derivative
    position_derivative_x = (position_error_x - last_position_error_x) / CONTROL_UPDATE_RATE
    position_derivative_y = (position_error_y - last_position_error_y) / CONTROL_UPDATE_RATE
    position_d_x = position_derivative_x * POSITION_KD
    position_d_y = position_derivative_y * POSITION_KD
    last_position_error_x = position_error_x
    last_position_error_y = position_error_y

    # Velocity PID Controller
    # Proportional
    velocity_p_x = velocity_error_x * VELOCITY_KP
    velocity_p_y = velocity_error_y * VELOCITY_KP
    # Integral (with anti-windup)
    velocity_integral_x += velocity_error_x * CONTROL_UPDATE_RATE
    velocity_integral_y += velocity_error_y * CONTROL_UPDATE_RATE
    velocity_integral_x = max(-0.05, min(0.05, velocity_integral_x))
    velocity_integral_y = max(-0.05, min(0.05, velocity_integral_y))
    velocity_i_x = velocity_integral_x * VELOCITY_KI
    velocity_i_y = velocity_integral_y * VELOCITY_KI
    # Derivative
    velocity_derivative_x = (velocity_error_x - last_velocity_error_x) / CONTROL_UPDATE_RATE
    velocity_derivative_y = (velocity_error_y - last_velocity_error_y) / CONTROL_UPDATE_RATE
    velocity_d_x = velocity_derivative_x * VELOCITY_KD
    velocity_d_y = velocity_derivative_y * VELOCITY_KD
    last_velocity_error_x = velocity_error_x
    last_velocity_error_y = velocity_error_y

    # Combine PID outputs
    position_correction_vx = position_p_x + position_i_x + position_d_x
    position_correction_vy = position_p_y + position_i_y + position_d_y
    velocity_correction_vx = velocity_p_x + velocity_i_x + velocity_d_x
    velocity_correction_vy = velocity_p_y + velocity_i_y + velocity_d_y

    # Total corrections
    total_vx = position_correction_vx + velocity_correction_vx
    total_vy = position_correction_vy + velocity_correction_vy

    # Apply limits
    total_vx = max(-MAX_CORRECTION, min(MAX_CORRECTION, total_vx))
    total_vy = max(-MAX_CORRECTION, min(MAX_CORRECTION, total_vy))

    # Store for GUI display
    current_correction_vx = total_vx
    current_correction_vy = total_vy
    return total_vx, total_vy


def update_history():
    """Update data history for plotting"""
    global start_time
    if start_time is None:
        start_time = time.time()
    current_time = time.time() - start_time
    
    # Add new data points
    time_history.append(current_time)
    velocity_x_history_plot.append(current_vx)
    velocity_y_history_plot.append(current_vy)
    position_x_history.append(integrated_position_x)
    position_y_history.append(integrated_position_y)
    correction_vx_history.append(current_correction_vx)
    correction_vy_history.append(current_correction_vy)
    height_history.append(current_height)
    complete_trajectory_x.append(integrated_position_x)
    complete_trajectory_y.append(integrated_position_y)
    
    # Trim history to max points
    if len(time_history) > max_history_points:
        time_history.pop(0)
        velocity_x_history_plot.pop(0)
        velocity_y_history_plot.pop(0)
        position_x_history.pop(0)
        position_y_history.pop(0)
        correction_vx_history.pop(0)
        correction_vy_history.pop(0)
        height_history.pop(0)


def motion_callback(timestamp, data, logconf):
    """Motion sensor data callback"""
    global current_height, motion_delta_x, motion_delta_y, sensor_data_ready
    global current_vx, current_vy, last_integration_time

    # Get sensor data
    current_height = data.get("stateEstimate.z", 0)
    motion_delta_x = data.get("motion.deltaX", 0)
    motion_delta_y = data.get("motion.deltaY", 0)
    sensor_data_ready = True

    # Calculate velocities
    raw_velocity_x = calculate_velocity(motion_delta_x, current_height)
    raw_velocity_y = calculate_velocity(motion_delta_y, current_height)

    # Apply smoothing
    current_vx = smooth_velocity(raw_velocity_x, velocity_x_history)
    current_vy = smooth_velocity(raw_velocity_y, velocity_y_history)

    # Dead reckoning position integration
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
    current_battery_voltage = data.get("pm.vbat", 0.0)
    battery_data_ready = True


def setup_logging(cf, logger=None):
    """Setup motion sensor and battery voltage logging"""
    log_motion = LogConfig(name="Motion", period_in_ms=SENSOR_PERIOD_MS)
    log_battery = LogConfig(name="Battery", period_in_ms=500)

    try:
        toc = cf.log.toc.toc
        # Setup motion logging
        motion_variables = [
            ("motion.deltaX", "int16_t"),
            ("motion.deltaY", "int16_t"),
            ("stateEstimate.z", "float"),
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
                if logger:
                    logger(f"Motion variable not found: {var_name}")

        if len(added_motion_vars) < 2:
            if logger:
                logger("ERROR: Not enough motion variables found!")
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
                except Exception as e:
                    if logger:
                        logger(f"Failed to add battery variable {var_name}: {e}")
            else:
                if logger:
                    logger(f"Battery variable not found: {var_name}")

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
            return None, None
        if len(added_battery_vars) > 0 and not log_battery.valid:
            if logger:
                logger("WARNING: Battery log configuration invalid!")
            log_battery = None

        # Start logging
        log_motion.start()
        if log_battery:
            log_battery.start()

        time.sleep(0.5)
        if logger:
            logger(f"Logging started - Motion: {len(added_motion_vars)} vars, Battery: {len(added_battery_vars)} vars")
        return log_motion, log_battery

    except Exception as e:
        error_msg = f"Logging setup failed: {str(e)}"
        if logger:
            logger(error_msg)
        raise Exception(error_msg)


def init_csv_logging(logger=None):
    """Initialize CSV logging"""
    global log_file, log_writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"position_hold_log_{timestamp}.csv"
    log_file = open(log_filename, mode="w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "Timestamp (s)", "Position X (m)", "Position Y (m)",
        "Height (m)", "Velocity X (m/s)", "Velocity Y (m/s)",
        "Correction VX", "Correction VY"
    ])
    if logger:
        logger(f"Logging to CSV: {log_filename}")


def log_to_csv():
    """Log current state to CSV"""
    global log_writer, start_time
    if log_writer is None or start_time is None:
        return
    elapsed = time.time() - start_time
    log_writer.writerow([
        f"{elapsed:.3f}", f"{integrated_position_x:.6f}", f"{integrated_position_y:.6f}",
        f"{current_height:.6f}", f"{current_vx:.6f}", f"{current_vy:.6f}",
        f"{current_correction_vx:.6f}", f"{current_correction_vy:.6f}"
    ])


def close_csv_logging(logger=None):
    """Close CSV log file"""
    global log_file
    if log_file:
        log_file.close()
        log_file = None
        if logger:
            logger("CSV log closed.")


class PositionHoldGUI:
    """
    GUI for Optical Flow Position Hold
    
    This class provides a user interface for testing optical flow-based
    position hold with adjustable PID parameters.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Dead Reckoning Optical Position Hold")
        self.root.geometry("1200x800")

        # Control variables
        self.flight_thread = None
        self.flight_running = False
        self.sensor_test_thread = None
        self.sensor_test_running = False

        self.create_ui()
        self.setup_plots()

        # Start animation
        self.anim = animation.FuncAnimation(
            self.fig, self.update_plots, interval=100, cache_frame_data=False
        )

        # Bind keyboard events
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.focus_set()

    def create_ui(self):
        """Create the user interface"""
        # Control panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Sensor Test button
        self.sensor_test_button = tk.Button(
            control_frame, text="Sensor Test (ARM)",
            command=self.start_sensor_test,
            bg="lightblue", fg="black", font=("Arial", 11, "bold"), width=16
        )
        self.sensor_test_button.pack(side=tk.LEFT, padx=5)

        # Start Position Hold button
        self.start_button = tk.Button(
            control_frame, text="Start Position Hold",
            command=self.start_flight,
            bg="green", fg="white", font=("Arial", 11, "bold"), width=16
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Debug mode checkbox
        self.debug_mode_var = tk.BooleanVar(value=DEBUG_MODE)
        tk.Checkbutton(
            control_frame, text="Debug Mode", variable=self.debug_mode_var,
            command=self.toggle_debug_mode, font=("Arial", 9)
        ).pack(side=tk.LEFT, padx=10)

        # Status display
        self.status_var = tk.StringVar(value="Status: Ready - Run Sensor Test to ARM")
        tk.Label(
            control_frame, textvariable=self.status_var,
            font=("Arial", 11, "bold"), fg="blue"
        ).pack(side=tk.LEFT, padx=20)

        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left side - Controls
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        # === Flight Parameters Frame ===
        params_frame = tk.LabelFrame(left_frame, text="Flight Parameters", padx=10, pady=10)
        params_frame.pack(fill=tk.X, pady=5)

        # Target Height
        height_frame = tk.Frame(params_frame)
        height_frame.pack(fill=tk.X, pady=2)
        tk.Label(height_frame, text="Target Height (m):", width=16, anchor=tk.W).pack(side=tk.LEFT)
        self.height_entry_var = tk.StringVar(value=str(TARGET_HEIGHT))
        tk.Entry(height_frame, textvariable=self.height_entry_var, width=8).pack(side=tk.LEFT, padx=5)

        # Hover Duration
        duration_frame = tk.Frame(params_frame)
        duration_frame.pack(fill=tk.X, pady=2)
        tk.Label(duration_frame, text="Hover Duration (s):", width=16, anchor=tk.W).pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value=str(HOVER_DURATION))
        tk.Entry(duration_frame, textvariable=self.duration_var, width=8).pack(side=tk.LEFT, padx=5)

        # TRIM VX
        trim_vx_frame = tk.Frame(params_frame)
        trim_vx_frame.pack(fill=tk.X, pady=2)
        tk.Label(trim_vx_frame, text="TRIM VX:", width=16, anchor=tk.W).pack(side=tk.LEFT)
        self.trim_vx_var = tk.StringVar(value=str(TRIM_VX))
        tk.Entry(trim_vx_frame, textvariable=self.trim_vx_var, width=8).pack(side=tk.LEFT, padx=5)

        # TRIM VY
        trim_vy_frame = tk.Frame(params_frame)
        trim_vy_frame.pack(fill=tk.X, pady=2)
        tk.Label(trim_vy_frame, text="TRIM VY:", width=16, anchor=tk.W).pack(side=tk.LEFT)
        self.trim_vy_var = tk.StringVar(value=str(TRIM_VY))
        tk.Entry(trim_vy_frame, textvariable=self.trim_vy_var, width=8).pack(side=tk.LEFT, padx=5)

        # === PID Parameters Frame ===
        pid_frame = tk.LabelFrame(left_frame, text="PID Parameters", padx=10, pady=10)
        pid_frame.pack(fill=tk.X, pady=5)

        # Container for two columns
        pid_columns = tk.Frame(pid_frame)
        pid_columns.pack(fill=tk.X)

        # Left column - Position PID
        pos_column = tk.Frame(pid_columns)
        pos_column.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        tk.Label(pos_column, text="Position PID:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        pos_kp_frame = tk.Frame(pos_column)
        pos_kp_frame.pack(fill=tk.X, pady=1)
        tk.Label(pos_kp_frame, text="Kp:", width=4, anchor=tk.W).pack(side=tk.LEFT)
        self.pos_kp_var = tk.StringVar(value=str(POSITION_KP))
        tk.Entry(pos_kp_frame, textvariable=self.pos_kp_var, width=6).pack(side=tk.LEFT)

        pos_ki_frame = tk.Frame(pos_column)
        pos_ki_frame.pack(fill=tk.X, pady=1)
        tk.Label(pos_ki_frame, text="Ki:", width=4, anchor=tk.W).pack(side=tk.LEFT)
        self.pos_ki_var = tk.StringVar(value=str(POSITION_KI))
        tk.Entry(pos_ki_frame, textvariable=self.pos_ki_var, width=6).pack(side=tk.LEFT)

        pos_kd_frame = tk.Frame(pos_column)
        pos_kd_frame.pack(fill=tk.X, pady=1)
        tk.Label(pos_kd_frame, text="Kd:", width=4, anchor=tk.W).pack(side=tk.LEFT)
        self.pos_kd_var = tk.StringVar(value=str(POSITION_KD))
        tk.Entry(pos_kd_frame, textvariable=self.pos_kd_var, width=6).pack(side=tk.LEFT)

        # Right column - Velocity PID
        vel_column = tk.Frame(pid_columns)
        vel_column.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(vel_column, text="Velocity PID:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        vel_kp_frame = tk.Frame(vel_column)
        vel_kp_frame.pack(fill=tk.X, pady=1)
        tk.Label(vel_kp_frame, text="Kp:", width=4, anchor=tk.W).pack(side=tk.LEFT)
        self.vel_kp_var = tk.StringVar(value=str(VELOCITY_KP))
        tk.Entry(vel_kp_frame, textvariable=self.vel_kp_var, width=6).pack(side=tk.LEFT)

        vel_ki_frame = tk.Frame(vel_column)
        vel_ki_frame.pack(fill=tk.X, pady=1)
        tk.Label(vel_ki_frame, text="Ki:", width=4, anchor=tk.W).pack(side=tk.LEFT)
        self.vel_ki_var = tk.StringVar(value=str(VELOCITY_KI))
        tk.Entry(vel_ki_frame, textvariable=self.vel_ki_var, width=6).pack(side=tk.LEFT)

        vel_kd_frame = tk.Frame(vel_column)
        vel_kd_frame.pack(fill=tk.X, pady=1)
        tk.Label(vel_kd_frame, text="Kd:", width=4, anchor=tk.W).pack(side=tk.LEFT)
        self.vel_kd_var = tk.StringVar(value=str(VELOCITY_KD))
        tk.Entry(vel_kd_frame, textvariable=self.vel_kd_var, width=6).pack(side=tk.LEFT)

        # Apply button (full width below columns)
        tk.Button(
            pid_frame, text="Apply All Values", command=self.apply_values,
            bg="green", fg="white", font=("Arial", 10)
        ).pack(pady=5)

        # === Real-time values display ===
        values_frame = tk.LabelFrame(left_frame, text="Real-Time Values", padx=10, pady=10)
        values_frame.pack(fill=tk.X, pady=5)

        self.height_var = tk.StringVar(value="Height: 0.000m")
        self.battery_var = tk.StringVar(value="Battery: N/A")
        self.vx_var = tk.StringVar(value="VX: 0.000 m/s")
        self.vy_var = tk.StringVar(value="VY: 0.000 m/s")
        self.pos_x_var = tk.StringVar(value="Pos X: 0.000m")
        self.pos_y_var = tk.StringVar(value="Pos Y: 0.000m")
        self.corr_vx_var = tk.StringVar(value="Corr VX: 0.000")
        self.corr_vy_var = tk.StringVar(value="Corr VY: 0.000")
        self.phase_var = tk.StringVar(value="Phase: IDLE")

        tk.Label(values_frame, textvariable=self.height_var, font=("Arial", 10), fg="blue").pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.battery_var, font=("Arial", 10), fg="orange").pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.vx_var, font=("Arial", 10)).pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.vy_var, font=("Arial", 10)).pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.pos_x_var, font=("Arial", 10), fg="darkgreen").pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.pos_y_var, font=("Arial", 10), fg="darkgreen").pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.corr_vx_var, font=("Arial", 10), fg="red").pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.corr_vy_var, font=("Arial", 10), fg="red").pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.phase_var, font=("Arial", 10, "bold"), fg="purple").pack(anchor=tk.W)

        # === Output log ===
        output_frame = tk.LabelFrame(left_frame, text="Output Log", padx=5, pady=5)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.output_text = tk.Text(output_frame, height=8, width=40, font=("Consolas", 9))
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Right side - Plots
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.fig = Figure(figsize=(9, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_plots(self):
        """Setup matplotlib plots"""
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Velocities
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Position (2D)
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Corrections
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Height

        # Velocities
        self.ax1.set_title("Velocities")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Velocity (m/s)")
        self.ax1.grid(True, alpha=0.3)
        (self.line_vx,) = self.ax1.plot([], [], "b-", label="VX")
        (self.line_vy,) = self.ax1.plot([], [], "r-", label="VY")
        self.ax1.legend()

        # 2D Position
        self.ax2.set_title("Position (Dead Reckoning)")
        self.ax2.set_xlabel("X (m)")
        self.ax2.set_ylabel("Y (m)")
        self.ax2.set_aspect("equal")
        self.ax2.grid(True, alpha=0.3)
        (self.line_pos,) = self.ax2.plot([], [], "purple", alpha=0.7, label="Trajectory")
        (self.current_pos,) = self.ax2.plot([], [], "ro", markersize=8, label="Current")
        self.ax2.plot(0, 0, "ko", markersize=10, markerfacecolor="yellow", label="Target")
        self.ax2.legend()

        # Corrections
        self.ax3.set_title("Control Corrections")
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Correction")
        self.ax3.grid(True, alpha=0.3)
        (self.line_corr_vx,) = self.ax3.plot([], [], "g-", label="Corr VX")
        (self.line_corr_vy,) = self.ax3.plot([], [], "m-", label="Corr VY")
        self.ax3.legend()

        # Height
        self.ax4.set_title("Height")
        self.ax4.set_xlabel("Time (s)")
        self.ax4.set_ylabel("Height (m)")
        self.ax4.grid(True, alpha=0.3)
        (self.line_height,) = self.ax4.plot([], [], "orange", label="Height")
        self.target_height_line = self.ax4.axhline(y=TARGET_HEIGHT, color="red", linestyle="--", alpha=0.7, label="Target")
        self.ax4.legend()

        self.fig.tight_layout()

    def update_plots(self, frame):
        """Update all plots with new data"""
        if not time_history:
            return []

        # Update value displays
        self.height_var.set(f"Height: {current_height:.3f}m")
        self.phase_var.set(f"Phase: {flight_phase}")
        if current_battery_voltage > 0:
            self.battery_var.set(f"Battery: {current_battery_voltage:.2f}V")
        self.vx_var.set(f"VX: {current_vx:.3f} m/s")
        self.vy_var.set(f"VY: {current_vy:.3f} m/s")
        self.pos_x_var.set(f"Pos X: {integrated_position_x:.3f}m")
        self.pos_y_var.set(f"Pos Y: {integrated_position_y:.3f}m")
        self.corr_vx_var.set(f"Corr VX: {current_correction_vx:.3f}")
        self.corr_vy_var.set(f"Corr VY: {current_correction_vy:.3f}")

        # Update plots
        try:
            self.line_vx.set_data(time_history, velocity_x_history_plot)
            self.line_vy.set_data(time_history, velocity_y_history_plot)

            if complete_trajectory_x and complete_trajectory_y:
                plot_x = [-x for x in complete_trajectory_x]
                self.line_pos.set_data(plot_x, complete_trajectory_y)
                self.current_pos.set_data([-integrated_position_x], [integrated_position_y])

            self.line_corr_vx.set_data(time_history, correction_vx_history)
            self.line_corr_vy.set_data(time_history, correction_vy_history)
            self.line_height.set_data(time_history, height_history)

            # Adjust axis limits
            if len(time_history) > 1:
                for ax in [self.ax1, self.ax3, self.ax4]:
                    ax.set_xlim(min(time_history), max(time_history))

                if velocity_x_history_plot and velocity_y_history_plot:
                    all_vel = velocity_x_history_plot + velocity_y_history_plot
                    if any(v != 0 for v in all_vel):
                        self.ax1.set_ylim(min(all_vel) - 0.01, max(all_vel) + 0.01)

                if complete_trajectory_x and complete_trajectory_y:
                    plot_x = [-x for x in complete_trajectory_x]
                    margin = max(max(plot_x) - min(plot_x), max(complete_trajectory_y) - min(complete_trajectory_y), 0.02) * 0.6
                    center_x = (max(plot_x) + min(plot_x)) / 2
                    center_y = (max(complete_trajectory_y) + min(complete_trajectory_y)) / 2
                    self.ax2.set_xlim(center_x - margin, center_x + margin)
                    self.ax2.set_ylim(center_y - margin, center_y + margin)

                if correction_vx_history and correction_vy_history:
                    all_corr = correction_vx_history + correction_vy_history
                    if any(c != 0 for c in all_corr):
                        self.ax3.set_ylim(min(all_corr) - 0.01, max(all_corr) + 0.01)

                if height_history:
                    self.ax4.set_ylim(min(height_history) - 0.05, max(height_history) + 0.05)

        except Exception:
            pass

        return [self.line_vx, self.line_vy, self.line_pos, self.current_pos,
                self.line_corr_vx, self.line_corr_vy, self.line_height]

    def log_to_output(self, message):
        """Log a message to the output window"""
        timestamp = time.strftime("%H:%M:%S")
        self.output_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.output_text.see(tk.END)

    def clear_output(self):
        """Clear the output log"""
        self.output_text.delete(1.0, tk.END)

    def clear_graphs(self):
        """Clear all graph data"""
        global time_history, velocity_x_history_plot, velocity_y_history_plot
        global position_x_history, position_y_history, height_history
        global correction_vx_history, correction_vy_history
        global complete_trajectory_x, complete_trajectory_y, start_time

        time_history.clear()
        velocity_x_history_plot.clear()
        velocity_y_history_plot.clear()
        position_x_history.clear()
        position_y_history.clear()
        correction_vx_history.clear()
        correction_vy_history.clear()
        height_history.clear()
        complete_trajectory_x.clear()
        complete_trajectory_y.clear()
        start_time = None

    def apply_values(self):
        """Apply all parameter values from UI"""
        global TARGET_HEIGHT, HOVER_DURATION, TRIM_VX, TRIM_VY
        global POSITION_KP, POSITION_KI, POSITION_KD
        global VELOCITY_KP, VELOCITY_KI, VELOCITY_KD
        
        try:
            TARGET_HEIGHT = float(self.height_entry_var.get())
            HOVER_DURATION = float(self.duration_var.get())
            TRIM_VX = float(self.trim_vx_var.get())
            TRIM_VY = float(self.trim_vy_var.get())
            POSITION_KP = float(self.pos_kp_var.get())
            POSITION_KI = float(self.pos_ki_var.get())
            POSITION_KD = float(self.pos_kd_var.get())
            VELOCITY_KP = float(self.vel_kp_var.get())
            VELOCITY_KI = float(self.vel_ki_var.get())
            VELOCITY_KD = float(self.vel_kd_var.get())
            
            # Update height line in plot
            self.target_height_line.set_ydata([TARGET_HEIGHT, TARGET_HEIGHT])
            
            self.log_to_output(f"Applied: Height={TARGET_HEIGHT:.2f}m, Duration={HOVER_DURATION:.0f}s")
            self.log_to_output(f"TRIM: VX={TRIM_VX:.2f}, VY={TRIM_VY:.2f}")
            self.log_to_output(f"Pos PID: Kp={POSITION_KP}, Ki={POSITION_KI}, Kd={POSITION_KD}")
            self.log_to_output(f"Vel PID: Kp={VELOCITY_KP}, Ki={VELOCITY_KI}, Kd={VELOCITY_KD}")
            self.status_var.set("Status: Values applied")
        except ValueError as e:
            self.status_var.set(f"Status: Invalid value - {str(e)}")
            self.log_to_output(f"Error: {str(e)}")

    def toggle_debug_mode(self):
        """Toggle debug mode"""
        global DEBUG_MODE
        DEBUG_MODE = self.debug_mode_var.get()
        mode_text = "ON (motors disabled)" if DEBUG_MODE else "OFF"
        self.log_to_output(f"Debug mode: {mode_text}")

    # ==================== SENSOR TEST ====================
    def start_sensor_test(self):
        """Start sensor test to ARM the drone"""
        if not self.sensor_test_running and not self.flight_running:
            self.sensor_test_running = True
            self.sensor_test_button.config(text="Stop Sensor Test", command=self.stop_sensor_test, bg="red")
            self.status_var.set("Status: Sensor Test Running - ARMING...")
            self.sensor_test_thread = threading.Thread(target=self.sensor_test_thread_func)
            self.sensor_test_thread.daemon = True
            self.sensor_test_thread.start()
        elif self.flight_running:
            self.status_var.set("Status: Cannot run Sensor Test during flight")

    def stop_sensor_test(self):
        """Stop sensor test"""
        global sensor_test_active
        if self.sensor_test_running:
            sensor_test_active = False
            self.sensor_test_running = False
            if self.sensor_test_thread and self.sensor_test_thread.is_alive():
                self.sensor_test_thread.join(timeout=2.0)
            self.status_var.set("Status: Sensor Test Stopped - ARMED ✓")
            self.sensor_test_button.config(text="Sensor Test (ARM)", command=self.start_sensor_test, bg="lightblue")

    def sensor_test_thread_func(self):
        """Sensor test thread"""
        global flight_phase, sensor_test_active, scf_instance
        global current_battery_voltage, battery_data_ready

        self.root.after(0, self.clear_output)
        self.root.after(0, self.clear_graphs)

        sensor_test_active = True
        flight_phase = "SENSOR_TEST"

        current_battery_voltage = 0.0
        battery_data_ready = False

        cflib.crtp.init_drivers()
        cf = Crazyflie(rw_cache="./cache")
        log_motion = None
        log_battery = None

        try:
            with SyncCrazyflie(DRONE_URI, cf=cf) as scf:
                scf_instance = scf

                log_motion, log_battery = setup_logging(cf, logger=self.log_to_output)
                if log_motion is not None:
                    time.sleep(1.0)

                if not DEBUG_MODE:
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.1)
                    cf.param.set_value("commander.enHighLevel", "1")
                    time.sleep(0.5)

                reset_position_tracking()

                self.log_to_output("Sensor test running - reading sensors...")
                while sensor_test_active:
                    if sensor_data_ready:
                        calculate_position_hold_corrections()
                    time.sleep(CONTROL_UPDATE_RATE)

        except Exception as e:
            flight_phase = "ERROR"
            self.log_to_output(f"Sensor Test Error: {str(e)}")
        finally:
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
            sensor_test_active = False
            flight_phase = "IDLE"
            self.sensor_test_running = False
            self.root.after(0, lambda: self.sensor_test_button.config(
                text="Sensor Test (ARM)", command=self.start_sensor_test, bg="lightblue"
            ))
            self.root.after(0, lambda: self.status_var.set("Status: Sensor Test Stopped - ARMED ✓"))

    # ==================== POSITION HOLD FLIGHT ====================
    def start_flight(self):
        """Start position hold flight"""
        if self.flight_running or self.sensor_test_running:
            if self.sensor_test_running:
                self.status_var.set("Status: Stop Sensor Test first!")
            return

        # Apply current values
        self.apply_values()

        # Safety checks
        if current_battery_voltage > 0 and current_battery_voltage < LOW_BATTERY_THRESHOLD:
            self.status_var.set(f"Status: Battery too low ({current_battery_voltage:.2f}V)!")
            return
        elif current_battery_voltage == 0.0:
            self.log_to_output("WARNING: Battery unknown - run Sensor Test first!")
            self.status_var.set("Status: Run Sensor Test first!")
            return

        if not sensor_data_ready:
            self.status_var.set("Status: Sensor data not ready - Run Sensor Test first!")
            return

        if current_height <= 0.0:
            self.status_var.set("Status: Invalid height reading!")
            return

        # Start flight
        self.flight_running = True
        self.start_button.config(text="Stop Flight", command=self.emergency_stop, bg="red")
        self.status_var.set("Status: Starting Position Hold Flight...")
        self.log_to_output("Position Hold flight started")

        self.flight_thread = threading.Thread(target=self.flight_thread_func)
        self.flight_thread.daemon = True
        self.flight_thread.start()

    def emergency_stop(self):
        """Emergency stop"""
        global flight_active, sensor_test_active
        flight_active = False
        sensor_test_active = False
        self.flight_running = False
        self.sensor_test_running = False

        self.start_button.config(text="Start Position Hold", command=self.start_flight, bg="green")
        self.sensor_test_button.config(text="Sensor Test (ARM)", command=self.start_sensor_test, bg="lightblue")
        self.status_var.set("Status: EMERGENCY STOPPED")
        self.log_to_output("EMERGENCY STOP!")

    def flight_thread_func(self):
        """Position hold flight thread"""
        global flight_active, flight_phase
        global current_battery_voltage, battery_data_ready
        global position_integration_enabled

        self.root.after(0, self.clear_output)
        self.root.after(0, self.clear_graphs)

        cflib.crtp.init_drivers()
        cf = Crazyflie(rw_cache="./cache")
        log_motion = None
        log_battery = None

        current_battery_voltage = 0.0
        battery_data_ready = False

        try:
            with SyncCrazyflie(DRONE_URI, cf=cf) as scf:
                scf_instance = scf
                flight_active = True

                log_motion, log_battery = setup_logging(cf, logger=self.log_to_output)
                use_position_hold = log_motion is not None
                if use_position_hold:
                    time.sleep(1.0)

                reset_position_tracking()

                if not DEBUG_MODE:
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.1)
                    cf.param.set_value("commander.enHighLevel", "1")
                    time.sleep(0.5)
                else:
                    self.log_to_output("DEBUG MODE: Motors disabled")

                # Takeoff
                flight_phase = "TAKEOFF"
                start_time_local = time.time()
                init_csv_logging(logger=self.log_to_output)

                while time.time() - start_time_local < TAKEOFF_TIME and flight_active:
                    if not DEBUG_MODE:
                        cf.commander.send_hover_setpoint(TRIM_VX, TRIM_VY, 0, TARGET_HEIGHT)
                    log_to_csv()
                    time.sleep(0.01)

                # Stabilization
                flight_phase = "STABILIZING"
                stabilization_start = time.time()
                while time.time() - stabilization_start < 3.0 and flight_active:
                    if use_position_hold and sensor_data_ready:
                        motion_vx, motion_vy = calculate_position_hold_corrections()
                    else:
                        motion_vx, motion_vy = 0.0, 0.0
                    log_to_csv()
                    total_vx = TRIM_VX + motion_vy
                    total_vy = TRIM_VY + motion_vx
                    if not DEBUG_MODE:
                        cf.commander.send_hover_setpoint(total_vx, total_vy, 0, TARGET_HEIGHT)
                    time.sleep(CONTROL_UPDATE_RATE)

                # Position Hold
                flight_phase = "POSITION_HOLD"
                hover_start = time.time()
                self.log_to_output(f"Position Hold active for {HOVER_DURATION:.0f}s")

                while time.time() - hover_start < HOVER_DURATION and flight_active:
                    if use_position_hold and sensor_data_ready:
                        motion_vx, motion_vy = calculate_position_hold_corrections()
                        if periodic_position_reset():
                            flight_phase = "POSITION_HOLD (RESET)"
                            self.log_to_output("Position reset to origin")
                        else:
                            flight_phase = "POSITION_HOLD"
                    else:
                        motion_vx, motion_vy = 0.0, 0.0

                    log_to_csv()
                    total_vx = TRIM_VX + motion_vy
                    total_vy = TRIM_VY + motion_vx
                    if not DEBUG_MODE:
                        cf.commander.send_hover_setpoint(total_vx, total_vy, 0, TARGET_HEIGHT)
                    time.sleep(CONTROL_UPDATE_RATE)

                # Landing
                flight_phase = "LANDING"
                landing_start = time.time()
                while time.time() - landing_start < LANDING_TIME and flight_active:
                    if not DEBUG_MODE:
                        cf.commander.send_hover_setpoint(TRIM_VX, TRIM_VY, 0, 0)
                    log_to_csv()
                    time.sleep(0.01)

                if not DEBUG_MODE:
                    cf.commander.send_setpoint(0, 0, 0, 0)
                flight_phase = "COMPLETE"
                self.log_to_output("Flight complete")

        except Exception as e:
            flight_phase = "ERROR"
            self.log_to_output(f"Error: {str(e)}")
        finally:
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
            self.root.after(0, lambda: self.start_button.config(
                text="Start Position Hold", command=self.start_flight, bg="green"
            ))
            self.root.after(0, lambda: self.status_var.set("Status: Flight Complete"))

    def on_key_press(self, event):
        """Handle key press events"""
        if event.keysym in ("Return", "KP_Enter"):
            self.emergency_stop()
            self.log_to_output("EMERGENCY STOP: Enter key pressed")


def main():
    """Main entry point"""
    try:
        cflib.crtp.init_drivers()
        print("Crazyflie CRTP drivers initialized")
    except Exception as e:
        print(f"Warning: cflib.crtp.init_drivers() failed: {e}")

    root = tk.Tk()
    app = PositionHoldGUI(root)

    def on_closing():
        global flight_active, sensor_test_active
        flight_active = False
        sensor_test_active = False
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
