"""
Dead Reckoning Joystick Control - Standalone Script
=====================================================
A simplified standalone script for joystick-based drone control.
This script is designed for learners to understand the core concepts
of joystick control without the complexity of the full master script.

Features:
- Sensor Test to arm/prepare the drone (required before joystick control)
- WASD keyboard control for drone movement
- Adjustable TRIM values (VX and VY)
- Adjustable target height
- Real-time velocity and position feedback
- Safety checks for battery and sensors

Usage:
1. Connect to the drone via UDP
2. Click "Sensor Test" to arm and verify sensors
3. Adjust TRIM and Height values if needed
4. Click "Start Joystick Control" button
5. Use W/A/S/D keys or GUI buttons to move the drone
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
TARGET_HEIGHT = 0.4  # Target hover height in meters
TAKEOFF_TIME = 0.5  # Time to takeoff and stabilize
LANDING_TIME = 0.5  # Time to land
DEBUG_MODE = False  # Set to True to disable motors (sensors and logging still work)

# Velocity and control parameters
VELOCITY_SMOOTHING_ALPHA = 1.5  # Filtering strength for velocity smoothing
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
HEIGHT_SENSOR_MIN_CHANGE = 0.015  # Minimum height change expected during takeoff (meters)

# Velocity calculation constants
DEG_TO_RAD = 3.1415926535 / 180.0
OPTICAL_FLOW_SCALE = 4.4  # Empirical scaling factor
USE_HEIGHT_SCALING = True

# Joystick parameters
JOYSTICK_SENSITIVITY = 0.9  # Default joystick sensitivity (0.1-2.0)

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

# Dead reckoning position integration (for display only)
integrated_position_x = 0.0
integrated_position_y = 0.0
last_integration_time = time.time()
position_integration_enabled = False

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
        velocity_constant = (5.4 * DEG_TO_RAD) / (30.0 * DT)
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
    """Dead reckoning: integrate velocity to position (for display only)"""
    global integrated_position_x, integrated_position_y
    if dt <= 0 or dt > 0.1:
        return
    integrated_position_x += vx * dt
    integrated_position_y += vy * dt


def reset_position_tracking():
    """Reset integrated position tracking"""
    global integrated_position_x, integrated_position_y, last_integration_time
    global position_integration_enabled
    integrated_position_x = 0.0
    integrated_position_y = 0.0
    last_integration_time = time.time()
    position_integration_enabled = True


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

    # Dead reckoning position integration (for display only)
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
    """Initialize CSV logging for position and height"""
    global log_file, log_writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"joystick_flight_log_{timestamp}.csv"
    log_file = open(log_filename, mode="w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "Timestamp (s)", "Position X (m)", "Position Y (m)",
        "Height (m)", "Velocity X (m/s)", "Velocity Y (m/s)"
    ])
    if logger:
        logger(f"Logging to CSV: {log_filename}")


def log_to_csv():
    """Log current state to CSV if logging is active"""
    global log_writer, start_time
    if log_writer is None or start_time is None:
        return
    elapsed = time.time() - start_time
    log_writer.writerow([
        f"{elapsed:.3f}", f"{integrated_position_x:.6f}", f"{integrated_position_y:.6f}",
        f"{current_height:.6f}", f"{current_vx:.6f}", f"{current_vy:.6f}"
    ])


def close_csv_logging(logger=None):
    """Close CSV log file"""
    global log_file
    if log_file:
        log_file.close()
        log_file = None
        if logger:
            logger("CSV log closed.")


class JoystickControlGUI:
    """
    Simplified GUI for Direct Joystick Control
    
    This class provides a user interface for controlling the drone using
    keyboard WASD keys or on-screen buttons, with real-time feedback.
    Sensor Test is required before starting joystick control.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Dead Reckoning Joystick Control")
        self.root.geometry("1100x750")

        # Control variables
        self.joystick_thread = None
        self.joystick_active = False
        self.sensor_test_thread = None
        self.sensor_test_running = False
        self.joystick_keys = {"w": False, "a": False, "s": False, "d": False}
        self.key_pressed_flags = {"w": False, "a": False, "s": False, "d": False}

        self.create_ui()
        self.setup_plots()

        # Start animation
        self.anim = animation.FuncAnimation(
            self.fig, self.update_plots, interval=100, cache_frame_data=False
        )

        # Bind keyboard events
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
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
        height_frame.pack(fill=tk.X, pady=3)
        tk.Label(height_frame, text="Target Height (m):", width=16, anchor=tk.W).pack(side=tk.LEFT)
        self.height_entry_var = tk.StringVar(value=str(TARGET_HEIGHT))
        tk.Entry(height_frame, textvariable=self.height_entry_var, width=8).pack(side=tk.LEFT, padx=5)

        # TRIM VX
        trim_vx_frame = tk.Frame(params_frame)
        trim_vx_frame.pack(fill=tk.X, pady=3)
        tk.Label(trim_vx_frame, text="TRIM VX:", width=16, anchor=tk.W).pack(side=tk.LEFT)
        self.trim_vx_var = tk.StringVar(value=str(TRIM_VX))
        tk.Entry(trim_vx_frame, textvariable=self.trim_vx_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(trim_vx_frame, text="(fwd/back)", font=("Arial", 8), fg="gray").pack(side=tk.LEFT)

        # TRIM VY
        trim_vy_frame = tk.Frame(params_frame)
        trim_vy_frame.pack(fill=tk.X, pady=3)
        tk.Label(trim_vy_frame, text="TRIM VY:", width=16, anchor=tk.W).pack(side=tk.LEFT)
        self.trim_vy_var = tk.StringVar(value=str(TRIM_VY))
        tk.Entry(trim_vy_frame, textvariable=self.trim_vy_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(trim_vy_frame, text="(left/right)", font=("Arial", 8), fg="gray").pack(side=tk.LEFT)

        # Apply button
        tk.Button(
            params_frame, text="Apply Values", command=self.apply_values,
            bg="green", fg="white", font=("Arial", 10)
        ).pack(pady=5)

        # === Joystick Control Frame ===
        joystick_frame = tk.LabelFrame(left_frame, text="Joystick Control", padx=10, pady=10)
        joystick_frame.pack(fill=tk.X, pady=5)

        # Sensitivity control
        sensitivity_frame = tk.Frame(joystick_frame)
        sensitivity_frame.pack(fill=tk.X, pady=5)
        tk.Label(sensitivity_frame, text="Sensitivity:").pack(side=tk.LEFT)
        self.sensitivity_var = tk.StringVar(value=str(JOYSTICK_SENSITIVITY))
        tk.Entry(sensitivity_frame, textvariable=self.sensitivity_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(sensitivity_frame, text="(0.1-2.0)", font=("Arial", 8), fg="gray").pack(side=tk.LEFT)

        # Joystick button layout (3x3 grid)
        buttons_frame = tk.Frame(joystick_frame)
        buttons_frame.pack(pady=10)

        # Forward (W)
        self.btn_forward = tk.Button(
            buttons_frame, text="↑\nW", bg="green", fg="white",
            font=("Arial", 12), width=6, height=2
        )
        self.btn_forward.grid(row=0, column=1, padx=3, pady=3)
        self.btn_forward.bind("<ButtonPress>", lambda e: self.start_continuous_movement("w"))
        self.btn_forward.bind("<ButtonRelease>", lambda e: self.stop_continuous_movement("w"))

        # Left (A)
        self.btn_left = tk.Button(
            buttons_frame, text="←\nA", bg="green", fg="white",
            font=("Arial", 12), width=6, height=2
        )
        self.btn_left.grid(row=1, column=0, padx=3, pady=3)
        self.btn_left.bind("<ButtonPress>", lambda e: self.start_continuous_movement("a"))
        self.btn_left.bind("<ButtonRelease>", lambda e: self.stop_continuous_movement("a"))

        # Stop (center)
        self.btn_stop = tk.Button(
            buttons_frame, text="STOP", bg="red", fg="white",
            font=("Arial", 12, "bold"), width=6, height=2,
            command=self.stop_joystick_control
        )
        self.btn_stop.grid(row=1, column=1, padx=3, pady=3)

        # Right (D)
        self.btn_right = tk.Button(
            buttons_frame, text="→\nD", bg="green", fg="white",
            font=("Arial", 12), width=6, height=2
        )
        self.btn_right.grid(row=1, column=2, padx=3, pady=3)
        self.btn_right.bind("<ButtonPress>", lambda e: self.start_continuous_movement("d"))
        self.btn_right.bind("<ButtonRelease>", lambda e: self.stop_continuous_movement("d"))

        # Backward (S)
        self.btn_backward = tk.Button(
            buttons_frame, text="↓\nS", bg="green", fg="white",
            font=("Arial", 12), width=6, height=2
        )
        self.btn_backward.grid(row=2, column=1, padx=3, pady=3)
        self.btn_backward.bind("<ButtonPress>", lambda e: self.start_continuous_movement("s"))
        self.btn_backward.bind("<ButtonRelease>", lambda e: self.stop_continuous_movement("s"))

        # Start button
        self.start_button = tk.Button(
            joystick_frame, text="Start Joystick Control",
            command=self.start_joystick_control,
            bg="green", fg="white", font=("Arial", 11, "bold"), width=20
        )
        self.start_button.pack(pady=10)

        # Joystick status
        self.joystick_status_var = tk.StringVar(value="Joystick: INACTIVE")
        tk.Label(
            joystick_frame, textvariable=self.joystick_status_var,
            font=("Arial", 10, "bold"), fg="blue"
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
        self.phase_var = tk.StringVar(value="Phase: IDLE")

        tk.Label(values_frame, textvariable=self.height_var, font=("Arial", 10), fg="blue").pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.battery_var, font=("Arial", 10), fg="orange").pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.vx_var, font=("Arial", 10)).pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.vy_var, font=("Arial", 10)).pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.pos_x_var, font=("Arial", 10), fg="darkgreen").pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.pos_y_var, font=("Arial", 10), fg="darkgreen").pack(anchor=tk.W)
        tk.Label(values_frame, textvariable=self.phase_var, font=("Arial", 10, "bold"), fg="red").pack(anchor=tk.W)

        # === Output log ===
        output_frame = tk.LabelFrame(left_frame, text="Output Log", padx=5, pady=5)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.output_text = tk.Text(output_frame, height=8, width=40, font=("Consolas", 9))
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Right side - Plots
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_plots(self):
        """Setup matplotlib plots"""
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Velocities
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Position (2D)
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Height
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Empty or info

        # Velocities
        self.ax1.set_title("Velocities")
        self.ax1.set_ylabel("Velocity (m/s)")
        self.ax1.grid(True, alpha=0.3)
        (self.line_vx,) = self.ax1.plot([], [], "b-", label="VX")
        (self.line_vy,) = self.ax1.plot([], [], "r-", label="VY")
        self.ax1.legend()

        # 2D Position
        self.ax2.set_title("Position")
        self.ax2.set_xlabel("X (m)")
        self.ax2.set_ylabel("Y (m)")
        self.ax2.set_aspect("equal")
        self.ax2.grid(True, alpha=0.3)
        (self.line_pos,) = self.ax2.plot([], [], "purple", alpha=0.7, label="Trajectory")
        (self.current_pos,) = self.ax2.plot([], [], "ro", markersize=8, label="Current")
        self.ax2.plot(0, 0, "ko", markersize=10, markerfacecolor="yellow", label="Origin")
        self.ax2.legend()

        # Height
        self.ax3.set_title("Height")
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Height (m)")
        self.ax3.grid(True, alpha=0.3)
        (self.line_height,) = self.ax3.plot([], [], "orange", label="Height")
        self.target_height_line = self.ax3.axhline(y=TARGET_HEIGHT, color="red", linestyle="--", alpha=0.7, label="Target")
        self.ax3.legend()

        # Info panel
        self.ax4.set_title("Controls Info")
        self.ax4.axis('off')
        info_text = """
Keyboard Controls:
  W - Forward
  S - Backward
  A - Left
  D - Right
  Enter - Emergency Stop

Steps:
1. Run Sensor Test (ARM)
2. Adjust TRIM/Height
3. Start Joystick Control
4. Control with WASD
"""
        self.ax4.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center', family='monospace')

        self.fig.tight_layout()

    def update_plots(self, frame):
        """Update all plots with new data"""
        if not time_history:
            return []

        # Update value displays
        self.height_var.set(f"Height: {current_height:.3f}m")
        self.phase_var.set(f"Phase: {flight_phase}")
        if current_battery_voltage > 0:
            color = "green" if current_battery_voltage > 3.5 else ("orange" if current_battery_voltage > LOW_BATTERY_THRESHOLD else "red")
            status = "" if current_battery_voltage > 3.5 else (" (Warning)" if current_battery_voltage > LOW_BATTERY_THRESHOLD else " (LOW!)")
            self.battery_var.set(f"Battery: {current_battery_voltage:.2f}V{status}")
        self.vx_var.set(f"VX: {current_vx:.3f} m/s")
        self.vy_var.set(f"VY: {current_vy:.3f} m/s")
        self.pos_x_var.set(f"Pos X: {integrated_position_x:.3f}m")
        self.pos_y_var.set(f"Pos Y: {integrated_position_y:.3f}m")

        # Update plots
        try:
            self.line_vx.set_data(time_history, velocity_x_history_plot)
            self.line_vy.set_data(time_history, velocity_y_history_plot)

            if complete_trajectory_x and complete_trajectory_y:
                plot_x = [-x for x in complete_trajectory_x]
                self.line_pos.set_data(plot_x, complete_trajectory_y)
                self.current_pos.set_data([-integrated_position_x], [integrated_position_y])

            self.line_height.set_data(time_history, height_history)

            # Adjust axis limits
            if len(time_history) > 1:
                for ax in [self.ax1, self.ax3]:
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

                if height_history:
                    self.ax3.set_ylim(min(height_history) - 0.05, max(height_history) + 0.05)

        except Exception:
            pass

        return [self.line_vx, self.line_vy, self.line_pos, self.current_pos, self.line_height]

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
        global complete_trajectory_x, complete_trajectory_y, start_time

        time_history.clear()
        velocity_x_history_plot.clear()
        velocity_y_history_plot.clear()
        position_x_history.clear()
        position_y_history.clear()
        height_history.clear()
        complete_trajectory_x.clear()
        complete_trajectory_y.clear()
        start_time = None

    def apply_values(self):
        """Apply flight parameter values from UI"""
        global TARGET_HEIGHT, TRIM_VX, TRIM_VY
        try:
            TARGET_HEIGHT = float(self.height_entry_var.get())
            TRIM_VX = float(self.trim_vx_var.get())
            TRIM_VY = float(self.trim_vy_var.get())
            
            # Update height line in plot
            self.target_height_line.set_ydata([TARGET_HEIGHT, TARGET_HEIGHT])
            
            self.log_to_output(f"Applied: Height={TARGET_HEIGHT:.2f}m, TRIM_VX={TRIM_VX:.2f}, TRIM_VY={TRIM_VY:.2f}")
            self.status_var.set(f"Status: Values applied - Height={TARGET_HEIGHT:.2f}m")
        except ValueError as e:
            self.status_var.set(f"Status: Invalid value - {str(e)}")
            self.log_to_output(f"Error applying values: {str(e)}")

    def toggle_debug_mode(self):
        """Toggle debug mode"""
        global DEBUG_MODE
        DEBUG_MODE = self.debug_mode_var.get()
        mode_text = "ON (motors disabled)" if DEBUG_MODE else "OFF"
        self.log_to_output(f"Debug mode: {mode_text}")

    def start_continuous_movement(self, key):
        """Start continuous movement for GUI buttons"""
        if not self.joystick_active:
            return
        key = key.lower()
        if key in ["w", "a", "s", "d"]:
            self.joystick_keys[key] = True
            active_keys = [k.upper() for k, v in self.joystick_keys.items() if v]
            self.joystick_status_var.set(f"Joystick: ACTIVE ({','.join(active_keys)})")

    def stop_continuous_movement(self, key):
        """Stop continuous movement for GUI buttons"""
        if not self.joystick_active:
            return
        key = key.lower()
        if key in ["w", "a", "s", "d"]:
            self.joystick_keys[key] = False
            active_keys = [k.upper() for k, v in self.joystick_keys.items() if v]
            if active_keys:
                self.joystick_status_var.set(f"Joystick: ACTIVE ({','.join(active_keys)})")
            else:
                self.joystick_status_var.set("Joystick: ACTIVE")

    def _key_to_direction(self, key):
        """Convert key to direction name"""
        directions = {"w": "Forward", "a": "Left", "s": "Backward", "d": "Right"}
        return directions.get(key, key.upper())

    # ==================== SENSOR TEST ====================
    def start_sensor_test(self):
        """Start sensor test to ARM the drone"""
        if not self.sensor_test_running and not self.joystick_active:
            self.sensor_test_running = True
            self.sensor_test_button.config(text="Stop Sensor Test", command=self.stop_sensor_test, bg="red")
            self.status_var.set("Status: Sensor Test Running - ARMING...")
            self.sensor_test_thread = threading.Thread(target=self.sensor_test_controller_thread)
            self.sensor_test_thread.daemon = True
            self.sensor_test_thread.start()
        elif self.joystick_active:
            self.status_var.set("Status: Cannot run Sensor Test while Joystick is active")

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

    def sensor_test_controller_thread(self):
        """Sensor test thread - connects to drone and reads sensors"""
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

                # Setup logging
                log_motion, log_battery = setup_logging(cf, logger=self.log_to_output)
                if log_motion is not None:
                    time.sleep(1.0)

                # Initialize (but don't arm motors in sensor test)
                if not DEBUG_MODE:
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.1)
                    cf.param.set_value("commander.enHighLevel", "1")
                    time.sleep(0.5)

                # Enable position integration for display
                reset_position_tracking()

                # Run sensor test loop
                self.log_to_output("Sensor test running - reading sensors...")
                while sensor_test_active:
                    flight_phase = "SENSOR_TEST"
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

    # ==================== JOYSTICK CONTROL ====================
    def start_joystick_control(self):
        """Start joystick control"""
        if self.joystick_active or self.sensor_test_running:
            if self.sensor_test_running:
                self.status_var.set("Status: Stop Sensor Test first!")
            return

        try:
            sensitivity = float(self.sensitivity_var.get())
            if sensitivity < 0.1 or sensitivity > 2.0:
                raise ValueError("Sensitivity must be between 0.1 and 2.0")

            # Apply current values
            self.apply_values()

            # Battery safety check
            if current_battery_voltage > 0 and current_battery_voltage < LOW_BATTERY_THRESHOLD:
                self.status_var.set(f"Status: Battery too low ({current_battery_voltage:.2f}V)!")
                return
            elif current_battery_voltage == 0.0:
                self.log_to_output("WARNING: Battery voltage unknown - run Sensor Test first!")
                self.status_var.set("Status: Run Sensor Test first to check battery!")
                return

            # Sensor safety check
            if not sensor_data_ready:
                self.status_var.set("Status: Sensor data not ready - Run Sensor Test first!")
                return

            if current_height <= 0.0:
                self.status_var.set("Status: Invalid height reading!")
                return

            # Start joystick control
            self.joystick_active = True
            self.start_button.config(state=tk.DISABLED)
            self.joystick_status_var.set("Joystick: ACTIVE")
            self.status_var.set("Status: Joystick Control Starting...")
            self.log_to_output("Joystick control started")

            self.root.focus_force()

            # Start control thread
            self.joystick_thread = threading.Thread(target=self.joystick_control_thread)
            self.joystick_thread.daemon = True
            self.joystick_thread.start()

        except ValueError as e:
            self.status_var.set(f"Status: {str(e)}")

    def stop_joystick_control(self):
        """Stop joystick control"""
        if self.joystick_active:
            self.joystick_active = False
            global flight_active
            flight_active = False

            if self.joystick_thread and self.joystick_thread.is_alive():
                self.joystick_thread.join(timeout=1.0)

            self.start_button.config(state=tk.NORMAL)
            self.joystick_status_var.set("Joystick: INACTIVE")
            self.status_var.set("Status: Joystick Control Stopped")
            self.log_to_output("Joystick control stopped")

    def emergency_stop(self):
        """Emergency stop"""
        global flight_active, sensor_test_active
        flight_active = False
        sensor_test_active = False
        self.joystick_active = False
        self.sensor_test_running = False

        self.joystick_keys = {"w": False, "a": False, "s": False, "d": False}
        self.key_pressed_flags = {"w": False, "a": False, "s": False, "d": False}

        self.start_button.config(state=tk.NORMAL)
        self.sensor_test_button.config(text="Sensor Test (ARM)", command=self.start_sensor_test, bg="lightblue")
        self.joystick_status_var.set("Joystick: EMERGENCY STOP")
        self.status_var.set("Status: EMERGENCY STOPPED")
        self.log_to_output("EMERGENCY STOP!")

    def joystick_control_thread(self):
        """Joystick control thread - direct control without position hold"""
        global flight_active, flight_phase
        global current_battery_voltage, battery_data_ready
        global integrated_position_x, integrated_position_y
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

                # Setup logging
                log_motion, log_battery = setup_logging(cf, logger=self.log_to_output)
                if log_motion is not None:
                    time.sleep(1.0)

                # Reset position tracking
                reset_position_tracking()

                # Initialize flight
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

                while time.time() - start_time_local < TAKEOFF_TIME and self.joystick_active:
                    if not DEBUG_MODE:
                        cf.commander.send_hover_setpoint(TRIM_VX, TRIM_VY, 0, TARGET_HEIGHT)
                    log_to_csv()
                    time.sleep(0.01)

                # Stabilization
                flight_phase = "STABILIZING"
                stabilization_start = time.time()
                while time.time() - stabilization_start < 2.0 and self.joystick_active:
                    if not DEBUG_MODE:
                        cf.commander.send_hover_setpoint(TRIM_VX, TRIM_VY, 0, TARGET_HEIGHT)
                    log_to_csv()
                    time.sleep(CONTROL_UPDATE_RATE)

                # Main control loop - DIRECT joystick control (no position hold)
                flight_phase = "JOYSTICK_CONTROL"
                self.log_to_output("Joystick control active - use WASD to move")

                while self.joystick_active:
                    try:
                        sensitivity = float(self.sensitivity_var.get())
                    except:
                        sensitivity = JOYSTICK_SENSITIVITY

                    # Calculate direct velocity commands
                    joystick_vx = 0.0
                    joystick_vy = 0.0

                    if self.joystick_keys["w"]:  # Forward
                        joystick_vy += sensitivity
                    if self.joystick_keys["s"]:  # Backward
                        joystick_vy -= sensitivity
                    if self.joystick_keys["a"]:  # Left
                        joystick_vx += sensitivity
                    if self.joystick_keys["d"]:  # Right
                        joystick_vx -= sensitivity

                    # Apply controls (with axis swap for Crazyflie coordinate system)
                    total_vx = TRIM_VX + joystick_vy
                    total_vy = TRIM_VY + joystick_vx

                    if not DEBUG_MODE:
                        cf.commander.send_hover_setpoint(total_vx, total_vy, 0, TARGET_HEIGHT)

                    log_to_csv()
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
            self.joystick_active = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.joystick_status_var.set("Joystick: INACTIVE"))
            self.root.after(0, lambda: self.status_var.set("Status: Flight Complete"))

    def on_key_press(self, event):
        """Handle key press events"""
        if event.keysym in ("Return", "KP_Enter"):
            self.emergency_stop()
            self.log_to_output("EMERGENCY STOP: Enter key pressed")
            return

        if not self.joystick_active:
            return

        key = event.char.lower()
        if key in ["w", "a", "s", "d"]:
            if not self.key_pressed_flags[key]:
                self.key_pressed_flags[key] = True
                self.joystick_keys[key] = True
                active_keys = [k.upper() for k, v in self.joystick_keys.items() if v]
                self.joystick_status_var.set(f"Joystick: ACTIVE ({','.join(active_keys)})")
                self.log_to_output(f"Moving: {self._key_to_direction(key)}")
            else:
                self.joystick_keys[key] = True

    def on_key_release(self, event):
        """Handle key release events"""
        if not self.joystick_active:
            return

        key = event.char.lower()
        if key in ["w", "a", "s", "d"]:
            if self.key_pressed_flags[key]:
                self.key_pressed_flags[key] = False
                self.joystick_keys[key] = False
                active_keys = [k.upper() for k, v in self.joystick_keys.items() if v]
                if active_keys:
                    self.joystick_status_var.set(f"Joystick: ACTIVE ({','.join(active_keys)})")
                else:
                    self.joystick_status_var.set("Joystick: ACTIVE")
            else:
                self.joystick_keys[key] = False


def main():
    """Main entry point"""
    try:
        cflib.crtp.init_drivers()
        print("Crazyflie CRTP drivers initialized")
    except Exception as e:
        print(f"Warning: cflib.crtp.init_drivers() failed: {e}")

    root = tk.Tk()
    app = JoystickControlGUI(root)

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
