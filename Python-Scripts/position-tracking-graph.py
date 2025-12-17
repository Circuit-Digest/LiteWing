import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# URI for your LiteWing drone
DRONE_URI = "udp://192.168.43.42"

# Initialize CRTP drivers
cflib.crtp.init_drivers()

# Shared data dictionary and storage for plotting
log_data = {}
data_history = {
    'velocity.x': [],
    'velocity.y': [],
    'position.x': [],
    'position.y': [],
    'stateEstimate.z': []
}
timestamps = []
start_time = None
max_points = 500

# Velocity calculation parameters
DEG_TO_RAD = 3.14159 / 180.0
SENSOR_PERIOD_MS = 20
DT = SENSOR_PERIOD_MS / 1000.0

# Simple threshold to ignore small spikes
VELOCITY_THRESHOLD = 0.008  # Ignore velocities smaller than this (m/s)

# Position tracking variables
current_position_x = 0.0
current_position_y = 0.0
previous_timestamp = None
last_integration_time = None

# Drift compensation parameters
DRIFT_COMPENSATION_RATE = 0.001  # How aggressively to correct drift (0.0 = no correction, 0.01 = aggressive)
position_drift_x = 0.0
position_drift_y = 0.0

# Simple smoothing variables (3-point average)
previous_velocity_x = 0.0
previous_velocity_y = 0.0
prev2_velocity_x = 0.0  # Second previous value
prev2_velocity_y = 0.0  # Second previous value
current_height = 0.0

# Store previous output values for plotting
previous_output = {
    'velocity.x': 0.0,
    'velocity.y': 0.0
}


def calculate_velocity(delta_value, altitude):
    """Convert delta value to linear velocity"""
    if altitude <= 0:
        return 0.0

    velocity_constant = (4.2 * DEG_TO_RAD) / (30.0 * DT)
    velocity = delta_value * altitude * velocity_constant
    return velocity


def integrate_position(vx, vy, dt):
    """Integrate velocity to position using trapezoidal rule with drift compensation"""
    global current_position_x, current_position_y, position_drift_x, position_drift_y

    if dt <= 0 or dt > 0.1:  # Sanity check for integration time
        return current_position_x, current_position_y

    # Trapezoidal integration: position += (velocity * dt)
    # For first-order integration, this is simply: dx = vx * dt
    delta_x = vx * dt
    delta_y = vy * dt

    # Apply integration
    current_position_x += delta_x
    current_position_y += delta_y

    # Simple drift compensation - gradually pull position towards zero when velocity is low
    velocity_magnitude = np.sqrt(vx * vx + vy * vy)

    if velocity_magnitude < VELOCITY_THRESHOLD * 2:  # Only apply when moving slowly
        # Gentle drift correction
        position_drift_x += current_position_x * DRIFT_COMPENSATION_RATE * dt
        position_drift_y += current_position_y * DRIFT_COMPENSATION_RATE * dt

        current_position_x -= current_position_x * DRIFT_COMPENSATION_RATE * dt
        current_position_y -= current_position_y * DRIFT_COMPENSATION_RATE * dt

    return current_position_x, current_position_y


class MotionLoggerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Data Logger - Position Tracking with Velocity Integration")
        self.root.geometry("1400x1000")
        self.cf = Crazyflie(rw_cache='./cache')
        self.scf = None
        self.logging_active = False
        self.paused = False
        self.anim = None
        self.data_received_count = 0
        self.connected = False
        self.log_motion = None

        # Connect to drone first
        try:
            print("Connecting to drone...")
            self.scf = SyncCrazyflie(DRONE_URI, cf=self.cf)
            self.scf.__enter__()
            print("✓ Connected successfully!")
            self.connected = True
            time.sleep(1.0)

            # Print available log variables for debugging
            self.print_available_variables()

            # Create UI elements
            self.create_ui()

            # Setup logging configuration
            self.setup_logging()

        except Exception as e:
            print(f"✗ Error connecting to drone: {e}")
            self.connected = False
            self.create_ui()
            return

    def print_available_variables(self):
        """Print available log variables for debugging"""
        print("\n=== Available Log Variables ===")
        try:
            toc = self.cf.log.toc.toc
            target_vars = ['motion.deltaX', 'motion.deltaY', 'stateEstimate.z']
            print(f"Checking target variables:")
            for var in target_vars:
                group, name = var.split('.')
                if group in toc and name in toc[group]:
                    print(f"  ✓ {var} - FOUND")
                else:
                    print(f"  ✗ {var} - NOT FOUND")
        except Exception as e:
            print(f"Error reading TOC: {e}")
        print("=== End Variable List ===\n")

    def create_ui(self):
        """Create the user interface"""
        # Control panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Connection status
        status_text = "Status: Connected" if self.connected else "Status: Disconnected"
        status_color = "green" if self.connected else "red"
        self.status_var = tk.StringVar(value=status_text)
        self.status_label = tk.Label(control_frame, textvariable=self.status_var,
                                     fg=status_color, font=("Arial", 12, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Control buttons
        self.start_button = tk.Button(control_frame, text="Start Logging",
                                      command=self.start_logging, bg="green", fg="white")
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = tk.Button(control_frame, text="Pause",
                                      command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(control_frame, text="Clear Data",
                                      command=self.clear_data)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # NEW: Reset position button
        self.reset_position_button = tk.Button(control_frame, text="Reset Position (0,0)",
                                               command=self.reset_position, bg="orange", fg="white")
        self.reset_position_button.pack(side=tk.LEFT, padx=5)

        # Data info
        self.data_info_var = tk.StringVar(value="Data Points: 0 | Callbacks: 0")
        self.data_info_label = tk.Label(control_frame, textvariable=self.data_info_var)
        self.data_info_label.pack(side=tk.RIGHT, padx=10)

        # Current values frame
        values_frame = tk.LabelFrame(self.root, text="Current Values", padx=10, pady=5)
        values_frame.pack(fill=tk.X, padx=10, pady=5)

        # Current value labels
        self.velocity_x_var = tk.StringVar(value="Velocity X: N/A")
        self.velocity_y_var = tk.StringVar(value="Velocity Y: N/A")
        self.height_var = tk.StringVar(value="Height: N/A")

        # NEW: Position display
        self.position_x_var = tk.StringVar(value="Position X: 0.000m")
        self.position_y_var = tk.StringVar(value="Position Y: 0.000m")

        tk.Label(values_frame, textvariable=self.velocity_x_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=15)
        tk.Label(values_frame, textvariable=self.velocity_y_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=15)
        tk.Label(values_frame, textvariable=self.height_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=15)

        # Position displays with different color
        tk.Label(values_frame, textvariable=self.position_x_var, font=("Arial", 10, "bold"), fg="blue").pack(
            side=tk.LEFT, padx=15)
        tk.Label(values_frame, textvariable=self.position_y_var, font=("Arial", 10, "bold"), fg="blue").pack(
            side=tk.LEFT, padx=15)

        # Filter info
        filter_info = f"Integration: Trapezoidal | Drift Comp: {DRIFT_COMPENSATION_RATE}"
        tk.Label(values_frame, text=filter_info, font=("Arial", 8), fg="green").pack(side=tk.RIGHT, padx=10)

        # ToF Height Display Window
        tof_frame = tk.LabelFrame(self.root, text="ToF Height Monitor (stateEstimate.z)", padx=10, pady=5)
        tof_frame.pack(fill=tk.X, padx=10, pady=5)

        # Large ToF display
        self.tof_display_var = tk.StringVar(value="ToF Height: N/A")
        self.tof_display_label = tk.Label(tof_frame, textvariable=self.tof_display_var,
                                          font=("Arial", 16, "bold"), fg="blue")
        self.tof_display_label.pack(side=tk.LEFT, padx=20)

        # ToF status indicator
        self.tof_status_var = tk.StringVar(value="Status: Disconnected")
        self.tof_status_label = tk.Label(tof_frame, textvariable=self.tof_status_var,
                                         font=("Arial", 12), fg="red")
        self.tof_status_label.pack(side=tk.LEFT, padx=20)

        # ToF reading count
        self.tof_count_var = tk.StringVar(value="Readings: 0")
        tk.Label(tof_frame, textvariable=self.tof_count_var, font=("Arial", 10)).pack(side=tk.RIGHT, padx=20)

        # Matplotlib figure with 4 subplots (2x2 layout)
        self.fig = Figure(figsize=(14, 10))

        # Create subplots in 2x2 layout
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Top left: Velocity X
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Top right: Velocity Y
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Bottom left: Height
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Bottom right: Position Plot (2D)

        # Setup plots
        self.setup_plots()

        # Embed plot in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_plots(self):
        """Setup the matplotlib plots"""
        # Velocity X plot
        self.ax1.set_title("Velocity X - Filtered (m/s)", fontsize=11)
        self.ax1.set_ylabel("Velocity X (m/s)")
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Velocity X')
        self.ax1.legend()

        # Velocity Y plot
        self.ax2.set_title("Velocity Y - Filtered (m/s)", fontsize=11)
        self.ax2.set_ylabel("Velocity Y (m/s)")
        self.ax2.grid(True, alpha=0.3)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2, label='Velocity Y')
        self.ax2.legend()

        # Height plot
        self.ax3.set_title("Height (stateEstimate.z)", fontsize=11)
        self.ax3.set_xlabel("Time (seconds)")
        self.ax3.set_ylabel("Height (m)")
        self.ax3.grid(True, alpha=0.3)
        self.line3, = self.ax3.plot([], [], 'g-', linewidth=2, label='Height')
        self.ax3.legend()

        # NEW: 2D Position plot
        self.ax4.set_title("2D Position Tracking (Integrated from Velocity)", fontsize=11)
        self.ax4.set_xlabel("X Position (m)")
        self.ax4.set_ylabel("Y Position (m)")
        self.ax4.grid(True, alpha=0.3)
        self.ax4.set_aspect('equal')  # Equal aspect ratio for realistic movement

        # Position trajectory line
        self.line4, = self.ax4.plot([], [], 'purple', linewidth=2, alpha=0.7, label='Trajectory')
        # Current position marker
        self.current_pos_marker, = self.ax4.plot([], [], 'ro', markersize=8, label='Current Position')
        # Origin marker
        self.ax4.plot(0, 0, 'ko', markersize=10, label='Origin (0,0)', markerfacecolor='yellow',
                      markeredgecolor='black')
        self.ax4.legend()

        # Adjust layout
        self.fig.tight_layout(pad=2.0)

    def setup_logging(self):
        """Setup logging configuration"""
        if not self.connected:
            print("Cannot setup logging - not connected to drone")
            return False

        print("Setting up logging configuration...")
        self.log_motion = LogConfig(name="Motion", period_in_ms=20)

        try:
            toc = self.cf.log.toc.toc
            variables_to_add = [
                ('motion.deltaX', 'int16_t'),
                ('motion.deltaY', 'int16_t'),
                ('stateEstimate.z', 'float')
            ]

            added_vars = []
            for var_name, var_type in variables_to_add:
                group, name = var_name.split('.')
                if group in toc and name in toc[group]:
                    try:
                        self.log_motion.add_variable(var_name, var_type)
                        added_vars.append(var_name)
                        print(f"✓ Added variable: {var_name}")
                    except Exception as e:
                        print(f"✗ Failed to add {var_name}: {e}")
                else:
                    print(f"✗ Variable not found: {var_name}")

            if len(added_vars) < 2:
                print("ERROR: Not enough required variables found!")
                self.status_var.set("Status: Missing motion variables")
                return False

            print(f"Successfully added {len(added_vars)} variables")

        except Exception as e:
            print(f"Error setting up variables: {e}")
            self.status_var.set(f"Status: Setup Error - {e}")
            return False

        # Set up callbacks
        self.log_motion.data_received_cb.add_callback(self.motion_callback)
        self.log_motion.error_cb.add_callback(self.logging_error_callback)
        self.log_motion.started_cb.add_callback(self.logging_started_callback)
        self.log_motion.added_cb.add_callback(self.logging_added_callback)

        print("Logging configuration setup complete!")
        return True

    def logging_error_callback(self, logconf, msg):
        """Callback when logging encounters an error"""
        print(f"Logging error: {msg}")
        self.status_var.set(f"Status: Logging Error - {msg}")

    def logging_started_callback(self, logconf, started):
        """Callback when logging starts/stops"""
        if started:
            print("✓ Logging started successfully!")
            self.status_var.set("Status: Logging Active")
        else:
            print("✗ Logging stopped")
            self.status_var.set("Status: Logging Stopped")

    def logging_added_callback(self, logconf, added):
        """Callback when log configuration is added"""
        if added:
            print("✓ Log configuration added successfully")
        else:
            print("✗ Failed to add log configuration")

    def motion_callback(self, timestamp, data, logconf):
        """Callback function - fast with position integration"""
        try:
            global current_height, previous_velocity_x, previous_velocity_y, prev2_velocity_x, prev2_velocity_y
            global previous_timestamp, last_integration_time, current_position_x, current_position_y

            self.data_received_count += 1
            current_time = time.time()

            # Extract raw data
            raw_delta_x = data.get('motion.deltaX', 0)
            raw_delta_y = data.get('motion.deltaY', 0)
            current_height = data.get('stateEstimate.z', 0)

            # Fast velocity calculation
            raw_velocity_x = calculate_velocity(raw_delta_x, current_height)
            raw_velocity_y = calculate_velocity(raw_delta_y, current_height)

            # Super fast 3-point average: (current + prev + prev2) / 3
            smoothed_velocity_x = (raw_velocity_x + previous_velocity_x + prev2_velocity_x) * 0.333
            smoothed_velocity_y = (raw_velocity_y + previous_velocity_y + prev2_velocity_y) * 0.333

            # Apply simple threshold to ignore small spikes
            if abs(smoothed_velocity_x) < VELOCITY_THRESHOLD:
                smoothed_velocity_x = 0.0
            if abs(smoothed_velocity_y) < VELOCITY_THRESHOLD:
                smoothed_velocity_y = 0.0

            # NEW: Position integration
            if last_integration_time is not None:
                dt = current_time - last_integration_time
                if 0.001 <= dt <= 0.1:  # Reasonable time delta
                    current_position_x, current_position_y = integrate_position(
                        smoothed_velocity_x, smoothed_velocity_y, dt)

            last_integration_time = current_time

            # Shift values for next iteration (very fast)
            prev2_velocity_x = previous_velocity_x
            prev2_velocity_y = previous_velocity_y
            previous_velocity_x = raw_velocity_x
            previous_velocity_y = raw_velocity_y

            # Store for plotting
            previous_output['velocity.x'] = smoothed_velocity_x
            previous_output['velocity.y'] = smoothed_velocity_y

            # Update UI every 10 callbacks
            if self.data_received_count % 10 == 0:
                self.velocity_x_var.set(f"VX: {smoothed_velocity_x:.3f} m/s")
                self.velocity_y_var.set(f"VY: {smoothed_velocity_y:.3f} m/s")
                self.height_var.set(f"Height: {current_height:.3f}m")
                self.position_x_var.set(f"Pos X: {current_position_x:.3f}m")
                self.position_y_var.set(f"Pos Y: {current_position_y:.3f}m")
                self.data_info_var.set(f"Callbacks: {self.data_received_count}")

            # Minimal console output
            if self.data_received_count % 200 == 0:
                print(
                    f"#{self.data_received_count} - VX: {smoothed_velocity_x:.3f}, VY: {smoothed_velocity_y:.3f}, PosX: {current_position_x:.3f}, PosY: {current_position_y:.3f}")

        except Exception as e:
            print(f"Error in motion callback: {e}")

    def reset_position(self):
        """Reset position to origin (0,0)"""
        global current_position_x, current_position_y, position_drift_x, position_drift_y, last_integration_time

        current_position_x = 0.0
        current_position_y = 0.0
        position_drift_x = 0.0
        position_drift_y = 0.0
        last_integration_time = None

        # Clear position history but keep velocity history
        data_history['position.x'].clear()
        data_history['position.y'].clear()

        # Update UI
        self.position_x_var.set("Pos X: 0.000m")
        self.position_y_var.set("Pos Y: 0.000m")

        print("Position reset to origin (0,0)")

    def update_plot(self, frame):
        """Update the plots with new data"""
        if not self.logging_active or self.paused:
            return [self.line1, self.line2, self.line3, self.line4, self.current_pos_marker]

        current_time = time.time() - start_time
        timestamps.append(current_time)

        # Always add data
        data_history['velocity.x'].append(previous_output['velocity.x'])
        data_history['velocity.y'].append(previous_output['velocity.y'])
        data_history['stateEstimate.z'].append(current_height)
        data_history['position.x'].append(current_position_x)
        data_history['position.y'].append(current_position_y)

        # Trim data to max_points
        if len(timestamps) > max_points:
            timestamps.pop(0)
            for param in data_history:
                data_history[param].pop(0)

        # Update plot lines
        if timestamps:
            self.line1.set_data(timestamps, data_history['velocity.x'])
            self.line2.set_data(timestamps, data_history['velocity.y'])
            self.line3.set_data(timestamps, data_history['stateEstimate.z'])

            # Update 2D position plot
            if data_history['position.x'] and data_history['position.y']:
                self.line4.set_data(data_history['position.x'], data_history['position.y'])
                # Current position marker
                self.current_pos_marker.set_data([current_position_x], [current_position_y])

            # Adjust time-based plot limits
            time_range = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1
            time_margin = time_range * 0.05

            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_xlim(min(timestamps) - time_margin, max(timestamps) + time_margin)

                # Set Y limits for velocity and height plots
                if ax == self.ax1 and data_history['velocity.x']:
                    x_data = data_history['velocity.x']
                    if x_data and any(x != 0 for x in x_data):
                        x_range = max(x_data) - min(x_data) if len(set(x_data)) > 1 else 0.1
                        x_margin = max(x_range * 0.1, 0.01)
                        self.ax1.set_ylim(min(x_data) - x_margin, max(x_data) + x_margin)

                elif ax == self.ax2 and data_history['velocity.y']:
                    y_data = data_history['velocity.y']
                    if y_data and any(y != 0 for y in y_data):
                        y_range = max(y_data) - min(y_data) if len(set(y_data)) > 1 else 0.1
                        y_margin = max(y_range * 0.1, 0.01)
                        self.ax2.set_ylim(min(y_data) - y_margin, max(y_data) + y_margin)

                elif ax == self.ax3 and data_history['stateEstimate.z']:
                    z_data = data_history['stateEstimate.z']
                    if z_data and any(z != 0 for z in z_data):
                        z_range = max(z_data) - min(z_data) if len(set(z_data)) > 1 else 0.1
                        z_margin = max(z_range * 0.1, 0.05)
                        self.ax3.set_ylim(min(z_data) - z_margin, max(z_data) + z_margin)
                    else:
                        self.ax3.set_ylim(0, 0.5)

            # Adjust 2D position plot limits
            if data_history['position.x'] and data_history['position.y']:
                pos_x_data = data_history['position.x']
                pos_y_data = data_history['position.y']

                if pos_x_data and pos_y_data:
                    x_range = max(pos_x_data) - min(pos_x_data)
                    y_range = max(pos_y_data) - min(pos_y_data)
                    max_range = max(x_range, y_range, 0.5)  # Minimum range of 0.5m

                    center_x = (max(pos_x_data) + min(pos_x_data)) / 2
                    center_y = (max(pos_y_data) + min(pos_y_data)) / 2

                    margin_range = max_range * 0.6
                    self.ax4.set_xlim(center_x - margin_range, center_x + margin_range)
                    self.ax4.set_ylim(center_y - margin_range, center_y + margin_range)

        return [self.line1, self.line2, self.line3, self.line4, self.current_pos_marker]

    def start_logging(self):
        """Start motion logging"""
        if not self.connected:
            print("✗ Cannot start logging - not connected to drone")
            self.status_var.set("Status: Not connected")
            return False

        if self.log_motion is None:
            print("✗ Cannot start logging - logging configuration not set up")
            self.status_var.set("Status: Logging not configured")
            return False

        global start_time

        print("Starting motion logging...")

        try:
            # Add log configuration
            print("Adding log configuration to Crazyflie...")
            self.cf.log.add_config(self.log_motion)
            time.sleep(0.5)

            # Check if the configuration is valid
            if not self.log_motion.valid:
                print("ERROR: Log configuration is not valid!")
                self.status_var.set("Status: Invalid log configuration")
                return False

            print("✓ Log configuration is valid")

            # Start logging
            print("Starting log configuration...")
            self.log_motion.start()
            time.sleep(0.5)

            self.logging_active = True
            start_time = time.time()

            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.status_var.set("Status: Starting...")

            # Start plot animation
            print("Starting plot animation...")
            self.anim = animation.FuncAnimation(
                self.fig, self.update_plot, interval=50, blit=True, cache_frame_data=False
            )
            self.canvas.draw()

            print("✓ Motion logging started successfully!")
            return True

        except Exception as e:
            print(f"ERROR starting logging: {e}")
            self.stop_logging()
            self.status_var.set(f"Status: Error - {e}")
            return False

    def toggle_pause(self):
        """Toggle pause/resume logging"""
        if not self.log_motion:
            return

        if self.paused:
            self.log_motion.start()
            self.paused = False
            self.pause_button.config(text="Pause")
            self.status_var.set("Status: Logging Active")
            print("Logging resumed")
        else:
            self.log_motion.stop()
            self.paused = True
            self.pause_button.config(text="Resume")
            self.status_var.set("Status: Paused")
            print("Logging paused")

    def clear_data(self):
        """Clear all collected data"""
        global start_time, previous_velocity_x, previous_velocity_y, prev2_velocity_x, prev2_velocity_y, current_height
        global current_position_x, current_position_y, position_drift_x, position_drift_y, last_integration_time

        # Clear data history
        for param in data_history:
            data_history[param].clear()
        timestamps.clear()

        # Reset simple variables
        previous_velocity_x = 0.0
        previous_velocity_y = 0.0
        prev2_velocity_x = 0.0
        prev2_velocity_y = 0.0
        current_height = 0.0

        # Reset position variables
        current_position_x = 0.0
        current_position_y = 0.0
        position_drift_x = 0.0
        position_drift_y = 0.0
        last_integration_time = None

        for param in previous_output:
            previous_output[param] = 0.0

        # Reset counters
        self.data_received_count = 0

        # Reset start time
        if self.logging_active:
            start_time = time.time()

        # Update UI
        self.data_info_var.set("Data Points: 0 | Callbacks: 0")
        self.position_x_var.set("Pos X: 0.000m")
        self.position_y_var.set("Pos Y: 0.000m")

        # Clear plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        self.setup_plots()
        self.canvas.draw()

        print("Data cleared and position reset")

    def stop_logging(self):
        """Stop motion logging"""
        if self.logging_active and self.log_motion is not None:
            self.log_motion.stop()
            self.logging_active = False

        # Update UI
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.status_var.set("Status: Stopped")

        print("Motion logging stopped")

    def on_closing(self):
        """Handle window closing"""
        print("Closing application...")
        self.stop_logging()

        if self.scf:
            try:
                self.scf.__exit__(None, None, None)
                print("Disconnected from drone")
            except:
                pass

        self.root.destroy()


if __name__ == "__main__":
    print("Starting Enhanced Motion Data Logger with Position Tracking...")
    print("Features: Velocity Integration → Position Tracking with Drift Compensation")
    print("Connecting to drone first...")

    root = tk.Tk()
    app = MotionLoggerUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("Application closed")