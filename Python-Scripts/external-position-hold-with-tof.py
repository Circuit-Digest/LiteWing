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
    'motion.deltaX': [],
    'motion.deltaY': [],
    'stateEstimate.z': []
}
timestamps = []
start_time = None
max_points = 500  # Keep more points for motion data

# Parameter labels and colors
param_labels = {
    'motion.deltaX': 'Delta X (Flow X)',
    'motion.deltaY': 'Delta Y (Flow Y)',
    'stateEstimate.z': 'Height (Z)'
}

param_colors = {
    'motion.deltaX': 'blue',
    'motion.deltaY': 'red',
    'stateEstimate.z': 'green'
}

# === Enhanced Filter Parameters ===
# Moving Average Filter
MOVING_AVG_WINDOW = 10  # Number of samples for moving average

# Threshold Filter - only update if change is significant
THRESHOLD_DELTA_X = 0.01  # Minimum change to register for Delta X
THRESHOLD_DELTA_Y = 0.01  # Minimum change to register for Delta Y

# EMA Filter Parameters
EMA_ALPHA = 0.1  # Reduced alpha for more smoothing (was 0.2)

# Height-based filtering parameters
TARGET_HEIGHT = 0.2  # Target height in meters
HEIGHT_TOLERANCE = 0.05  # Tolerance in meters (±0.05m around 0.2m)
MIN_HEIGHT = TARGET_HEIGHT - HEIGHT_TOLERANCE  # 0.15m
MAX_HEIGHT = TARGET_HEIGHT + HEIGHT_TOLERANCE  # 0.25m

# Store filter data
moving_avg_buffer = {
    'motion.deltaX': [],
    'motion.deltaY': []
}

# Store previous filtered values
ema_filtered = {
    'motion.deltaX': 0.0,
    'motion.deltaY': 0.0
}

# Store previous output values for threshold filtering
previous_output = {
    'motion.deltaX': 0.0,
    'motion.deltaY': 0.0
}

# Store current height for filtering logic
current_height = 0.0


def apply_moving_average(param, new_value):
    """Apply moving average filter"""
    if param not in moving_avg_buffer:
        moving_avg_buffer[param] = []

    moving_avg_buffer[param].append(new_value)

    # Keep only the last WINDOW samples
    if len(moving_avg_buffer[param]) > MOVING_AVG_WINDOW:
        moving_avg_buffer[param].pop(0)

    # Return average
    return sum(moving_avg_buffer[param]) / len(moving_avg_buffer[param])


def apply_threshold_filter(param, new_value, threshold):
    """Apply threshold filter - only update if change is significant"""
    change = abs(new_value - previous_output[param])

    if change >= threshold:
        previous_output[param] = new_value
        return new_value
    else:
        return previous_output[param]


def is_height_valid():
    """Check if current height is within the valid range for motion detection"""
    return MIN_HEIGHT <= current_height <= MAX_HEIGHT


def apply_combined_filter(param, raw_value):
    """Apply combined filtering: Moving Average + EMA + Threshold"""
    # Step 1: Moving Average (reduces high-frequency noise)
    ma_filtered = apply_moving_average(param, raw_value)

    # Step 2: EMA filter (further smoothing)
    ema_filtered[param] = EMA_ALPHA * ma_filtered + (1 - EMA_ALPHA) * ema_filtered[param]

    # Step 3: Threshold filter (prevents small meaningless changes)
    threshold_val = THRESHOLD_DELTA_X if 'deltaX' in param else THRESHOLD_DELTA_Y
    final_filtered = apply_threshold_filter(param, ema_filtered[param], threshold_val)

    return final_filtered


class MotionLoggerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Data Logger - Delta X, Y & Height (Filtered)")
        self.root.geometry("1200x800")  # Increased height for 3 plots
        self.cf = Crazyflie(rw_cache='./cache')
        self.scf = None
        self.logging_active = False
        self.paused = False
        self.anim = None
        self.data_received_count = 0

        # Connect to drone first
        try:
            print("Connecting to drone...")
            self.scf = SyncCrazyflie(DRONE_URI, cf=self.cf)
            self.scf.__enter__()
            print("Connected successfully!")
            time.sleep(1.0)  # Give connection time to stabilize

            # Print available log variables for debugging
            self.print_available_variables()

        except Exception as e:
            print(f"Error connecting to drone: {e}")
            self.show_error_and_exit(f"Connection failed: {e}")
            return

        # Create UI elements
        self.create_ui()

        # Setup logging configuration
        self.setup_logging()

    def print_available_variables(self):
        """Print available log variables for debugging"""
        print("\n=== Available Log Variables ===")
        try:
            toc = self.cf.log.toc.toc
            motion_vars = []
            all_vars = []

            for group_name in sorted(toc.keys()):
                group = toc[group_name]
                for var_name in sorted(group.keys()):
                    full_name = f"{group_name}.{var_name}"
                    all_vars.append(full_name)
                    if 'motion' in group_name.lower() or 'flow' in group_name.lower():
                        motion_vars.append(full_name)

            print("Motion-related variables:")
            if motion_vars:
                for var in motion_vars:
                    print(f"  - {var}")
            else:
                print("  - No motion variables found")

            print(f"\nTotal variables available: {len(all_vars)}")
            print("First 20 variables:")
            for var in all_vars[:20]:
                print(f"  - {var}")

            # Check specifically for our target variables
            target_vars = ['motion.deltaX', 'motion.deltaY', 'stateEstimate.z']
            print(f"\nChecking target variables:")
            for var in target_vars:
                group, name = var.split('.')
                if group in toc and name in toc[group]:
                    print(f"  ✓ {var} - FOUND")
                else:
                    print(f"  ✗ {var} - NOT FOUND")

        except Exception as e:
            print(f"Error reading TOC: {e}")
        print("=== End Variable List ===\n")

    def show_error_and_exit(self, error_msg):
        """Show error message and exit"""
        error_window = tk.Toplevel(self.root)
        error_window.title("Connection Error")
        error_window.geometry("400x200")

        tk.Label(error_window, text="Failed to connect to drone!",
                 font=("Arial", 14), fg="red").pack(pady=20)
        tk.Label(error_window, text=error_msg,
                 font=("Arial", 10)).pack(pady=10)
        tk.Label(error_window, text="Please check your drone connection and try again.",
                 font=("Arial", 10)).pack(pady=10)

        tk.Button(error_window, text="Exit",
                  command=self.root.destroy).pack(pady=20)

    def create_ui(self):
        """Create the user interface"""
        # Control panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Connection status
        self.status_var = tk.StringVar(value="Status: Connected")
        self.status_label = tk.Label(control_frame, textvariable=self.status_var,
                                     fg="green", font=("Arial", 12, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Control buttons
        self.start_button = tk.Button(control_frame, text="Start Logging",
                                      command=self.start_logging, bg="green", fg="white")
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.pause_button = tk.Button(control_frame, text="Pause",
                                      command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=10)

        self.clear_button = tk.Button(control_frame, text="Clear Data",
                                      command=self.clear_data)
        self.clear_button.pack(side=tk.LEFT, padx=10)

        # Data info
        self.data_info_var = tk.StringVar(value="Data Points: 0 | Callbacks: 0")
        self.data_info_label = tk.Label(control_frame, textvariable=self.data_info_var)
        self.data_info_label.pack(side=tk.RIGHT, padx=10)

        # Current values frame (added height and status)
        values_frame = tk.LabelFrame(self.root, text="Current Filtered Values", padx=10, pady=5)
        values_frame.pack(fill=tk.X, padx=10, pady=5)

        # Current value labels (Delta X, Delta Y, and Height)
        self.delta_x_var = tk.StringVar(value="Delta X: N/A")
        self.delta_y_var = tk.StringVar(value="Delta Y: N/A")
        self.height_var = tk.StringVar(value="Height: N/A")
        self.height_status_var = tk.StringVar(value="Status: Waiting...")

        tk.Label(values_frame, textvariable=self.delta_x_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=20)
        tk.Label(values_frame, textvariable=self.delta_y_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=20)
        tk.Label(values_frame, textvariable=self.height_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=20)
        tk.Label(values_frame, textvariable=self.height_status_var, font=("Arial", 10, "bold"), fg="orange").pack(
            side=tk.LEFT, padx=20)

        # Filter info
        filter_info = f"Filters: MA({MOVING_AVG_WINDOW}) + EMA({EMA_ALPHA}) + Threshold(X:{THRESHOLD_DELTA_X}, Y:{THRESHOLD_DELTA_Y})"
        height_info = f"Height Range: {MIN_HEIGHT:.2f}m - {MAX_HEIGHT:.2f}m"
        tk.Label(values_frame, text=filter_info, font=("Arial", 8), fg="gray").pack(side=tk.RIGHT, padx=10)
        tk.Label(values_frame, text=height_info, font=("Arial", 8), fg="blue").pack(side=tk.RIGHT, padx=10)

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

        # Matplotlib figure with 3 subplots
        self.fig = Figure(figsize=(12, 8))

        # Create subplots (3 now)
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        self.ax3 = self.fig.add_subplot(3, 1, 3)

        # Setup plots
        self.setup_plots()

        # Embed plot in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_plots(self):
        """Setup the matplotlib plots"""
        # Delta X plot
        self.ax1.set_title("Delta X (Flow X measurement) - Filtered", fontsize=12)
        self.ax1.set_ylabel("Delta X")
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Delta X (Filtered)')
        self.ax1.legend()

        # Delta Y plot
        self.ax2.set_title("Delta Y (Flow Y measurement) - Filtered", fontsize=12)
        self.ax2.set_ylabel("Delta Y")
        self.ax2.grid(True, alpha=0.3)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2, label='Delta Y (Filtered)')
        self.ax2.legend()

        # Height plot
        self.ax3.set_title("Height (stateEstimate.z)", fontsize=12)
        self.ax3.set_xlabel("Time (seconds)")
        self.ax3.set_ylabel("Height (m)")
        self.ax3.grid(True, alpha=0.3)
        self.line3, = self.ax3.plot([], [], 'g-', linewidth=2, label='Height')
        # Add horizontal lines for height limits
        self.ax3.axhline(y=MIN_HEIGHT, color='orange', linestyle='--', alpha=0.7, label=f'Min Height ({MIN_HEIGHT}m)')
        self.ax3.axhline(y=MAX_HEIGHT, color='orange', linestyle='--', alpha=0.7, label=f'Max Height ({MAX_HEIGHT}m)')
        self.ax3.axhline(y=TARGET_HEIGHT, color='red', linestyle='-', alpha=0.8, label=f'Target ({TARGET_HEIGHT}m)')
        self.ax3.legend()

        # Adjust layout
        self.fig.tight_layout()

    def setup_logging(self):
        """Setup logging configuration"""
        print("Setting up logging configuration...")

        # Create log configuration for motion data (deltaX, deltaY, and height)
        self.log_motion = LogConfig(name="Motion", period_in_ms=20)  # 50Hz

        try:
            # Check if variables exist before adding them
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
                    self.log_motion.add_variable(var_name, var_type)
                    added_vars.append(var_name)
                    print(f"✓ Added variable: {var_name}")
                else:
                    print(f"✗ Variable not found: {var_name}")

            if len(added_vars) < 3:
                print("ERROR: Not all required variables found!")
                self.status_var.set("Status: Missing variables")
                return

            print(f"Successfully added {len(added_vars)} variables")

        except Exception as e:
            print(f"Error setting up variables: {e}")
            self.status_var.set(f"Status: Setup Error - {e}")
            return

        # Set up callbacks
        self.log_motion.data_received_cb.add_callback(self.motion_callback)
        self.log_motion.error_cb.add_callback(self.logging_error_callback)
        self.log_motion.started_cb.add_callback(self.logging_started_callback)
        self.log_motion.added_cb.add_callback(self.logging_added_callback)

        print("Logging configuration setup complete!")

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
        """Callback function to handle motion data"""
        try:
            global current_height
            self.data_received_count += 1
            log_data.update(data)

            # Extract raw data
            raw_delta_x = data.get('motion.deltaX', 0)
            raw_delta_y = data.get('motion.deltaY', 0)
            current_height = data.get('stateEstimate.z', 0)

            # Apply enhanced filtering to motion data
            filtered_delta_x = apply_combined_filter('motion.deltaX', raw_delta_x)
            filtered_delta_y = apply_combined_filter('motion.deltaY', raw_delta_y)

            # Check if height is within valid range
            height_valid = is_height_valid()

            # Update UI labels
            if height_valid:
                # Show filtered values when height is valid
                self.delta_x_var.set(f"Delta X: {filtered_delta_x:.3f}")
                self.delta_y_var.set(f"Delta Y: {filtered_delta_y:.3f}")
                self.height_status_var.set("Status: ACTIVE")
            else:
                # Show "N/A" when height is out of range
                self.delta_x_var.set("Delta X: N/A (Height)")
                self.delta_y_var.set("Delta Y: N/A (Height)")
                self.height_status_var.set("Status: HEIGHT OUT OF RANGE")

            # Always show current height in both displays
            self.height_var.set(f"Height: {current_height:.3f}m")

            # Update ToF display window
            self.tof_display_var.set(f"ToF Height: {current_height:.3f}m")
            self.tof_count_var.set(f"Readings: {self.data_received_count}")

            if height_valid:
                self.tof_status_var.set("Status: IN RANGE")
                self.tof_status_label.config(fg="green")
            else:
                self.tof_status_var.set("Status: OUT OF RANGE")
                self.tof_status_label.config(fg="red")

            # Update data info
            self.data_info_var.set(f"Data Points: {len(timestamps)} | Callbacks: {self.data_received_count}")

            # Print to console (every 50th callback to avoid spam)
            if self.data_received_count % 50 == 0:
                status = "ACTIVE" if height_valid else "OUT_OF_RANGE"
                print(f"Motion #{self.data_received_count} - Height: {current_height:.3f}m [{status}] | "
                      f"DX: {filtered_delta_x:.3f}, DY: {filtered_delta_y:.3f}")

        except Exception as e:
            print(f"Error in motion callback: {e}")

    def update_plot(self, frame):
        """Update the plots with new data"""
        if not self.logging_active or self.paused:
            return [self.line1, self.line2, self.line3]

        current_time = time.time() - start_time
        timestamps.append(current_time)

        # Update data history
        height_valid = is_height_valid()

        if height_valid:
            # Only add ALL values (including height) when height is valid
            data_history['motion.deltaX'].append(previous_output['motion.deltaX'])
            data_history['motion.deltaY'].append(previous_output['motion.deltaY'])
            data_history['stateEstimate.z'].append(current_height)
        else:
            # Add None/zero values when height is invalid (will show as gaps in plot)
            data_history['motion.deltaX'].append(0)
            data_history['motion.deltaY'].append(0)
            data_history['stateEstimate.z'].append(0)  # Don't plot height when invalid

        # Trim data to max_points
        if len(timestamps) > max_points:
            timestamps.pop(0)
            for param in data_history:
                data_history[param].pop(0)

        # Update plot lines
        if timestamps:
            self.line1.set_data(timestamps, data_history['motion.deltaX'])
            self.line2.set_data(timestamps, data_history['motion.deltaY'])
            self.line3.set_data(timestamps, data_history['stateEstimate.z'])

            # Adjust plot limits
            time_range = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1
            time_margin = time_range * 0.05

            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_xlim(min(timestamps) - time_margin, max(timestamps) + time_margin)

            # Set Y limits for each plot
            if data_history['motion.deltaX']:
                x_data = [x for x in data_history['motion.deltaX'] if x != 0]  # Exclude zeros
                if x_data:
                    x_range = max(x_data) - min(x_data) if len(set(x_data)) > 1 else 0.1
                    x_margin = max(x_range * 0.1, 0.01)
                    self.ax1.set_ylim(min(x_data) - x_margin, max(x_data) + x_margin)

            if data_history['motion.deltaY']:
                y_data = [y for y in data_history['motion.deltaY'] if y != 0]  # Exclude zeros
                if y_data:
                    y_range = max(y_data) - min(y_data) if len(set(y_data)) > 1 else 0.1
                    y_margin = max(y_range * 0.1, 0.01)
                    self.ax2.set_ylim(min(y_data) - y_margin, max(y_data) + y_margin)

            if data_history['stateEstimate.z']:
                z_data = [z for z in data_history['stateEstimate.z'] if z != 0]  # Exclude zeros
                if z_data:
                    z_range = max(z_data) - min(z_data) if len(set(z_data)) > 1 else 0.1
                    z_margin = max(z_range * 0.1, 0.05)
                    self.ax3.set_ylim(min(z_data) - z_margin, max(z_data) + z_margin)
                else:
                    # Set default range around target height when no valid data
                    self.ax3.set_ylim(0.1, 0.3)

        return [self.line1, self.line2, self.line3]

    def start_logging(self):
        """Start motion logging"""
        global start_time

        print("Starting motion logging...")

        try:
            # Add log configuration
            print("Adding log configuration to Crazyflie...")
            self.cf.log.add_config(self.log_motion)

            # Wait a moment for the configuration to be processed
            time.sleep(0.5)

            # Check if the configuration is valid
            if not self.log_motion.valid:
                print("ERROR: Log configuration is not valid!")
                self.status_var.set("Status: Invalid log configuration")
                return

            print("✓ Log configuration is valid")

            # Start logging
            print("Starting log configuration...")
            self.log_motion.start()

            # Wait a moment for logging to start
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

        except Exception as e:
            print(f"ERROR starting logging: {e}")
            self.stop_logging()
            self.status_var.set(f"Status: Error - {e}")

    def toggle_pause(self):
        """Toggle pause/resume logging"""
        if self.paused:
            # Resume
            self.log_motion.start()
            self.paused = False
            self.pause_button.config(text="Pause")
            self.status_var.set("Status: Logging Active")
            print("Logging resumed")
        else:
            # Pause
            self.log_motion.stop()
            self.paused = True
            self.pause_button.config(text="Resume")
            self.status_var.set("Status: Paused")
            print("Logging paused")

    def clear_data(self):
        """Clear all collected data"""
        global start_time

        # Clear data history
        for param in data_history:
            data_history[param].clear()
        timestamps.clear()

        # Reset filter buffers
        for param in moving_avg_buffer:
            moving_avg_buffer[param].clear()

        # Reset filter states
        for param in ema_filtered:
            ema_filtered[param] = 0.0

        for param in previous_output:
            previous_output[param] = 0.0

        # Reset height
        global current_height
        current_height = 0.0

        # Reset counters
        self.data_received_count = 0

        # Reset start time
        if self.logging_active:
            start_time = time.time()

        # Update UI
        self.data_info_var.set("Data Points: 0 | Callbacks: 0")

        # Clear plots
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()

        self.setup_plots()
        self.canvas.draw()

        print("Data and filters cleared")

    def stop_logging(self):
        """Stop motion logging"""
        if self.logging_active:
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
    print("Starting Motion Data Logger (Delta X/Y Only - Enhanced Filtering)...")
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