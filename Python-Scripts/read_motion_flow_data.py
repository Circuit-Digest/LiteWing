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
    'motion.motion': []
}
timestamps = []
start_time = None
max_points = 500  # Keep more points for motion data

# Parameter labels and colors
param_labels = {
    'motion.deltaX': 'Delta X (Flow X)',
    'motion.deltaY': 'Delta Y (Flow Y)',
    'motion.motion': 'Motion Detected'
}

param_colors = {
    'motion.deltaX': 'blue',
    'motion.deltaY': 'red',
    'motion.motion': 'green'
}

# === EMA Filter Parameters ===
EMA_ALPHA = 0.2  # Smoothing factor, 0 < alpha <= 1 (user can tune this)

# Store previous filtered values
ema_filtered = {
    'motion.deltaX': 0.0,
    'motion.deltaY': 0.0
}


class MotionLoggerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Data Logger - Delta X & Delta Y")
        self.root.geometry("1200x800")
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
            target_vars = ['motion.deltaX', 'motion.deltaY', 'motion.motion']
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

        # Current values frame
        values_frame = tk.LabelFrame(self.root, text="Current Values", padx=10, pady=5)
        values_frame.pack(fill=tk.X, padx=10, pady=5)

        # Current value labels
        self.delta_x_var = tk.StringVar(value="Delta X: N/A")
        self.delta_y_var = tk.StringVar(value="Delta Y: N/A")
        self.motion_var = tk.StringVar(value="Motion: N/A")

        tk.Label(values_frame, textvariable=self.delta_x_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=20)
        tk.Label(values_frame, textvariable=self.delta_y_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=20)
        tk.Label(values_frame, textvariable=self.motion_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=20)

        # Matplotlib figure with subplots
        self.fig = Figure(figsize=(12, 8))

        # Create subplots
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
        self.ax1.set_title("Delta X (Flow X measurement)", fontsize=12)
        self.ax1.set_ylabel("Delta X")
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Delta X')
        self.ax1.legend()

        # Delta Y plot
        self.ax2.set_title("Delta Y (Flow Y measurement)", fontsize=12)
        self.ax2.set_ylabel("Delta Y")
        self.ax2.grid(True, alpha=0.3)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2, label='Delta Y')
        self.ax2.legend()

        # Motion detection plot
        self.ax3.set_title("Motion Detection", fontsize=12)
        self.ax3.set_xlabel("Time (seconds)")
        self.ax3.set_ylabel("Motion")
        self.ax3.grid(True, alpha=0.3)
        self.line3, = self.ax3.plot([], [], 'g-', linewidth=2, label='Motion')
        self.ax3.legend()

        # Adjust layout
        self.fig.tight_layout()

    def setup_logging(self):
        """Setup logging configuration"""
        print("Setting up logging configuration...")

        # Create log configuration for motion data
        self.log_motion = LogConfig(name="Motion", period_in_ms=20)  # 50Hz

        try:
            # Check if variables exist before adding them
            toc = self.cf.log.toc.toc

            variables_to_add = [
                ('motion.deltaX', 'int16_t'),
                ('motion.deltaY', 'int16_t'),
                ('motion.motion', 'uint8_t')
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

            if not added_vars:
                print("ERROR: No motion variables found!")
                self.status_var.set("Status: No motion variables available")
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
            self.data_received_count += 1
            log_data.update(data)

            # Extract data
            delta_x = data.get('motion.deltaX', 0)
            delta_y = data.get('motion.deltaY', 0)
            motion_detect = data.get('motion.motion', 0)

            # === Apply EMA filter ===
            global ema_filtered
            ema_filtered['motion.deltaX'] = EMA_ALPHA * delta_x + (1 - EMA_ALPHA) * ema_filtered['motion.deltaX']
            ema_filtered['motion.deltaY'] = EMA_ALPHA * delta_y + (1 - EMA_ALPHA) * ema_filtered['motion.deltaY']

            # Update UI labels (show filtered values)
            self.delta_x_var.set(f"Delta X: {ema_filtered['motion.deltaX']:.2f}")
            self.delta_y_var.set(f"Delta Y: {ema_filtered['motion.deltaY']:.2f}")
            self.motion_var.set(f"Motion: {motion_detect}")

            # Update data info
            self.data_info_var.set(f"Data Points: {len(timestamps)} | Callbacks: {self.data_received_count}")

            # Print to console (every 50th callback to avoid spam)
            if self.data_received_count % 50 == 0:
                print(
                    f"Motion #{self.data_received_count} - DeltaX: {ema_filtered['motion.deltaX']:.2f}, DeltaY: {ema_filtered['motion.deltaY']:.2f}, Motion: {motion_detect}")

        except Exception as e:
            print(f"Error in motion callback: {e}")

    def update_plot(self, frame):
        """Update the plots with new data"""
        if not self.logging_active or self.paused:
            return [self.line1, self.line2, self.line3]

        current_time = time.time() - start_time
        timestamps.append(current_time)

        # Update data history (use filtered values for deltaX and deltaY)
        data_history['motion.deltaX'].append(ema_filtered['motion.deltaX'])
        data_history['motion.deltaY'].append(ema_filtered['motion.deltaY'])
        data_history['motion.motion'].append(log_data.get('motion.motion', 0))

        # Trim data to max_points
        if len(timestamps) > max_points:
            timestamps.pop(0)
            for param in data_history:
                data_history[param].pop(0)

        # Update plot lines
        if timestamps:
            self.line1.set_data(timestamps, data_history['motion.deltaX'])
            self.line2.set_data(timestamps, data_history['motion.deltaY'])
            self.line3.set_data(timestamps, data_history['motion.motion'])

            # Adjust plot limits
            time_range = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1
            time_margin = time_range * 0.05

            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_xlim(min(timestamps) - time_margin, max(timestamps) + time_margin)

            # Set Y limits for each plot
            if data_history['motion.deltaX']:
                x_data = data_history['motion.deltaX']
                x_range = max(x_data) - min(x_data) if len(set(x_data)) > 1 else 1
                x_margin = x_range * 0.1 if x_range > 0 else 1
                self.ax1.set_ylim(min(x_data) - x_margin, max(x_data) + x_margin)

            if data_history['motion.deltaY']:
                y_data = data_history['motion.deltaY']
                y_range = max(y_data) - min(y_data) if len(set(y_data)) > 1 else 1
                y_margin = y_range * 0.1 if y_range > 0 else 1
                self.ax2.set_ylim(min(y_data) - y_margin, max(y_data) + y_margin)

            if data_history['motion.motion']:
                motion_data = data_history['motion.motion']
                self.ax3.set_ylim(min(motion_data) - 0.5, max(motion_data) + 0.5)

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

        print("Data cleared")

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
    print("Starting Motion Data Logger...")
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