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

# Initialize CRTP driverscf
cflib.crtp.init_drivers()

# Shared data dictionary and storage for plotting
log_data = {}
data_history = {param: [] for param in [
    'stateEstimate.pitch', 'stateEstimate.roll', 'stateEstimate.yaw',
    'pwm.m1_pwm', 'pwm.m2_pwm', 'pwm.m3_pwm', 'pwm.m4_pwm',
    'controller.cmd_pitch', 'controller.cmd_roll', 'controller.cmd_yaw',
    'controller.pitchRate', 'controller.rollRate', 'controller.yawRate'
]}
timestamps = []
start_time = None
max_points = 200  # Limit the number of points to keep the plot responsive

# Parameter labels and custom colors
param_labels = {
    'stateEstimate.pitch': 'K_Pitch (deg)',
    'stateEstimate.roll': 'K_Roll (deg)',
    'stateEstimate.yaw': 'K_Yaw (deg)',
    'pwm.m1_pwm': 'M1 (PWM)',
    'pwm.m2_pwm': 'M2 (PWM)',
    'pwm.m3_pwm': 'M3 (PWM)',
    'pwm.m4_pwm': 'M4 (PWM)',
    'controller.cmd_pitch': 'Cmd_Pitch (deg)',
    'controller.cmd_roll': 'Cmd_Roll (deg)',
    'controller.cmd_yaw': 'Cmd_Yaw (deg)',
    'controller.pitchRate': 'D_PitchRate',
    'controller.rollRate': 'D_RollRate',
    'controller.yawRate': 'D_YawRate'
}

# Custom color palette with distinct motor colors
param_colors = {
    'stateEstimate.pitch': 'blue',        # K_Pitch: Blue
    'stateEstimate.roll': 'lightblue',    # K_Roll: Light Blue
    'stateEstimate.yaw': 'darkblue',      # K_Yaw: Dark Blue
    'pwm.m1_pwm': 'green',                # M1: Green
    'pwm.m2_pwm': 'orange',               # M2: Orange
    'pwm.m3_pwm': 'teal',                 # M3: Teal
    'pwm.m4_pwm': 'saddlebrown',          # M4: Brown
    'controller.cmd_pitch': 'red',        # Cmd_Pitch: Red
    'controller.cmd_roll': 'lightcoral',  # Cmd_Roll: Light Coral
    'controller.cmd_yaw': 'darkred',      # Cmd_Yaw: Dark Red
    'controller.pitchRate': 'purple',     # D_PitchRate: Purple
    'controller.rollRate': 'violet',      # D_RollRate: Violet
    'controller.yawRate': 'darkviolet'    # D_YawRate: Dark Violet
}

class DroneLoggerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Parameter Logger")
        self.cf = Crazyflie(rw_cache='./cache')
        self.scf = None
        self.logging_active = False
        self.paused = False
        self.anim = None

        # Connect to drone and arm it on launch
        try:
            self.scf = SyncCrazyflie(DRONE_URI, cf=self.cf)
            self.scf.__enter__()
            print("Connecting to drone...")
            time.sleep(2.0)
            # Arm the drone with zero setpoint
            print("Arming drone with zero thrust...")
            self.cf.commander.send_setpoint(0, 0, 0, 0)  # roll, pitch, yaw, thrust
            time.sleep(0.1)
        except Exception as e:
            print(f"Error connecting to drone: {e}")
            self.on_closing()

        # UI Elements
        # Battery display label
        self.battery_var = tk.StringVar(value="Battery: N/A V")
        self.battery_label = tk.Label(root, textvariable=self.battery_var)
        self.battery_label.pack(anchor='ne')  # Top-right corner

        self.start_button = tk.Button(root, text="Start", command=self.start_logging)
        self.start_button.pack()

        self.pause_button = tk.Button(root, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack()

        # Thrust slider
        self.thrust_var = tk.IntVar(value=0)
        self.thrust_slider = tk.Scale(root, from_=0, to=60000, orient=tk.HORIZONTAL, label="Thrust", variable=self.thrust_var, length=300)
        self.thrust_slider.pack()

        # Checkboxes for selecting parameters to plot
        self.plot_vars = {}
        checkbox_frame = tk.Frame(root)
        checkbox_frame.pack()
        for param, label in param_labels.items():
            var = tk.BooleanVar(value=True)  # Default to True (all plotted)
            self.plot_vars[param] = var
            tk.Checkbutton(checkbox_frame, text=label, variable=var).pack(side=tk.LEFT)
            # Bind callback to update plot visibility when checkbox changes
            var.trace_add("write", lambda *args, p=param: self.update_plot_visibility(p))

        # Matplotlib figure
        self.fig = Figure(figsize=(10, 5))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")
        self.lines = {param: self.ax.plot([], [], label=label, color=param_colors[param])[0] for param, label in param_labels.items()}
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        self.ax.grid(True)
        self.fig.subplots_adjust(right=0.75)  # Make space for the legend

        # Embed plot in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        # LogConfigs
        self.log_battery_state = LogConfig(name="BatteryState", period_in_ms=50)
        self.log_battery_state.add_variable('pm.vbat', 'float')
        self.log_battery_state.add_variable('stateEstimate.pitch', 'float')
        self.log_battery_state.add_variable('stateEstimate.roll', 'float')
        self.log_battery_state.add_variable('stateEstimate.yaw', 'float')

        self.log_motor = LogConfig(name="Motor", period_in_ms=50)
        self.log_motor.add_variable('pwm.m1_pwm', 'uint32_t')
        self.log_motor.add_variable('pwm.m2_pwm', 'uint32_t')
        self.log_motor.add_variable('pwm.m3_pwm', 'uint32_t')
        self.log_motor.add_variable('pwm.m4_pwm', 'uint32_t')

        self.log_controller = LogConfig(name="Controller", period_in_ms=50)
        self.log_controller.add_variable('controller.cmd_pitch', 'float')
        self.log_controller.add_variable('controller.cmd_roll', 'float')
        self.log_controller.add_variable('controller.cmd_yaw', 'float')
        self.log_controller.add_variable('controller.pitchRate', 'float')
        self.log_controller.add_variable('controller.rollRate', 'float')
        self.log_controller.add_variable('controller.yawRate', 'float')

        # Set up callbacks
        self.log_battery_state.data_received_cb.add_callback(self.battery_state_callback)
        self.log_motor.data_received_cb.add_callback(self.motor_callback)
        self.log_controller.data_received_cb.add_callback(self.controller_callback)

    def update_plot_visibility(self, param):
        # Update the visibility of the plot line when checkbox changes
        self.lines[param].set_visible(self.plot_vars[param].get())
        # Adjust y-axis limits based on visible data
        visible_data = [data_history[p] for p, var in self.plot_vars.items() if var.get() and data_history[p]]
        if visible_data:
            min_val = min(min(data) for data in visible_data)
            max_val = max(max(data) for data in visible_data)
            self.ax.set_ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))
        else:
            self.ax.set_ylim(-1, 1)
        self.canvas.draw()

    def battery_state_callback(self, timestamp, data, logconf):
        log_data.update(data)

    def motor_callback(self, timestamp, data, logconf):
        log_data.update(data)

    def controller_callback(self, timestamp, data, logconf):
        log_data.update(data)
        # Update battery display
        battery_value = log_data.get('pm.vbat', 'N/A')
        self.battery_var.set(f"Battery: {battery_value:.2f} V" if isinstance(battery_value, float) else "Battery: N/A V")

        # Print to console
        output = []
        output.append(f"Battery: {log_data.get('pm.vbat', 'N/A'):.2f} V")
        output.append(f"K_Pitch: {log_data.get('stateEstimate.pitch', 'N/A'):.2f} deg")
        output.append(f"K_Roll: {log_data.get('stateEstimate.roll', 'N/A'):.2f} deg")
        output.append(f"K_Yaw: {log_data.get('stateEstimate.yaw', 'N/A'):.2f} deg")
        output.append(f"M1: {log_data.get('pwm.m1_pwm', 'N/A')} PWM")
        output.append(f"M2: {log_data.get('pwm.m2_pwm', 'N/A')} PWM")
        output.append(f"M3: {log_data.get('pwm.m3_pwm', 'N/A')} PWM")
        output.append(f"M4: {log_data.get('pwm.m4_pwm', 'N/A')} PWM")
        output.append(f"Cmd_Pitch: {log_data.get('controller.cmd_pitch', 'N/A'):.2f} deg")
        output.append(f"Cmd_Roll: {log_data.get('controller.cmd_roll', 'N/A'):.2f} deg")
        output.append(f"Cmd_Yaw: {log_data.get('controller.cmd_yaw', 'N/A'):.2f} deg")
        output.append(f"D_PitchRate: {log_data.get('controller.pitchRate', 'N/A'):.2f}")
        output.append(f"D_RollRate: {log_data.get('controller.rollRate', 'N/A'):.2f}")
        output.append(f"D_YawRate: {log_data.get('controller.yawRate', 'N/A'):.2f}")
        print(", ".join(output))

        # Send thrust setpoint if logging is active and not paused
        if self.logging_active and not self.paused:
            thrust = self.thrust_var.get()
            self.cf.commander.send_setpoint(0, 0, 0, thrust)

    def update_plot(self, frame):
        if not self.logging_active or self.paused:
            return self.lines.values()

        current_time = time.time() - start_time
        timestamps.append(current_time)

        # Update data history
        for param in data_history:
            value = log_data.get(param, 0.0)  # Default to 0 if not available
            data_history[param].append(value)

        # Trim data to max_points
        if len(timestamps) > max_points:
            timestamps.pop(0)
            for param in data_history:
                data_history[param].pop(0)

        # Update plot lines
        for param, line in self.lines.items():
            if self.plot_vars[param].get():  # Only plot if checkbox is selected
                line.set_data(timestamps, data_history[param])
                line.set_visible(True)
            else:
                line.set_visible(False)

        # Adjust plot limits
        if timestamps:
            self.ax.set_xlim(timestamps[0], timestamps[-1])
            visible_data = [data_history[param] for param, var in self.plot_vars.items() if var.get() and data_history[param]]
            if visible_data:
                min_val = min(min(data) for data in visible_data)
                max_val = max(max(data) for data in visible_data)
                self.ax.set_ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))
            else:
                self.ax.set_ylim(-1, 1)

        return self.lines.values()

    def start_logging(self):
        global start_time
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)

        # Start logging
        try:
            self.cf.log.add_config(self.log_battery_state)
            self.cf.log.add_config(self.log_motor)
            self.cf.log.add_config(self.log_controller)

            self.log_battery_state.start()
            self.log_motor.start()
            self.log_controller.start()
            self.logging_active = True
            start_time = time.time()

            # Start plot animation
            self.anim = animation.FuncAnimation(self.fig, self.update_plot, interval=50, blit=True)
            self.canvas.draw()

        except Exception as e:
            print(f"Error: {e}")
            self.stop_logging()

    def toggle_pause(self):
        if self.paused:
            # Resume
            self.log_battery_state.start()
            self.log_motor.start()
            self.log_controller.start()
            self.paused = False
            self.pause_button.config(text="Pause")
        else:
            # Pause
            self.log_battery_state.stop()
            self.log_motor.stop()
            self.log_controller.stop()
            self.paused = True
            self.pause_button.config(text="Resume")
            # Send zero thrust when pausing
            self.cf.commander.send_setpoint(0, 0, 0, 0)

    def stop_logging(self):
        if self.logging_active:
            self.log_battery_state.stop()
            self.log_motor.stop()
            self.log_controller.stop()
            self.logging_active = False
            # Send zero thrust when stopping
            self.cf.commander.send_setpoint(0, 0, 0, 0)
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)

    def on_closing(self):
        self.stop_logging()
        if self.scf:
            self.scf.__exit__(None, None, None)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DroneLoggerUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
    print("Data collection complete")