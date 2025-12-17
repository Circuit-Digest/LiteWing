"""Standalone optical flow sensor test for the LiteWing flight stabilizer shield.

Streams raw optical flow deltas, computed velocities, and the integrated XY
trajectory to both the console and a matplotlib-enabled Tk GUI.
"""

import math
import threading
import time
from collections import deque
import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

DRONE_URI = "udp://192.168.43.42"
LOG_PERIOD_MS = 50 # 20 Hz logging interval
DT = LOG_PERIOD_MS / 1000.0 # DT is the logging interval converted to seconds; used when integrating velocities to compute position displacement between samples.
DEG_TO_RAD = math.pi / 180.0 # conversion factor from degrees to radians
ALPHA = 0.7 # IIR smoothing factor for velocity values (0 < ALPHA < 1)
VELOCITY_THRESHOLD = 0.005  # m/s - velocities below this are clamped to zero
INVERT_X_AXIS_DEFAULT = True # sensor is mounted inverted on the LiteWing shield
HISTORY_LENGTH = 400 # number of samples to keep in history for plotting


def calculate_velocity(delta_value: int, altitude_m: float) -> float:
    """
    Convert a raw optical-flow delta reading into a velocity (m/s).

    The flow sensor gives pixel deltas per frame; to convert to meters per second
    we take into account the altitude above ground and the (approximate)
    optical geometry. The conversion is linear with altitude: at higher altitude,
    the same angular delta corresponds to a larger ground displacement.

    Args:
        delta_value: Raw integer optical-flow delta reported by sensor (counts).
        altitude_m: Estimated altitude in meters from the state estimator.

    Returns:
        Velocity in meters per second along the given axis.
    """
    # If altitude is not positive we can't compute velocity reliably
    if altitude_m <= 0:
        return 0.0

    # velocity_constant encodes conversion factors (sensor FoV, resolution,
    # and logging period) into a per-sample scaling factor. 
    velocity_constant = (5.4 * DEG_TO_RAD) / (30.0 * DT)    # 5.4° = sensor field of view 30   = pixel resolution DT   = sample time (5 ms typical)
    return delta_value * altitude_m * velocity_constant


def smooth_velocity(new_value: float, history: list[float]) -> float:
    """
    Simple IIR smoothing for velocity values.

    The function takes a new sample and a history list used to compute a
    first-order IIR (exponential) smoothing using alpha. The history is
    stored to allow the GUI to also display recent values. To avoid drift or
    reporting near-zero noise as movement, values below VELOCITY_THRESHOLD are
    clamped to zero.
    """
    history.append(new_value)
    if len(history) < 2:
        # Not enough history to smooth yet, return the raw value
        return new_value
    smoothed = history[-1] * ALPHA + history[-2] * (1 - ALPHA)
    # Reject very small velocities to avoid reacting to noise
    if abs(smoothed) < VELOCITY_THRESHOLD:
        return 0.0
    return smoothed


class OpticalFlowApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LiteWing Optical Flow Test")
        self.root.geometry("1100x760")

        # GUI state variables 
        self.status_var = tk.StringVar(value="Status: Idle")
        self.delta_var = tk.StringVar(value="ΔX: 0, ΔY: 0")
        self.velocity_var = tk.StringVar(value="Velocity XY: 0.000, 0.000 m/s")
        self.height_var = tk.StringVar(value="Height: 0.000 m")
        self.squal_var = tk.StringVar(value="Surface Quality: 0")

        # Sensor mount is inverted in the LiteWing hardware; keep a fixed flag
        # to map the raw delta to the GUI reported value.
        self.invert_x = bool(INVERT_X_AXIS_DEFAULT)

        self._build_controls()
        self._build_plot()

        # Threading primitives to start/stop the logging thread and protect
        # access to the shared history variables used by GUI and the logger
        self.stop_event = threading.Event()
        self.connection_thread: threading.Thread | None = None

        self.data_lock = threading.Lock()
        # Circular buffers preserving a fixed amount of history for plotting
        self.time_history = deque(maxlen=HISTORY_LENGTH)
        self.vx_history = deque(maxlen=HISTORY_LENGTH)
        self.vy_history = deque(maxlen=HISTORY_LENGTH)
        # Integrated XY trajectory in meters computed by integrating velocities
        self.pos_x_history = deque(maxlen=HISTORY_LENGTH)
        self.pos_y_history = deque(maxlen=HISTORY_LENGTH)
        # Initialize smoothing history for each axis
        self.smoothed_vx = [0.0]
        self.smoothed_vy = [0.0]
        # Position integration state (meters)
        self.position_x = 0.0
        self.position_y = 0.0
        self.last_sample_time: float | None = None
        self.last_console_print = 0.0

        # Schedule periodic GUI updates
        self.root.after(100, self._refresh_gui)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_controls(self) -> None:
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=6)

        # Start/Stop buttons to open/close the radio connection and begin
        # logging from the flight stabilizer's flow sensor.
        tk.Button(top_frame, text="Start", command=self.start, bg="#28a745", fg="white", width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Stop", command=self.stop, bg="#dc3545", fg="white", width=12).pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, textvariable=self.status_var, font=("Arial", 11, "bold"), fg="blue").pack(side=tk.LEFT, padx=20)

        value_frame = tk.Frame(self.root)
        value_frame.pack(fill=tk.X, padx=10, pady=6)

        tk.Label(value_frame, textvariable=self.delta_var, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        tk.Label(value_frame, textvariable=self.velocity_var, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        tk.Label(value_frame, textvariable=self.height_var, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        tk.Label(value_frame, textvariable=self.squal_var, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)

    def _build_plot(self) -> None:
        self.figure = Figure(figsize=(11, 6.5), dpi=100)
        self.ax_vel = self.figure.add_subplot(2, 1, 1)
        # First subplot shows the recent X/Y velocities (m/s)
        self.ax_vel.set_title("Velocities")
        self.ax_vel.set_ylabel("Velocity (m/s)")
        self.ax_vel.grid(True, alpha=0.3)
        (self.vx_line,) = self.ax_vel.plot([], [], label="VX", color="tab:red")
        (self.vy_line,) = self.ax_vel.plot([], [], label="VY", color="tab:green")
        self.ax_vel.legend(loc="upper right")

        self.ax_pos = self.figure.add_subplot(2, 1, 2)
        # Second subplot shows the integrated XY trajectory in meters
        self.ax_pos.set_title("Integrated Trajectory")
        self.ax_pos.set_xlabel("X (m)")
        self.ax_pos.set_ylabel("Y (m)")
        self.ax_pos.set_aspect("equal")
        self.ax_pos.grid(True, alpha=0.3)
        (self.trajectory_line,) = self.ax_pos.plot([], [], color="tab:blue", linewidth=2)
        (self.current_point,) = self.ax_pos.plot([], [], marker="o", color="tab:orange")

        canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas = canvas

    def start(self) -> None:
        # If the background connection thread is already running, do nothing
        if self.connection_thread and self.connection_thread.is_alive():
            return
        self.stop_event.clear()
        # Spawn a background thread which manages the Crazyflie connection
        # and receives logged optical flow data without blocking the GUI.
        self.connection_thread = threading.Thread(target=self._connection_worker, daemon=True)
        self.connection_thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _connection_worker(self) -> None:
        # This worker runs in a separate thread and does the radio I/O via the
        # cfclient library. It registers log variables, starts the log stream,
        # and then waits until the stop_event is triggered to clean up.
        self._set_status("Status: Connecting...")
        try:
            # Initialize the Crazyradio drivers (no debug radio by default)
            cflib.crtp.init_drivers(enable_debug_driver=False)
            with SyncCrazyflie(DRONE_URI, cf=Crazyflie(rw_cache="./cache")) as scf:
                cf = scf.cf
                self._set_status("Status: Connected")

                # Create a log config that retrieves sensor deltas and the
                # estimated height repeatedly at LOG_PERIOD_MS intervals.
                log_config = LogConfig(name="OpticalFlow", period_in_ms=LOG_PERIOD_MS)
                variables = [
                    ("motion.deltaX", "int16_t"),
                    ("motion.deltaY", "int16_t"),
                    ("motion.squal", "uint8_t"),
                    ("stateEstimate.z", "float"),
                ]

                # Query the CF TOC (Table of Contents) to ensure variables are
                # present in the running firmware, and add them to the log.
                if not self._add_variables_if_available(cf, log_config, variables):
                    self._set_status("Status: Optical flow variables unavailable")
                    return

                log_config.data_received_cb.add_callback(self._log_callback)
                cf.log.add_config(log_config)
                log_config.start()
                print("[Flow] Logging started")

                # Wait/multiplex until the GUI signals stop via stop_event.
                while not self.stop_event.is_set():
                    time.sleep(0.1)

                log_config.stop()
                print("[Flow] Logging stopped")
        except Exception as exc:  # noqa: BLE001
            print(f"[Flow] Connection error: {exc}")
            self._set_status("Status: Error - check console")
        finally:
            self._set_status("Status: Idle")

    def _add_variables_if_available(self, cf: Crazyflie, log_config: LogConfig, candidates: list[tuple[str, str]]) -> bool:
        # Access the Crazyflie Log TOC which describes available log variables
        toc = cf.log.toc.toc
        added = 0
        for full_name, var_type in candidates:
            group, name = full_name.split(".", maxsplit=1)
            if group in toc and name in toc[group]:
                log_config.add_variable(full_name, var_type)
                print(f"[Flow] Logging {full_name}")
                added += 1
            else:
                print(f"[Flow] Missing {full_name}")
        return added > 0

    def _log_callback(self, timestamp: int, data: dict, _: LogConfig) -> None:
        delta_x = data.get("motion.deltaX", 0)
        delta_y = data.get("motion.deltaY", 0)
        squal = data.get("motion.squal", 0)
        altitude = data.get("stateEstimate.z", 0.0)

        # Sensor is mounted inverted; map delta_x accordingly (maps raw -> GUI)
        delta_x_mapped = -delta_x if self.invert_x else delta_x
        raw_vx = calculate_velocity(delta_x_mapped, altitude)
        raw_vy = calculate_velocity(delta_y, altitude)
        vx = smooth_velocity(raw_vx, self.smoothed_vx)
        vy = smooth_velocity(raw_vy, self.smoothed_vy)

        # Timestamp the sample to compute a true delta-time between events
        now = time.time()
        # Compute time since last sample; clamp to a sane max to avoid
        # integration jumps if sampling pauses.
        if self.last_sample_time is None:
            dt = DT
        else:
            dt = max(0.0, min(now - self.last_sample_time, 0.5))
        self.last_sample_time = now

        # Integrate velocity -> position using the measured time delta
        self.position_x += vx * dt
        self.position_y += vy * dt

        with self.data_lock:
            # Append new sample to each history buffer used for plotting
            self.time_history.append(now)
            self.vx_history.append(vx)
            self.vy_history.append(vy)
            self.pos_x_history.append(self.position_x)
            self.pos_y_history.append(self.position_y)
            # Store mapped delta_x since the GUI will show the mapped axis
            self.latest_values = (delta_x_mapped, delta_y, vx, vy, altitude, squal)

        if time.time() - self.last_console_print >= 1.0:
            self.last_console_print = time.time()
            # Print raw delta and computed velocities to the console every
            # second for fast debugging without relying on the GUI.
            print(
                f"[Flow] ΔX={delta_x} ΔY={delta_y} | VX={vx:.3f} m/s VY={vy:.3f} m/s | "
                f"Height={altitude:.3f} m | Squal={squal}"
            )

    def _refresh_gui(self) -> None:
        with self.data_lock:
            if getattr(self, "latest_values", None):
                delta_x, delta_y, vx, vy, altitude, squal = self.latest_values
                self.delta_var.set(f"ΔX: {delta_x}, ΔY: {delta_y}")
                self.velocity_var.set(f"Velocity XY: {vx:.3f}, {vy:.3f} m/s")
                self.height_var.set(f"Height: {altitude:.3f} m")
                self.squal_var.set(f"Surface Quality: {squal}")

                # Build relative time axis for plotting
                times = list(self.time_history)
                if times:
                    t0 = times[0]
                    rel_times = [t - t0 for t in times]

                    vx_vals = list(self.vx_history)
                    vy_vals = list(self.vy_history)
                    pos_x = list(self.pos_x_history)
                    pos_y = list(self.pos_y_history)

                    # Update lines for velocities
                    self.vx_line.set_data(rel_times, vx_vals)
                    self.vy_line.set_data(rel_times, vy_vals)

                    # Keep the right-most 20 seconds visible in the velocity
                    # subplot for context while streaming.
                    last_time = rel_times[-1] if rel_times[-1] > 1 else 1
                    self.ax_vel.set_xlim(max(0, last_time - 20), last_time + 1)
                    # Combine velocities to compute symmetrical Y limits around
                    # the current min/max so the plots remain stable.
                    combined_vel = vx_vals + vy_vals
                    vmin = min(combined_vel) if combined_vel else -0.1
                    vmax = max(combined_vel) if combined_vel else 0.1
                    margin = max(0.05, (vmax - vmin) * 0.2)
                    self.ax_vel.set_ylim(vmin - margin, vmax + margin)

                    # Update trajectory plot with integrated positions (meters).
                    self.trajectory_line.set_data(pos_x, pos_y)
                    if pos_x and pos_y:
                        self.current_point.set_data([pos_x[-1]], [pos_y[-1]])
                        # Auto-zoom and center the trajectory plot while
                        # maintaining aspect ratio for accurate XY scaling.
                        xmin, xmax = min(pos_x), max(pos_x)
                        ymin, ymax = min(pos_y), max(pos_y)
                        span_x = max(0.1, xmax - xmin)
                        span_y = max(0.1, ymax - ymin)
                        center_x = (xmin + xmax) / 2
                        center_y = (ymin + ymax) / 2
                        pad_x = span_x * 0.4
                        pad_y = span_y * 0.4
                        self.ax_pos.set_xlim(center_x - pad_x, center_x + pad_x)
                        self.ax_pos.set_ylim(center_y - pad_y, center_y + pad_y)

                    self.canvas.draw_idle()

        self.root.after(100, self._refresh_gui)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _on_close(self) -> None:
        self.stop()
        self.root.after(200, self.root.destroy)


def main() -> None:
    root = tk.Tk()
    app = OpticalFlowApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
