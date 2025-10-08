import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Configuration parameters
DRONE_URI = "udp://192.168.43.42"
TARGET_HEIGHT = 0.2  # Target hover height in meters
TAKEOFF_TIME = 1.0  # Time to takeoff and stabilize
HOVER_DURATION = 3.0  # How long to hover with position hold
LANDING_TIME = 0.5  # Time to land

# Trim corrections (from your working values)
TRIM_VX = 0.1  # Forward/backward trim correction
TRIM_VY = -0.08  # Left/right trim correction

# Improved motion sensor parameters
SENSITIVITY = 8  # Increased from 8 for faster response
MAX_CORRECTION = 0.25  # Increased from 0.15 for stronger corrections
VELOCITY_THRESHOLD = 0.005  # Reduced from 0.008 for more sensitivity

# NEW: Position integration parameters
POSITION_GAIN = 0.5  # How much to correct based on accumulated position drift
MAX_POSITION_ERROR = 0.05  # Maximum position error in meters
INTEGRATION_TIME = 0.1  # Time window for position integration

# Velocity calculation parameters
DEG_TO_RAD = 3.14159 / 180.0
SENSOR_PERIOD_MS = 10  # Reduced from 20ms for faster updates
DT = SENSOR_PERIOD_MS / 1000.0

# Global variables for sensor data
current_height = 0.0
motion_delta_x = 0
motion_delta_y = 0
sensor_data_ready = False

# Improved filtering - reduced history for faster response
velocity_x_history = [0.0, 0.0]  # Reduced from 3 points
velocity_y_history = [0.0, 0.0]

# Position tracking variables
accumulated_position_x = 0.0
accumulated_position_y = 0.0
last_update_time = time.time()

# Variables to track calculated velocities
current_vx = 0.0
current_vy = 0.0

# NEW: Predictive velocity tracking
velocity_trend_x = 0.0
velocity_trend_y = 0.0
last_vx = 0.0
last_vy = 0.0


def calculate_velocity(delta_value, altitude):
    """Convert delta value to linear velocity"""
    if altitude <= 0:
        return 0.0

    velocity_constant = (4.2 * DEG_TO_RAD) / (30.0 * DT)
    velocity = delta_value * altitude * velocity_constant
    return velocity


def smooth_velocity(new_velocity, history):
    """Apply 2-point smoothing filter for faster response"""
    history[1] = history[0]
    history[0] = new_velocity

    # Weighted average favoring recent data
    smoothed = (history[0] * 0.7) + (history[1] * 0.3)

    if abs(smoothed) < VELOCITY_THRESHOLD:
        smoothed = 0.0

    return smoothed


def update_position_tracking(vx, vy):
    """Track accumulated position drift"""
    global accumulated_position_x, accumulated_position_y, last_update_time
    global velocity_trend_x, velocity_trend_y, last_vx, last_vy

    current_time = time.time()
    dt = current_time - last_update_time

    if dt > 0.001:  # Avoid division by very small numbers
        # Update position estimates
        accumulated_position_x += vx * dt
        accumulated_position_y += vy * dt

        # Calculate velocity trends for prediction
        velocity_trend_x = (vx - last_vx) / dt
        velocity_trend_y = (vy - last_vy) / dt

        # Limit position error accumulation
        accumulated_position_x = max(-MAX_POSITION_ERROR,
                                     min(MAX_POSITION_ERROR, accumulated_position_x))
        accumulated_position_y = max(-MAX_POSITION_ERROR,
                                     min(MAX_POSITION_ERROR, accumulated_position_y))

        last_vx = vx
        last_vy = vy
        last_update_time = current_time


def motion_callback(timestamp, data, logconf):
    """Callback to receive motion sensor data"""
    global current_height, motion_delta_x, motion_delta_y, sensor_data_ready, current_vx, current_vy

    current_height = data.get('stateEstimate.z', 0)
    motion_delta_x = data.get('motion.deltaX', 0)
    motion_delta_y = data.get('motion.deltaY', 0)
    sensor_data_ready = True

    # Calculate and store current velocities for logging
    raw_velocity_x = calculate_velocity(motion_delta_x, current_height)
    raw_velocity_y = calculate_velocity(motion_delta_y, current_height)

    current_vx = smooth_velocity(raw_velocity_x, velocity_x_history)
    current_vy = smooth_velocity(raw_velocity_y, velocity_y_history)

    # Update position tracking
    update_position_tracking(current_vx, current_vy)


def get_advanced_position_corrections():
    """Calculate enhanced position corrections with prediction"""
    if not sensor_data_ready or current_height <= 0:
        return 0.0, 0.0

    # Velocity-based corrections (immediate response)
    velocity_correction_vx = -current_vx * SENSITIVITY
    velocity_correction_vy = -current_vy * SENSITIVITY

    # Position-based corrections (accumulated drift)
    position_correction_vx = -accumulated_position_x * POSITION_GAIN
    position_correction_vy = -accumulated_position_y * POSITION_GAIN

    # Predictive corrections (anticipate future movement)
    prediction_time = 0.05  # Predict 50ms ahead
    predicted_vx = current_vx + (velocity_trend_x * prediction_time)
    predicted_vy = current_vy + (velocity_trend_y * prediction_time)
    predictive_correction_vx = -predicted_vx * (SENSITIVITY * 0.3)
    predictive_correction_vy = -predicted_vy * (SENSITIVITY * 0.3)

    # Combine all corrections
    total_correction_vx = (velocity_correction_vx +
                           position_correction_vx +
                           predictive_correction_vx)
    total_correction_vy = (velocity_correction_vy +
                           position_correction_vy +
                           predictive_correction_vy)

    # Apply limits
    total_correction_vx = max(-MAX_CORRECTION, min(MAX_CORRECTION, total_correction_vx))
    total_correction_vy = max(-MAX_CORRECTION, min(MAX_CORRECTION, total_correction_vy))

    return total_correction_vx, total_correction_vy


def setup_logging(cf):
    """Setup motion sensor logging with faster update rate"""
    print("Setting up motion sensor logging...")

    log_motion = LogConfig(name="Motion", period_in_ms=SENSOR_PERIOD_MS)

    try:
        # Check TOC like in your first script
        toc = cf.log.toc.toc
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
                    log_motion.add_variable(var_name, var_type)
                    added_vars.append(var_name)
                    print(f"✓ Added variable: {var_name}")
                except Exception as e:
                    print(f"✗ Failed to add {var_name}: {e}")
            else:
                print(f"✗ Variable not found: {var_name}")

        if len(added_vars) < 2:
            print("ERROR: Not enough required variables found!")
            return None

        log_motion.data_received_cb.add_callback(motion_callback)

        cf.log.add_config(log_motion)
        time.sleep(0.5)

        if not log_motion.valid:
            print("ERROR: Log configuration is not valid!")
            return None

        log_motion.start()
        time.sleep(0.5)

        print(f"✓ Motion logging started at {SENSOR_PERIOD_MS}ms intervals")
        return log_motion

    except Exception as e:
        print(f"✗ Error setting up logging: {e}")
        return None


def main():
    """Main flight sequence using SyncCrazyflie"""

    # Initialize CRTP drivers
    cflib.crtp.init_drivers()

    print("=== Advanced Responsive Position Hold Flight ===")
    print(f"Target height: {TARGET_HEIGHT}m")
    print(f"Trim: vx={TRIM_VX}, vy={TRIM_VY}")
    print(f"Sensor update rate: {SENSOR_PERIOD_MS}ms")
    print(f"Enhanced sensitivity: {SENSITIVITY}")

    # Create Crazyflie instance
    cf = Crazyflie(rw_cache='./cache')
    log_config = None

    try:
        # Use SyncCrazyflie context manager
        print("\nConnecting to drone...")
        with SyncCrazyflie(DRONE_URI, cf=cf) as scf:
            print("✓ Connected successfully!")
            time.sleep(1.0)

            # Setup logging (now cf is properly initialized)
            log_config = setup_logging(cf)
            if not log_config:
                print("✗ Failed to setup logging - continuing with basic trim only")
            else:
                time.sleep(1.0)  # Let logging stabilize

            # Send zero setpoint to unlock safety
            print("Unlocking safety...")
            cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.1)

            # Enable high-level commander
            cf.param.set_value('commander.enHighLevel', '1')
            print("High-level commander activated")
            time.sleep(0.5)

            # Phase 1: Quick Takeoff
            print(f"\nPhase 1: Taking off to {TARGET_HEIGHT}m...")
            start_time = time.time()
            last_print_time = 0

            while time.time() - start_time < TAKEOFF_TIME:
                cf.commander.send_hover_setpoint(TRIM_VX, TRIM_VY, 0, TARGET_HEIGHT)
                time.sleep(0.01)  # Reduced from 0.02 for faster updates

                # Print status every 0.2 seconds during takeoff
                elapsed = time.time() - start_time
                if elapsed - last_print_time >= 0.2:
                    if log_config and sensor_data_ready:
                        print(f"Takeoff: {current_height:.3f}m | VX: {current_vx:.3f} | VY: {current_vy:.3f}")
                    else:
                        print(f"Takeoff: {current_height:.3f}m | VX: N/A | VY: N/A")
                    last_print_time = elapsed

            print(f"✓ Takeoff complete! Height: {current_height:.3f}m")

            # Reset position tracking after takeoff
            global accumulated_position_x, accumulated_position_y, last_update_time
            accumulated_position_x = 0.0
            accumulated_position_y = 0.0
            last_update_time = time.time()

            # Phase 2: Enhanced Position Hold
            print(f"\nPhase 2: Enhanced position hold for {HOVER_DURATION}s...")
            start_time = time.time()
            last_print_time = 0

            while time.time() - start_time < HOVER_DURATION:
                # Get enhanced motion-based corrections
                if log_config and sensor_data_ready:
                    motion_vx, motion_vy = get_advanced_position_corrections()
                else:
                    motion_vx, motion_vy = 0.0, 0.0  # Fallback to no corrections

                # Combine trim + motion corrections
                total_vx = TRIM_VX + motion_vy
                total_vy = TRIM_VY + motion_vx

                # Send combined setpoint at higher frequency
                cf.commander.send_hover_setpoint(total_vx, total_vy, 0, TARGET_HEIGHT)
                time.sleep(0.02)  # Reduced from 0.05 for more responsive control

                # Print detailed status every 0.3 seconds
                elapsed = time.time() - start_time
                remaining = HOVER_DURATION - elapsed
                if elapsed - last_print_time >= 0.3:
                    if log_config and sensor_data_ready:
                        print(f"Hold: {remaining:.1f}s | H: {current_height:.3f}m | "
                              f"VX: {current_vx:.3f} VY: {current_vy:.3f} | "
                              f"PosErr: X={accumulated_position_x:.3f} Y={accumulated_position_y:.3f} | "
                              f"Corr: vx={motion_vx:.3f} vy={motion_vy:.3f}")
                    else:
                        print(f"Hold: {remaining:.1f}s | Height: {current_height:.3f}m | "
                              f"VX: N/A | VY: N/A | Mode: Basic trim only")
                    last_print_time = elapsed

            print("✓ Enhanced position hold complete")

            # Phase 3: Landing
            print("\nPhase 3: Landing...")
            start_time = time.time()
            last_print_time = 0

            while time.time() - start_time < LANDING_TIME:
                cf.commander.send_hover_setpoint(TRIM_VX, TRIM_VY, 0, 0)
                time.sleep(0.01)

                # Print status every 0.2 seconds during landing
                elapsed = time.time() - start_time
                if elapsed - last_print_time >= 0.2:
                    if log_config and sensor_data_ready:
                        print(f"Landing: {current_height:.3f}m | VX: {current_vx:.3f} | VY: {current_vy:.3f}")
                    else:
                        print(f"Landing: {current_height:.3f}m | VX: N/A | VY: N/A")
                    last_print_time = elapsed

            print(f"\n✓ Landed! Final height: {current_height:.3f}m")

            # Stop motors
            cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.1)

            print("\n=== Flight Complete ===")
            if log_config and sensor_data_ready:
                print(f"Final VX: {current_vx:.3f} m/s, Final VY: {current_vy:.3f} m/s")
                print(f"Final position error: X={accumulated_position_x:.3f}m, Y={accumulated_position_y:.3f}m")

            # Cleanup logging before exiting context
            if log_config:
                try:
                    log_config.stop()
                    log_config = None
                except:
                    pass

    except Exception as e:
        print(f"\n✗ Error during flight: {e}")

    finally:
        # Additional cleanup if needed
        if log_config:
            try:
                log_config.stop()
            except:
                pass


if __name__ == "__main__":
    main()