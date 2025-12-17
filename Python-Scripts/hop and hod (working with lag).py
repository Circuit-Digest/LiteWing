import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Configuration parameters
DRONE_URI = "udp://192.168.43.42"
TARGET_HEIGHT = 0.2  # Target hover height in meters
TAKEOFF_TIME = 0.0  # Time to takeoff and stabilize
HOVER_DURATION = 8.0  # How long to hover with position hold
LANDING_TIME = 0  # Time to land

# Trim corrections (from your working values)
TRIM_VX = 0.15  # Forward/backward trim correction
TRIM_VY = -0.06  # Left/right trim correction

# Motion sensor parameters
SENSITIVITY = 15  # Position correction sensitivity
MAX_CORRECTION = 0.1  # Maximum position correction
VELOCITY_THRESHOLD = 0.005

# Velocity calculation parameters
DEG_TO_RAD = 3.14159 / 180.0
SENSOR_PERIOD_MS = 5
DT = SENSOR_PERIOD_MS / 1000.0

# Global variables for sensor data
current_height = 0.0
motion_delta_x = 0
motion_delta_y = 0
sensor_data_ready = False

# Simple smoothing filter
velocity_x_history = [0.0, 0.0, 0.0]
velocity_y_history = [0.0, 0.0, 0.0]

# Variables to track calculated velocities
current_vx = 0.0
current_vy = 0.0


def calculate_velocity(delta_value, altitude):
    """Convert delta value to linear velocity"""
    if altitude <= 0:
        return 0.0

    velocity_constant = (4.2 * DEG_TO_RAD) / (30.0 * DT)
    velocity = delta_value * altitude * velocity_constant
    return velocity


def smooth_velocity(new_velocity, history):
    """Apply 3-point smoothing filter"""
    history[2] = history[1]
    history[1] = history[0]
    history[0] = new_velocity

    smoothed = sum(history) / 3.0

    if abs(smoothed) < VELOCITY_THRESHOLD:
        smoothed = 0.0

    return smoothed


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


def get_position_corrections():
    """Calculate position corrections based on motion"""
    if not sensor_data_ready or current_height <= 0:
        return 0.0, 0.0

    # Calculate corrections (invert to counteract movement)
    correction_vx = -current_vx * SENSITIVITY
    correction_vy = -current_vy * SENSITIVITY

    # Apply limits
    correction_vx = max(-MAX_CORRECTION, min(MAX_CORRECTION, correction_vx))
    correction_vy = max(-MAX_CORRECTION, min(MAX_CORRECTION, correction_vy))

    return correction_vx, correction_vy


def setup_logging(cf):
    """Setup motion sensor logging"""
    print("Setting up motion sensor logging...")

    log_motion = LogConfig(name="Motion", period_in_ms=20)

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

        print("✓ Motion logging started")
        return log_motion

    except Exception as e:
        print(f"✗ Error setting up logging: {e}")
        return None


def main():
    """Main flight sequence using SyncCrazyflie"""

    # Initialize CRTP drivers
    cflib.crtp.init_drivers()

    print("=== Simple Position Hold Flight (SyncCrazyflie Version) ===")
    print(f"Target height: {TARGET_HEIGHT}m")
    print(f"Trim: vx={TRIM_VX}, vy={TRIM_VY}")

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

            # Phase 1: Quick Takeoff with VX/VY logging
            print(f"\nPhase 1: Taking off to {TARGET_HEIGHT}m...")
            start_time = time.time()
            last_print_time = 0

            while time.time() - start_time < TAKEOFF_TIME:
                cf.commander.send_hover_setpoint(TRIM_VX, TRIM_VY, 0, TARGET_HEIGHT)
                time.sleep(0.02)

                # Print VX/VY every 0.2 seconds during takeoff
                elapsed = time.time() - start_time
                if elapsed - last_print_time >= 0.2:
                    if log_config and sensor_data_ready:
                        print(f"Takeoff: {current_height:.3f}m | VX: {current_vx:.3f} | VY: {current_vy:.3f}")
                    else:
                        print(f"Takeoff: {current_height:.3f}m | VX: N/A | VY: N/A")
                    last_print_time = elapsed

            print(f"✓ Takeoff complete! Height: {current_height:.3f}m")

            # Phase 2: Position Hold with Motion Corrections and VX/VY logging
            print(f"\nPhase 2: Position hold for {HOVER_DURATION}s...")
            start_time = time.time()
            last_print_time = 0

            while time.time() - start_time < HOVER_DURATION:
                # Get motion-based corrections if logging is working
                if log_config and sensor_data_ready:
                    motion_vx, motion_vy = get_position_corrections()
                else:
                    motion_vx, motion_vy = 0.0, 0.0  # Fallback to no corrections

                # Combine trim + motion corrections
                total_vx = TRIM_VX + motion_vy
                total_vy = TRIM_VY + motion_vx

                # Send combined setpoint
                cf.commander.send_hover_setpoint(total_vx, total_vy, 0, TARGET_HEIGHT)
                time.sleep(0.05)

                # Print VX/VY every 0.2 seconds during hover
                elapsed = time.time() - start_time
                remaining = HOVER_DURATION - elapsed
                if elapsed - last_print_time >= 0.2:
                    if log_config and sensor_data_ready:
                        print(f"Hold: {remaining:.1f}s | Height: {current_height:.3f}m | "
                              f"VX: {current_vx:.3f} | VY: {current_vy:.3f} | "
                              f"Corrections: vx={motion_vx:.3f} vy={motion_vy:.3f} |"
                              f"Total: Tvx={total_vx:.3f} Tvy={total_vy:.3f}")
                    else:
                        print(f"Hold: {remaining:.1f}s | Height: {current_height:.3f}m | "
                              f"VX: N/A | VY: N/A | Mode: Basic trim only")
                    last_print_time = elapsed

            print("✓ Position hold complete")

            # Phase 3: Landing with VX/VY logging
            print("\nPhase 3: Landing...")
            start_time = time.time()
            last_print_time = 0

            while time.time() - start_time < LANDING_TIME:
                cf.commander.send_hover_setpoint(TRIM_VX, TRIM_VY, 0, 0)
                time.sleep(0.02)

                # Print VX/VY every 0.2 seconds during landing
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