import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# URI for your LiteWing drone
DRONE_URI = "udp://192.168.43.42"

# Initialize CRTP drivers
cflib.crtp.init_drivers()

# Shared data dictionary
motion_data = {}


def motion_callback(timestamp, data, logconf):
    """Callback function to handle motion data"""
    motion_data.update(data)

    # Print motion data to console
    delta_x = motion_data.get('motion.deltaX', 'N/A')
    delta_y = motion_data.get('motion.deltaY', 'N/A')
    motion_detect = motion_data.get('motion.motion', 'N/A')

    print(f"Motion - DeltaX: {delta_x}, DeltaY: {delta_y}, Motion: {motion_detect}")


def main():
    print("Connecting to Crazyflie...")

    try:
        # Create Crazyflie instance
        cf = Crazyflie(rw_cache='./cache')

        with SyncCrazyflie(DRONE_URI, cf=cf) as scf:
            print("Connected successfully!")

            # Create log configuration for motion data with CORRECT data types
            log_motion = LogConfig(name="Motion", period_in_ms=100)
            log_motion.add_variable('motion.deltaX', 'int16_t')  # Changed from 'float' to 'int16_t'
            log_motion.add_variable('motion.deltaY', 'int16_t')  # Changed from 'float' to 'int16_t'
            log_motion.add_variable('motion.motion', 'uint8_t')  # This was already correct

            # Set up callback
            log_motion.data_received_cb.add_callback(motion_callback)

            # Add and start logging
            cf.log.add_config(log_motion)
            log_motion.start()

            print("Motion logging started. Press Ctrl+C to stop...")

            # Keep the program running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping motion logging...")
                log_motion.stop()

    except Exception as e:
        print(f"Error: {e}")

    print("Motion logging stopped.")


if __name__ == "__main__":
    main()