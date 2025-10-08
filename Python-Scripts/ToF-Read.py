import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# URI for your LiteWing drone
DRONE_URI = "udp://192.168.43.42:2390"  # Try a different port

# Initialize CRTP drivers
cflib.crtp.init_drivers()


class ToFHeightReader:
    def __init__(self):
        self.cf = Crazyflie(rw_cache='./cache')
        self.scf = None
        self.logging_active = False
        self.data_received_count = 0
        self.log_tof = None  # Initialize to None
        self.connected = False

        # Connect to drone
        try:
            print("Connecting to drone...")
            self.scf = SyncCrazyflie(DRONE_URI, cf=self.cf)
            self.scf.__enter__()
            print("✓ Connected successfully!")
            self.connected = True
            time.sleep(1.0)  # Give connection time to stabilize

            # Print available ToF variables
            self.print_tof_variables()

            # Setup logging only if connected successfully
            self.setup_logging()

        except Exception as e:
            print(f"✗ Error connecting to drone: {e}")
            print("\nTroubleshooting steps:")
            print("1. Make sure no other programs are connected to the drone")
            print("2. Close any running motion logger or drone applications")
            print("3. Wait a few seconds and try again")
            print("4. Check if drone is powered on and connected to WiFi")
            self.connected = False
            return

    def print_tof_variables(self):
        """Print available ToF/range related variables"""
        print("\n=== Available ToF/Range Variables ===")
        try:
            toc = self.cf.log.toc.toc
            tof_vars = []

            for group_name in sorted(toc.keys()):
                group = toc[group_name]
                for var_name in sorted(group.keys()):
                    full_name = f"{group_name}.{var_name}"
                    # Look for ToF, range, height, distance related variables
                    if any(keyword in full_name.lower() for keyword in
                           ['tof', 'range', 'distance', 'height', 'vl53', 'zrange']):
                        tof_vars.append(full_name)

            if tof_vars:
                print("Found ToF/Range variables:")
                for var in tof_vars:
                    print(f"  - {var}")
            else:
                print("No ToF/Range variables found")

            # Also check common variable names
            common_tof_vars = [
                'range.zrange', 'range.front', 'range.back', 'range.up', 'range.down',
                'tof.distance', 'stabilizer.z', 'stateEstimate.z'
            ]

            print(f"\nChecking common ToF variable names:")
            for var in common_tof_vars:
                if '.' in var:
                    group, name = var.split('.')
                    if group in toc and name in toc[group]:
                        print(f"  ✓ {var} - FOUND")
                    else:
                        print(f"  ✗ {var} - NOT FOUND")

        except Exception as e:
            print(f"Error reading variables: {e}")
        print("=== End ToF Variable List ===\n")

    def setup_logging(self):
        """Setup logging configuration for stateEstimate.z"""
        print("Setting up stateEstimate.z logging configuration...")

        # Create log configuration
        self.log_tof = LogConfig(name="StateEstimate", period_in_ms=50)  # 20Hz

        try:
            toc = self.cf.log.toc.toc

            # Only try to add stateEstimate.z
            var_name = 'stateEstimate.z'
            var_type = 'float'

            group, name = var_name.split('.')
            if group in toc and name in toc[group]:
                try:
                    self.log_tof.add_variable(var_name, var_type)
                    print(f"✓ Added variable: {var_name}")
                except Exception as e:
                    print(f"✗ Failed to add {var_name}: {e}")
                    return False
            else:
                print(f"✗ Variable {var_name} not found!")
                print("Available variables in stateEstimate group:")
                if 'stateEstimate' in toc:
                    for available_var in toc['stateEstimate']:
                        print(f"  - stateEstimate.{available_var}")
                return False

            print("Successfully added stateEstimate.z variable")

        except Exception as e:
            print(f"Error setting up variables: {e}")
            return False

        # Set up callbacks
        self.log_tof.data_received_cb.add_callback(self.tof_callback)
        self.log_tof.error_cb.add_callback(self.logging_error_callback)
        self.log_tof.started_cb.add_callback(self.logging_started_callback)

        print("ToF logging configuration setup complete!")
        return True

    def tof_callback(self, timestamp, data, logconf):
        """Callback function to handle stateEstimate.z data"""
        try:
            self.data_received_count += 1

            # Simple display of stateEstimate.z data
            height_z = data.get('stateEstimate.z', 0)
            print(f"Height: {height_z:.3f}m")

        except Exception as e:
            print(f"Error: {e}")

    def logging_error_callback(self, logconf, msg):
        """Callback when logging encounters an error"""
        print(f"✗ Logging error: {msg}")

    def logging_started_callback(self, logconf, started):
        """Callback when logging starts/stops"""
        if started:
            print("✓ ToF logging started successfully!")
        else:
            print("✗ ToF logging stopped")

    def start_logging(self):
        """Start stateEstimate.z logging"""
        if not self.connected:
            print("✗ Cannot start logging - not connected to drone")
            return False

        if self.log_tof is None:
            print("✗ Cannot start logging - logging configuration not set up")
            return False

        print("\nStarting stateEstimate.z logging...")

        try:
            # Add log configuration
            self.cf.log.add_config(self.log_tof)
            time.sleep(0.5)

            # Check if valid
            if not self.log_tof.valid:
                print("ERROR: Log configuration is not valid!")
                return False

            # Start logging
            self.log_tof.start()
            time.sleep(0.5)

            self.logging_active = True
            print("✓ stateEstimate.z logging active - Press Ctrl+C to stop")
            return True

        except Exception as e:
            print(f"ERROR starting logging: {e}")
            return False

    def stop_logging(self):
        """Stop stateEstimate.z logging"""
        if self.logging_active and self.log_tof is not None:
            self.log_tof.stop()
            self.logging_active = False
            print("\n✓ stateEstimate.z logging stopped")

    def disconnect(self):
        """Disconnect from drone"""
        self.stop_logging()
        if self.scf:
            try:
                self.scf.__exit__(None, None, None)
                print("✓ Disconnected from drone")
            except:
                pass


def main():
    print("=== Height Reader for LiteWing Drone ===")
    print("This program reads height values from stateEstimate.z")
    print("-" * 50)

    # Create height reader
    height_reader = ToFHeightReader()

    # Only try to start logging if connected successfully
    if height_reader.connected and height_reader.start_logging():
        try:
            # Keep running until interrupted
            while True:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            height_reader.disconnect()
    else:
        if not height_reader.connected:
            print("Cannot start logging - connection failed")
        else:
            print("Failed to start height logging")
        height_reader.disconnect()


if __name__ == "__main__":
    main()