import time
import cflib.crtp
from cflib.crazyflie import Crazyflie

# URI for your LiteWing drone
DRONE_URI = "udp://192.168.43.42"


def initialize_drone():
    """Initialize and connect to the drone"""
    print("Initializing drivers...")
    cflib.crtp.init_drivers()  # using crazy real time protocol

    cf = Crazyflie(rw_cache='./cache')  # Create a Crazyflie object instance
    try:
        print(f"Connecting to {DRONE_URI}...")  # Connecting to specific uniform resource identifier
        cf.open_link(DRONE_URI)
        print("Connected to drone!")
        return cf
    except Exception as e:
        print(f"Error initializing drone: {e}")
        return None


def terminate_drone(cf):
    """Safely disconnect from the drone"""
    try:
        print("Closing drone connection...")
        cf.close_link()
        print("Drone connection closed!")
    except Exception as e:
        print(f"Error terminating drone connection: {e}")


def main():
    # Initialize drone
    cf = initialize_drone()

    if cf is not None:
        # Stay connected for 10 seconds doing nothing
        print("Connected! Waiting for 4 seconds...")
        time.sleep(4)

        # Terminate connection
        terminate_drone(cf)
    else:
        print("Failed to connect to drone.")


if __name__ == "__main__":
    main()