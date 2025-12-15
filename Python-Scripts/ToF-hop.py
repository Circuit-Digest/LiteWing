import time
import cflib.crtp
from cflib.crazyflie import Crazyflie

# URI for your LiteWing drone
DRONE_URI = "udp://192.168.43.42"

# Initialize CRTP drivers
cflib.crtp.init_drivers()

# Create Crazyflie instance
cf = Crazyflie()

try:
    # Connect to the drone
    print("Connecting to drone...")
    cf.open_link(DRONE_URI)
    print("Connected to drone. Waiting for stability...")
    time.sleep(1.0)

    # Send zero setpoint to unlock safety
    print("Unlocking safety...")
    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)

    # Enable high-level commander
    cf.param.set_value('commander.enHighLevel', '1')
    print("High-level commander activated")
    time.sleep(0.5)

    # Takeoff to 0.1m height
    print("Taking off...")
    cf.commander.send_hover_setpoint(0.4, -0.05, 0, 0.3) #def send_hover_setpoint(self, vx, vy, yawrate, zdistance)

    # Hold position for 50ms
    time.sleep(5)  # 50ms

    # Land
    print("Landing...")
    cf.commander.send_hover_setpoint(0, 0, 0, 0)
    #time.sleep(1.0)  # Give time to land

    print("Flight complete!")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Always close the connection
    try:
        cf.close_link()
        print("Connection closed")
    except:
        pass