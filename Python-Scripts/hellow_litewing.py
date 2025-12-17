import time
import cflib.crtp
from cflib.crazyflie import Crazyflie

# URI for your LiteWing drone
DRONE_URI = "udp://192.168.43.42"

# Initialize CRTP drivers
cflib.crtp.init_drivers()

# Basic flight test
print("Hello World to LiteWing")

# Create Crazyflie instance
cf = Crazyflie()

# Connect to the drone
print("Connecting to drone...")
cf.open_link(DRONE_URI)

print("Connected to drone. Waiting for stability...")
time.sleep(1.0)  # Wait after connection

# First send zero setpoint to unlock safety
print("Sending zero setpoint to unlock safety...")
cf.commander.send_setpoint(0, 0, 0, 0)
time.sleep(0.1)

# Flight parameters
roll = 0.0
pitch = 0.0
yaw = 0
thrust = 10000  # Thrust value is 10000 minimum and 60000 maximum

# Start motors
print("Starting motors at minimum speed...")
cf.commander.send_setpoint(roll, pitch, yaw, thrust)
time.sleep(1)

# Stop the motors
print("Stopping motors...")
cf.commander.send_setpoint(0, 0, 0, 0)
time.sleep(0.1)
# Close the connection
cf.close_link()

print("Test complete")