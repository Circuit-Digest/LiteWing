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
roll = 0
pitch = 0
yaw = 0

starting_thrust = 30000
maximum_thrust = 40001 #maximum value possible is 60000


# Gradually increase thrust to take off change pitch and roll values for direction
print("Starting motors...")
for thrust in range(starting_thrust, maximum_thrust, 1000): #increase thrust from 10000 to 30000 in increments of 100
    cf.commander.send_setpoint(roll, pitch, yaw, thrust)
    print(f"Setting thrust to: {thrust}")
    time.sleep(0.2)


# Stop the motors
print("Stopping motors...")
cf.commander.send_setpoint(0, 0, 0, 0)
time.sleep(0.1)
# Close the connection
cf.close_link()

print("Test complete")