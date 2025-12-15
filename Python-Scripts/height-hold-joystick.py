import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
import hid

FLIGHT_TIME = 10

# URI for your LiteWing drone
DRONE_URI = "udp://192.168.43.42"

# Open the gamepad device (DragonRise Generic USB Joystick)
device = hid.device()
device.open(0x0079, 0x0006)  # Vendor ID and Product ID from your gamepad
print("Opened:", device.get_product_string())

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

cf.param.set_value('commander.enHighLevel', '1')
print("High-level commander activated")

# Send commands to the drone
#cf.commander.send_hover_setpoint(0, 0, 0, 0.2)
#time.sleep(0.1)


# Main control loop with timer
start_time = time.time()
while True:
    # Read data from gamepad
    data = device.read(64)
    if data:
        # Convert from 0-255 to -1 to 1 range
        joystick_a1 = (data[0] - 128) / 128
        joystick_a2 = (data[1] - 128) / 128

        # Convert joystick values to drone commands
        vx = -joystick_a2 * 0.5  # Forward/Backward (scaled to ±0.5 m/s)
        vy = -joystick_a1 * 0.5  # Left/Right (scaled to ±0.5 m/s)

        #joystick_a3 = (data[3] )#- 128) / 128  # Addaxis 3 reading
        print(vx, vy)
        cf.commander.send_hover_setpoint(vx, vy, 0, 0.5)


    if time.time() - start_time > FLIGHT_TIME:
        break

# Stop the motors
print("Stopping motors...")
cf.commander.send_setpoint(0, 0, 0, 0)
time.sleep(0.1)
# Close the connection
cf.close_link()

print("Test complete")