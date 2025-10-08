import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
import serial

# URI for your LiteWing drone
DRONE_URI = "udp://192.168.43.42"

# Connect to serial port
#ser = serial.Serial('/dev/cu.usbserial-0001', 115200, timeout=1) #use for serial wire connection
ser = serial.Serial("/dev/tty.ESP32_BT", baudrate=9600, timeout=1) #use for bluetooth connection
print("Connected to BT")

# Give the serial connection time to establish
time.sleep(1)

# Initialize CRTP drivers
cflib.crtp.init_drivers()

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


# Clear any initial data from serial
ser.reset_input_buffer()

# Read data continuously
while True:
    try:
        # Read a line of data
        line = ser.readline().decode('utf-8').strip()

        # Check if there's data
        if line:
            # Split the data and extract values
            values = line.split(',')

            # Clean the values before converting to float
            # This removes any duplicate decimal points
            clean_x = values[0].replace('0.0', '0.', 1) if '0.0' in values[0] else values[0]
            clean_y = values[1].replace('0.0', '0.', 1) if '0.0' in values[1] else values[1]
            trigger = values[2]

            if trigger == "0":
                cf.commander.send_setpoint(0, 0, 0, 0)
                ser.reset_input_buffer()
                print("waiting for trigger")

            if trigger == "1":
                gyroX = float(clean_x)
                gyroY = float(clean_y)

                # Map gyroX and gyroY (-5 to +5) to vx and vy (-0.5 to +0.5)
                vx = round(gyroX / 10.0, 1)  # Rounded to 1 decimal place
                vy = -round(gyroY / 10.0, 1)  # Rounded to 1 decimal place

                # Print only gyroX and gyroY
                print(vx, vy, trigger)

                cf.commander.send_hover_setpoint(vx, vy, 0, 0.5)
    except:
        # Skip any problematic lines
        pass

