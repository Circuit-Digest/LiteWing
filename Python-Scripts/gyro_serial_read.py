import serial
import time

# Connect to serial port
ser = serial.Serial('/dev/cu.usbserial-0001', 115200, timeout=1)
print(f"Connected to /dev/cu.usbserial-0001 at 115200 baud")

# Give the serial connection time to establish
time.sleep(2)

# Clear any initial data
ser.reset_input_buffer()

# Read data continuously
while True:
    try:
        # Read a line of data
        line = ser.readline().decode('utf-8').strip()

        # Check if there's data
        if line:
            # Split the data and extract only the first two values
            values = line.split(',')

            # Clean the values before converting to float
            # This removes any duplicate decimal points
            clean_x = values[0].replace('0.0', '0.', 1) if '0.0' in values[0] else values[0]
            clean_y = values[1].replace('0.0', '0.', 1) if '0.0' in values[1] else values[1]

            gyroX = float(clean_x)
            gyroY = float(clean_y)

            # Map gyroX and gyroY (-5 to +5) to vx and vy (-0.5 to +0.5)
            vx = round(gyroX / 10.0, 1)  # Rounded to 1 decimal place
            vy = -round(gyroY / 10.0, 1)  # Rounded to 1 decimal place

            # Print only gyroX and gyroY
            print(vx, vy)
    except:
        # Skip any problematic lines
        pass