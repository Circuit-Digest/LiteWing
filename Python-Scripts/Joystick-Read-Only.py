import hid
import time

# Install the library first:
# pip install hidapi

# Open the gamepad device (DragonRise Generic USB Joystick)
device = hid.device()
device.open(0x0079, 0x0006)  # Vendor ID and Product ID from your gamepad
print("Opened:", device.get_product_string())

while True:
    # Read data from gamepad
    data = device.read(64)
    if data:
        # Convert from 0-255 to -1 to 1 range
        joystick_a1 = (data[0] - 128) / 128
        joystick_a2 = (data[1] - 128) / 128
        #joystick_a3 = (data[3] )#- 128) / 128  # Addaxis 3 reading

        print(joystick_a1, joystick_a2)

    time.sleep(0.01)  # Small delay to prevent CPU overuse