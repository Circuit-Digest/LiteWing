import serial
import time
import os

# Replace with your Bluetooth device's serial port
BLUETOOTH_PORT = "/dev/tty.ESP32_BT"  # macOS (change as needed)

try:
    # Open Bluetooth serial connection
    bt_serial = serial.Serial(BLUETOOTH_PORT, baudrate=9600, timeout=1)
    print("Connected to Bluetooth device!")

    start_time = time.time()  # Start time for 5-second limit

    while time.time() - start_time < 5:  # Run for 5 seconds
        data = bt_serial.readline().decode('utf-8').strip()
        if data:
            print(f"Received: {data}")

except serial.SerialException as e:
    print(f"Serial error: {e}")

finally:
    if 'bt_serial' in locals():
        print("Flushing input buffer...")
        bt_serial.reset_input_buffer()  # Flush input buffer
        print("Flushing output buffer...")
        bt_serial.reset_output_buffer()  # Flush output buffer
        print("Closing Bluetooth connection...")
        bt_serial.close()
        time.sleep(2)  # Give time for OS to fully release the port
        print("Bluetooth connection closed. Exiting.")

    # 💡 Extra Step: Reset Bluetooth on macOS (if needed)
    os.system("sudo pkill -f 'bluetoothd'")  # Force restart Bluetooth service (requires sudo)
