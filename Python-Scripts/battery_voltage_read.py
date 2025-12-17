import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
import socket
import threading
import struct

# URI for your LiteWing drone
DRONE_URI = "udp://192.168.43.42"

# Variable to store voltage data
voltage_data = {}

# Packet logging
packet_log = []


def log_packet(direction, data, address=None):
    """Log packets with timestamp and direction"""
    timestamp = time.time()
    packet_info = {
        'timestamp': timestamp,
        'direction': direction,  # 'SENT' or 'RECEIVED'
        'data': data,
        'hex': ' '.join(f'{b:02x}' for b in data),
        'length': len(data),
        'address': address
    }
    packet_log.append(packet_info)

    print(f"[{direction}] {len(data)} bytes: {packet_info['hex']}")
    if len(data) > 0:
        print(f"    First byte: 0x{data[0]:02x}, Header info: Port={data[0] >> 4}, Channel={data[0] & 0x0F}")


# Monkey patch the socket to intercept packets
original_socket_send = socket.socket.send
original_socket_sendto = socket.socket.sendto
original_socket_recv = socket.socket.recv
original_socket_recvfrom = socket.socket.recvfrom


def logged_send(self, data):
    log_packet('SENT', data)
    return original_socket_send(self, data)


def logged_sendto(self, data, address):
    log_packet('SENT', data, address)
    return original_socket_sendto(self, data, address)


def logged_recv(self, bufsize):
    data = original_socket_recv(self, bufsize)
    log_packet('RECEIVED', data)
    return data


def logged_recvfrom(self, bufsize):
    data, address = original_socket_recvfrom(self, bufsize)
    log_packet('RECEIVED', data, address)
    return data, address


# Apply the monkey patches
socket.socket.send = logged_send
socket.socket.sendto = logged_sendto
socket.socket.recv = logged_recv
socket.socket.recvfrom = logged_recvfrom


def voltage_callback(timestamp, data, logconf):
    """Callback function to handle voltage data"""
    voltage_data.update(data)
    print(f"CALLBACK - Voltage data received: {data}")


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


def read_voltage_once(cf):
    """Read voltage once and print it"""
    try:
        print("\n=== STARTING VOLTAGE LOGGING SETUP ===")

        # Create log configuration for battery voltage
        log_voltage = LogConfig(name="Voltage", period_in_ms=100)
        log_voltage.add_variable('pm.vbat', 'float')

        # Set up callback
        log_voltage.data_received_cb.add_callback(voltage_callback)

        print("Adding log config...")
        # Add and start logging
        cf.log.add_config(log_voltage)

        print("Starting log...")
        log_voltage.start()

        print("Reading voltage...")

        # Wait a bit for data to come in
        time.sleep(2)

        # Print the voltage
        voltage = voltage_data.get('pm.vbat', 'N/A')
        if isinstance(voltage, float):
            print(f"Battery Voltage: {voltage:.2f} V")
        else:
            print("Battery Voltage: N/A V")

        print("Stopping log...")
        # Stop logging
        log_voltage.stop()

        print("=== VOLTAGE LOGGING COMPLETE ===\n")

    except Exception as e:
        print(f"Error reading voltage: {e}")


def terminate_drone(cf):
    """Safely disconnect from the drone"""
    try:
        print("Closing drone connection...")
        cf.close_link()
        print("Drone connection closed!")
    except Exception as e:
        print(f"Error terminating drone connection: {e}")


def save_packet_log():
    """Save packet log to file for analysis"""
    try:
        with open('packet_log.txt', 'w') as f:
            f.write("CRTP Packet Log\n")
            f.write("================\n\n")

            for i, packet in enumerate(packet_log):
                f.write(f"Packet #{i + 1}\n")
                f.write(f"Timestamp: {packet['timestamp']:.6f}\n")
                f.write(f"Direction: {packet['direction']}\n")
                f.write(f"Length: {packet['length']} bytes\n")
                f.write(f"Hex: {packet['hex']}\n")
                if packet['address']:
                    f.write(f"Address: {packet['address']}\n")

                # Try to decode CRTP header
                if packet['length'] > 0:
                    header = packet['data'][0]
                    port = (header >> 4) & 0x0F
                    channel = header & 0x0F
                    f.write(f"CRTP - Port: {port}, Channel: {channel}\n")

                f.write("-" * 50 + "\n\n")

        print(f"Packet log saved to 'packet_log.txt' ({len(packet_log)} packets)")

    except Exception as e:
        print(f"Error saving packet log: {e}")


def print_packet_summary():
    """Print a summary of captured packets"""
    print(f"\n=== PACKET SUMMARY ===")
    print(f"Total packets captured: {len(packet_log)}")

    sent_packets = [p for p in packet_log if p['direction'] == 'SENT']
    received_packets = [p for p in packet_log if p['direction'] == 'RECEIVED']

    print(f"Sent: {len(sent_packets)}")
    print(f"Received: {len(received_packets)}")

    # Group by CRTP port
    ports = {}
    for packet in packet_log:
        if packet['length'] > 0:
            port = (packet['data'][0] >> 4) & 0x0F
            if port not in ports:
                ports[port] = {'sent': 0, 'received': 0}
            ports[port][packet['direction'].lower()] += 1

    print("\nBy CRTP Port:")
    for port, counts in ports.items():
        print(f"  Port {port}: Sent={counts['sent']}, Received={counts['received']}")


def main():
    print("=" * 60)
    print("DRONE PACKET LOGGER")
    print("=" * 60)

    # Initialize drone
    cf = initialize_drone()

    if cf is not None:
        # Wait 4 seconds doing nothing
        print("Connected! Waiting for 4 seconds...")
        time.sleep(4)

        # Read voltage once
        read_voltage_once(cf)

        # Wait another 4 seconds
        print("Waiting for another 4 seconds...")
        time.sleep(4)

        # Terminate connection
        terminate_drone(cf)

        # Print summary and save log
        print_packet_summary()
        save_packet_log()

    else:
        print("Failed to connect to drone.")


if __name__ == "__main__":
    main()