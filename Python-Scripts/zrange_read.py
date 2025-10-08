import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
import socket

# URI for your LiteWing drone
DRONE_URI = "udp://192.168.43.42"

# Packet logging
packet_log = []


def log_packet(direction, data, address=None):
    """Log packets with timestamp, direction and detailed info"""
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
        print(f"    Port={data[0] >> 4}, Channel={data[0] & 0x0F}")


# Monkey patch socket to capture all packets
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


# Apply packet logging
socket.socket.send = logged_send
socket.socket.sendto = logged_sendto
socket.socket.recv = logged_recv
socket.socket.recvfrom = logged_recvfrom


def test_height_hold_sequence():
    """Test different height hold commands and log all packets"""
    print("=" * 60)
    print("HEIGHT HOLD PACKET LOGGER")
    print("=" * 60)

    # Initialize and connect
    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache='./cache')

    try:
        print("\n=== 1. CONNECTING TO DRONE ===")
        cf.open_link(DRONE_URI)
        time.sleep(1.0)

        print("\n=== 2. ARMING DRONE ===")
        cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(0.1)

        print("\n=== 3. ENABLING HIGH-LEVEL COMMANDER ===")
        cf.param.set_value('commander.enHighLevel', '1')
        time.sleep(0.5)

        print("\n=== 4. HEIGHT HOLD TESTS ===")

        # Test 1: Basic hover at 0.5m
        print("\n--- Test 1: Hover at 0.5m, no movement (vx=0, vy=0) ---")
        print("🎯 EXPECTING: Drone should take off and hover at 0.5m")
        cf.commander.send_hover_setpoint(0, 0, 0, 0.5)
        time.sleep(3.0)

        # Test 2: Different height - 1.0m
        print("\n--- Test 2: Hover at 1.0m, no movement (vx=0, vy=0) ---")
        print("🎯 EXPECTING: Drone should climb to 1.0m and hover")
        cf.commander.send_hover_setpoint(0, 0, 0, 1.0)
        time.sleep(3.0)

        # Test 3: Very small forward velocity (minimal hover test)
        print("\n--- Test 3: Minimal forward (vx=0.01) at 0.8m ---")
        print("🎯 EXPECTING: Drone should barely move forward, mostly hover")
        cf.commander.send_hover_setpoint(0.01, 0, 0, 0.8)
        time.sleep(3.0)

        # Test 4: Very small sideways velocity
        print("\n--- Test 4: Minimal sideways (vy=0.01) at 0.8m ---")
        print("🎯 EXPECTING: Drone should barely move sideways, mostly hover")
        cf.commander.send_hover_setpoint(0, 0.01, 0, 0.8)
        time.sleep(3.0)

        # Test 5: Back to pure hover after movement
        print("\n--- Test 5: Pure hover after movement (vx=0, vy=0) at 0.8m ---")
        print("🎯 EXPECTING: Drone should stop moving and hover in place")
        cf.commander.send_hover_setpoint(0, 0, 0, 0.8)
        time.sleep(3.0)

        # Test 6: Normal forward movement
        print("\n--- Test 6: Normal forward (vx=0.3) at 0.8m ---")
        print("🎯 EXPECTING: Drone should move forward clearly")
        cf.commander.send_hover_setpoint(0.3, 0, 0, 0.8)
        time.sleep(3.0)

        # Test 7: Stop forward movement, return to hover
        print("\n--- Test 7: Stop forward, hover (vx=0, vy=0) at 0.8m ---")
        print("🎯 EXPECTING: Drone should stop forward movement and hover")
        cf.commander.send_hover_setpoint(0, 0, 0, 0.8)
        time.sleep(3.0)

        # Test 8: Side movement with height change
        print("\n--- Test 8: Move right (vy=0.2) + climb to 1.2m ---")
        print("🎯 EXPECTING: Drone should move right and climb simultaneously")
        cf.commander.send_hover_setpoint(0, 0.2, 0, 1.2)
        time.sleep(3.0)

        # Test 9: Combined movement
        print("\n--- Test 9: Combined movement (vx=0.1, vy=0.1) at 0.7m ---")
        print("🎯 EXPECTING: Drone should move diagonally and descend")
        cf.commander.send_hover_setpoint(0.1, 0.1, 0, 0.7)
        time.sleep(3.0)

        # Test 10: Rapid height changes with zero velocity
        print("\n--- Test 10: Rapid height changes (vx=0, vy=0) ---")
        print("🎯 EXPECTING: Drone should change height quickly")
        cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
        time.sleep(1.5)
        cf.commander.send_hover_setpoint(0, 0, 0, 1.0)
        time.sleep(1.5)
        cf.commander.send_hover_setpoint(0, 0, 0, 0.6)
        time.sleep(1.5)

        # Test 11: Yaw rotation with height hold
        print("\n--- Test 11: Rotate (yaw_rate=30) at 0.5m ---")
        print("🎯 EXPECTING: Drone should rotate in place at 0.5m")
        cf.commander.send_hover_setpoint(0, 0, 30, 0.5)
        time.sleep(3.0)

        # Test 12: Stop rotation, hover
        print("\n--- Test 12: Stop rotation, hover (yaw=0) at 0.5m ---")
        print("🎯 EXPECTING: Drone should stop rotating and hover")
        cf.commander.send_hover_setpoint(0, 0, 0, 0.5)
        time.sleep(3.0)

        # Test 13: Edge case - zero height (should land)
        print("\n--- Test 13: Zero height (should trigger landing) ---")
        print("🎯 EXPECTING: Drone should land/motors should stop")
        cf.commander.send_hover_setpoint(0, 0, 0, 0.0)
        time.sleep(3.0)

        # Test 14: Try to take off again after landing
        print("\n--- Test 14: Take off again after landing (0.6m) ---")
        print("🎯 EXPECTING: Drone should take off again")
        cf.commander.send_hover_setpoint(0, 0, 0, 0.6)
        time.sleep(3.0)

        # Test 15: Continuous hover test (no movement)
        print("\n--- Test 15: Extended hover test (30 seconds at 0.8m) ---")
        print("🎯 EXPECTING: Drone should hover steadily for 30 seconds")
        start_time = time.time()
        while time.time() - start_time < 30:
            cf.commander.send_hover_setpoint(0, 0, 0, 0.8)
            time.sleep(0.1)  # Send at 10Hz

        print("\n=== 5. LANDING ===")
        cf.commander.send_hover_setpoint(0, 0, 0, 0.2)
        time.sleep(2.0)
        cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(0.5)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cf.close_link()
        print("\n=== 6. DISCONNECTED ===")


def analyze_packets():
    """Analyze captured packets for height hold patterns"""
    print("\n" + "=" * 60)
    print("PACKET ANALYSIS")
    print("=" * 60)

    print(f"Total packets captured: {len(packet_log)}")

    # Group packets by type
    commander_packets = []
    param_packets = []

    for packet in packet_log:
        if packet['length'] > 0:
            port = (packet['data'][0] >> 4) & 0x0F
            channel = packet['data'][0] & 0x0F

            if port == 3:  # Commander port
                commander_packets.append(packet)
            elif port == 2:  # Parameter port
                param_packets.append(packet)

    print(f"\nCommander packets (Port 3): {len(commander_packets)}")
    print(f"Parameter packets (Port 2): {len(param_packets)}")

    print("\n=== KEY COMMANDER PACKETS ===")
    for i, packet in enumerate(commander_packets[-10:]):  # Show last 10
        print(f"Packet {i}: {packet['hex']}")
        if packet['length'] >= 15:  # Hover setpoint packets are typically longer
            print(f"  -> Likely hover setpoint: vx, vy, yaw_rate, height data")

    print("\n=== PARAMETER SETTING PACKETS ===")
    for packet in param_packets[-5:]:  # Show last 5 parameter packets
        if packet['direction'] == 'SENT':
            print(f"Parameter set: {packet['hex']}")


def save_packet_analysis():
    """Save detailed packet analysis to file"""
    try:
        with open('height_hold_packets.txt', 'w') as f:
            f.write("HEIGHT HOLD PACKET ANALYSIS\n")
            f.write("=" * 50 + "\n\n")

            # Group packets by operation
            operations = [
                "CONNECTING TO DRONE",
                "ARMING DRONE",
                "ENABLING HIGH-LEVEL COMMANDER",
                "Test 1: Hover at 0.5m",
                "Test 2: Hover at 1.0m",
                "Test 3: Move forward",
                "Test 4: Move right",
                "Test 5: Combined movement",
                "Test 6: Rotate",
                "LANDING"
            ]

            for i, packet in enumerate(packet_log):
                f.write(f"Packet #{i + 1}\n")
                f.write(f"Timestamp: {packet['timestamp']:.6f}\n")
                f.write(f"Direction: {packet['direction']}\n")
                f.write(f"Length: {packet['length']} bytes\n")
                f.write(f"Hex: {packet['hex']}\n")

                if packet['length'] > 0:
                    header = packet['data'][0]
                    port = (header >> 4) & 0x0F
                    channel = header & 0x0F
                    f.write(f"CRTP - Port: {port}, Channel: {channel}\n")

                    # Identify packet type
                    if port == 3:
                        f.write("Type: COMMANDER PACKET\n")
                        if packet['length'] >= 15:
                            f.write("Subtype: Likely hover setpoint (vx, vy, yaw, height)\n")
                    elif port == 2:
                        f.write("Type: PARAMETER PACKET\n")

                f.write("-" * 30 + "\n\n")

        print(f"Detailed analysis saved to 'height_hold_packets.txt'")

    except Exception as e:
        print(f"Error saving analysis: {e}")


if __name__ == "__main__":
    # Run the height hold test sequence
    test_height_hold_sequence()

    # Analyze the captured packets
    analyze_packets()

    # Save detailed analysis
    save_packet_analysis()

    print("\n" + "=" * 60)
    print("SUMMARY FOR DART IMPLEMENTATION:")
    print("=" * 60)
    print("1. Check 'height_hold_packets.txt' for complete packet details")
    print("2. Look for Port 3 (Commander) packets for hover setpoints")
    print("3. Parameter packets show how to enable high-level mode")
    print("4. Each height/velocity change will show exact byte patterns")
    print("5. Use these exact packets in your Dart UDP implementation")