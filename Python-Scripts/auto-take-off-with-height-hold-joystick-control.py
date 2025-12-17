import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
import subprocess
import signal
import sys
import cv2
import numpy as np
from collections import deque

# WiFi Configuration
DRONE_SSID = "LightWing_F09E9E29511D"
DRONE_PASSWORD = "12345678"
HOME_SSID = "Semicon Media"
HOME_PASSWORD = "cracksen1605"
DRONE_URI = "udp://192.168.43.42"

# Flight parameters
TAKEOFF_HEIGHT = 0.4  # meters
FLIGHT_TIME = 5.0  # seconds


def connect_to_litewing():
    """Connect to LiteWing drone's WiFi network"""
    signal.signal(signal.SIGINT, lambda signal, frame: return_to_home_wifi())
    signal.signal(signal.SIGTERM, lambda signal, frame: return_to_home_wifi())
    print(f"Connecting to LiteWing ({DRONE_SSID})...")
    try:
        result = subprocess.run(
            ['networksetup', '-setairportnetwork', 'en0', DRONE_SSID, DRONE_PASSWORD],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"LiteWing WiFi connection error: {result.stderr}")
            return False
        time.sleep(3)
        return True
    except Exception as e:
        print(f"Failed to connect to LiteWing: {e}")
        return False


def return_to_home_wifi():
    """Reconnect to home WiFi network"""
    print("\nReconnecting to home WiFi...")
    subprocess.run(['networksetup', '-setairportnetwork', 'en0', HOME_SSID, HOME_PASSWORD])
    time.sleep(2)
    sys.exit(0)


def initialize_camera():
    """Initialize camera and return capture object"""
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("AVFoundation failed, trying default...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Default failed, trying index 1...")
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                raise Exception("Could not open any video stream")

    print("Camera opened successfully!")
    return cap


def process_frame(frame, frame_height, frame_width, center_x, center_y, smooth_x, smooth_y):
    """Process a single frame and return velocity values"""
    vx, vy = 0, 0

    # Convert frame to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    return gray, frame, vx, vy


def run_flight_sequence(scf):
    """Execute takeoff, camera-controlled flight, and landing sequence"""
    cf = scf.cf

    # Initialize camera
    cap = initialize_camera()
    ret, frame = cap.read()
    if not ret:
        raise Exception("Couldn't read initial frame")

    # Setup frame parameters
    frame_height, frame_width = frame.shape[:2]
    center_x = frame_width // 2
    center_y = frame_height // 2
    smooth_x = deque(maxlen=10)
    smooth_y = deque(maxlen=10)
    prev_frame = None

    print("\nCamera feed initialized. Press 'SPACE' to begin takeoff sequence or 'q' to quit.")

    # Wait for space key
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 0)
        gray, frame, vx, vy = process_frame(frame, frame_height, frame_width,
                                            center_x, center_y, smooth_x, smooth_y)

        # Draw crosshair
        cv2.line(frame, (center_x, 0), (center_x, frame_height), (255, 255, 255), 1)
        cv2.line(frame, (0, center_y), (frame_width, center_y), (255, 255, 255), 1)

        cv2.imshow('Motion Tracking', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord(' '):
            break

    try:
        print("\nArming drone...")
        cf.platform.send_arming_request(True)
        time.sleep(1.0)

        # Takeoff sequence
        print(f"Taking off to {TAKEOFF_HEIGHT}m...")
        steps = 5
        for i in range(steps):
            height = (i + 1) * TAKEOFF_HEIGHT / steps
            cf.commander.send_hover_setpoint(0, 0, 0, height)
            time.sleep(0.1)

        print(f"Starting camera control... Will land after {FLIGHT_TIME} seconds")
        start_time = time.time()

        while time.time() - start_time < FLIGHT_TIME:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 0)

            if prev_frame is None:
                prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
                continue

            # Process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # Motion detection
            frame_diff = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            vx, vy = 0, 0

            if contours:
                valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

                if valid_contours:
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center_obj_x = x + w // 2
                    center_obj_y = y + h // 2

                    smooth_x.append(center_obj_x)
                    smooth_y.append(center_obj_y)

                    if len(smooth_x) > 0:
                        smoothed_x = int(sum(smooth_x) / len(smooth_x))
                        smoothed_y = int(sum(smooth_y) / len(smooth_y))

                        # Calculate velocities
                        vx = -((center_y - smoothed_y) / (frame_height / 2)) * 1.2
                        vy = ((smoothed_x - center_x) / (frame_width / 2)) * 1.2

                        # Clamp values
                        vx = max(min(vx, 0.8), -0.8)
                        vy = max(min(vy, 0.8), -0.8)

                        # Draw visual indicators
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(frame, (smoothed_x, smoothed_y), 5, (0, 0, 255), -1)

            # Update display
            cv2.line(frame, (center_x, 0), (center_x, frame_height), (255, 255, 255), 1)
            cv2.line(frame, (0, center_y), (frame_width, center_y), (255, 255, 255), 1)

            cv2.putText(frame, f'vx: {vx:.2f} m/s', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'vy: {vy:.2f} m/s', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Motion Tracking', frame)

            # Send commands to the drone
            cf.commander.send_hover_setpoint(
                vx,  # Forward/Backward velocity
                vy,  # Left/Right velocity
                0,  # Yaw rate (no rotation)
                TAKEOFF_HEIGHT  # Maintain height
            )

            prev_frame = gray

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)

        # Landing sequence
        print("\nInitiating landing sequence...")
        for i in range(steps):
            height = TAKEOFF_HEIGHT * (steps - i - 1) / steps
            cf.commander.send_hover_setpoint(0, 0, 0, height)
            time.sleep(0.1)

        # Final landing and disarming
        cf.commander.send_stop_setpoint()
        time.sleep(0.1)
        print("Disarming drone...")
        cf.platform.send_arming_request(False)

    except Exception as e:
        print(f"Flight error: {e}")
        # Emergency stop
        cf.commander.send_stop_setpoint()
        cf.platform.send_arming_request(False)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    print("LiteWing Camera-Based Flight Control")
    print("-" * 30)

    if not connect_to_litewing():
        return_to_home_wifi()
        return

    try:
        # Initialize CRTP drivers
        cflib.crtp.init_drivers()

        # Create and connect to the drone
        with SyncCrazyflie(DRONE_URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            # Attempt to activate high-level commander
            try:
                scf.cf.param.set_value('commander.enHighLevel', '1')
                print("High-level commander activated")
            except Exception as e:
                print(f"Could not activate high-level commander: {e}")

            # Execute flight sequence
            run_flight_sequence(scf)

    except Exception as e:
        print(f"Operation error: {e}")
    finally:
        return_to_home_wifi()


if __name__ == '__main__':
    main()