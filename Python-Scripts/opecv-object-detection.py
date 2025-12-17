import cv2
import numpy as np
from collections import deque


def track_orange():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    # Initialize deques for smoothing with a specified window size
    smooth_x = deque(maxlen=10)  # Adjust window size (10) for more/less smoothing
    smooth_y = deque(maxlen=10)

    # Initialize last known good position
    last_x, last_y = None, None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define orange color range in HSV
            lower_orange = np.array([5, 100, 100])
            upper_orange = np.array([15, 255, 255])

            mask = cv2.inRange(hsv, lower_orange, upper_orange)

            # Noise reduction
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the largest contour (assuming it's our target object)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 500:  # Minimum area threshold
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # Calculate center point
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Add to smoothing deques
                    smooth_x.append(center_x)
                    smooth_y.append(center_y)

                    # Calculate smoothed position
                    if len(smooth_x) > 0:
                        smoothed_x = int(sum(smooth_x) / len(smooth_x))
                        smoothed_y = int(sum(smooth_y) / len(smooth_y))

                        # Update last known good position
                        last_x, last_y = smoothed_x, smoothed_y

                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Draw smoothed center point
                        cv2.circle(frame, (smoothed_x, smoothed_y), 5, (0, 0, 255), -1)  # Red dot

                        # Optional: Draw trajectory line
                        if len(smooth_x) > 1:
                            prev_x = int(sum(list(smooth_x)[:-1]) / (len(smooth_x) - 1))
                            prev_y = int(sum(list(smooth_y)[:-1]) / (len(smooth_y) - 1))
                            cv2.line(frame, (prev_x, prev_y), (smoothed_x, smoothed_y),
                                     (255, 0, 0), 2)  # Blue line

                        # Display coordinates
                        cv2.putText(frame, f'({smoothed_x}, {smoothed_y})',
                                    (smoothed_x + 10, smoothed_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # If no object detected but we have a last known position
            elif last_x is not None and last_y is not None:
                cv2.putText(frame, "Object Lost!", (last_x, last_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Mask', mask)
            cv2.imshow('Video Stream', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    track_orange()