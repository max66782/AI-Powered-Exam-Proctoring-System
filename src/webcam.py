import time
import cv2

# Open default webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

prev_time = 0
while True:
    # Read frame
    ret, frame = cap.read()

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    if not ret:
        print("Failed to grab frame")
        break

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # Display frame
    cv2.imshow("AI Proctor - Webcam Test", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()