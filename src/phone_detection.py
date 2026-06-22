import cv2
import time
from datetime import datetime
from ultralytics import YOLO


def log_event(event_type):
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        "logs/events.log",
        "a"
    ) as file:

        file.write(
            f"{timestamp} | "
            f"{event_type}\n"
        )


# Load YOLO model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0)

# Variables
phone_start = None
phone_logged = False

# Thresholds
CONFIDENCE_THRESHOLD = 0.30
PHONE_THRESHOLD = 3

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    phone_detected = False

    # Check all detected objects
    for box in results[0].boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        # Debug print
        print(
            f"{class_name}: "
            f"{confidence:.2f}"
        )

        if (
            class_name == "cell phone"
            and confidence > CONFIDENCE_THRESHOLD
        ):
            phone_detected = True

    # Phone timer logic
    if phone_detected:

        if phone_start is None:
            phone_start = time.time()

        elapsed = (
            time.time()
            - phone_start
        )

        cv2.putText(
            annotated_frame,
            f"Phone: {elapsed:.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if (
            elapsed > PHONE_THRESHOLD
            and not phone_logged
        ):

            print(
                "PHONE_DETECTED"
            )

            log_event(
                "PHONE_DETECTED"
            )

            phone_logged = True

    else:

        phone_start = None
        phone_logged = False

    # Display frame
    cv2.imshow(
        "Phone Detection",
        annotated_frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()