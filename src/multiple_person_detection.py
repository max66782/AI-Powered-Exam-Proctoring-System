from datetime import datetime
import cv2
import time
import mediapipe as mp

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection

def log_event(event_type, face_count):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        "logs/events.log",
        "a"
    ) as file:

        file.write(
            f"{timestamp} | "
            f"{event_type} | "
            f"FACE_COUNT={face_count}\n"
        )

# Webcam
cap = cv2.VideoCapture(0)

# Timers
multiple_face_start = None

# Event flag
event_logged = False

missing_face_start = None
missing_event_logged = False

with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detection:

    while True:

        ret, frame = cap.read()

        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_detection.process(
            rgb_frame
        )

        face_count = 0

        if results.detections:
            face_count = len(
                results.detections
            )
        

        # Multiple face logic

        if face_count > 1:

            if multiple_face_start is None:
                multiple_face_start = time.time()

            elapsed_time = (
            time.time()
            - multiple_face_start
            )

            if elapsed_time > 3 and not event_logged:

                print(
                    "MULTIPLE_PERSON_DETECTED"
                )

                log_event(
                    "MULTIPLE_PERSON_DETECTED",
                    face_count
                )

                event_logged = True

            cv2.putText(
            frame,
            f"Timer: {elapsed_time:.1f}s",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
            )

        else:

            multiple_face_start = None
            event_logged = False

        # Missing face logic

        if face_count == 0:

            if missing_face_start is None:
                missing_face_start = time.time()

            missing_elapsed = (
            time.time()
            - missing_face_start
            )

            cv2.putText(
                frame,
                f"Missing: {missing_elapsed:.1f}s",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            if (
                missing_elapsed > 10
                and not missing_event_logged
            ):

                print(
                "FACE_MISSING_DETECTED"
                )

                log_event(
                "FACE_MISSING_DETECTED",
                face_count
                )

                missing_event_logged = True

        else:

            missing_face_start = None
            missing_event_logged = False


        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Multiple Person Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()