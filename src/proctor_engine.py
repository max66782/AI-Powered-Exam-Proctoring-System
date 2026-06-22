import cv2
import time
import mediapipe as mp
from ultralytics import YOLO

from datetime import datetime
import numpy as np
import os
import yaml
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis


class ProctorEngine:

    def __init__(self):

        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Phone detection state
        self.phone_start = None
        self.phone_logged = False
        self.phone_last_seen = None
        self.PHONE_GRACE_PERIOD = 1.0

        # Multiple person state
        self.multiple_face_start = None
        self.multiple_face_logged = False

        #Missing face
        self.missing_face_start = None
        self.missing_face_logged = False

        self.MISSING_FACE_THRESHOLD = config[
            "missing_face_threshold"
        ]

        # Head direction state
        self.away_start = None
        self.away_logged = False

        self.LOOKING_AWAY_THRESHOLD = config[
            "looking_away_threshold"
        ]

        # Identity verification state
        self.identity_logged = False

        self.unknown_start = None
        self.unknown_logged = False

        self.UNAUTHORIZED_THRESHOLD = config[
            "unauthorized_threshold"
        ]

        # Identity check throttling
        self.last_identity_check = 0
        self.cached_identity_status = "NO_FACE"
        self.cached_distance = None

        # Face verification threshold
        self.FACE_MATCH_THRESHOLD = config[
            "face_match_threshold"
        ]

        # YOLO
        self.model = YOLO("yolov8n.pt")

        # InsightFace
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)

        self.reference_embedding = np.load(
            "data/embeddings/krishn.npy"
        )

        # MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection

        self.face_detector = (
            self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
        )

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = (
            self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

        # Thresholds
        self.CONFIDENCE_THRESHOLD = config[
            "phone_confidence_threshold"
        ]

        self.PHONE_THRESHOLD = config[
            "phone_threshold"
        ]

        self.MULTIPLE_FACE_THRESHOLD = config[
            "multiple_face_threshold"
        ]

        # Evidence storage
        os.makedirs(
            "evidence",
            exist_ok=True
        )

        # Risk scoring
        self.risk_score = 0

        self.phone_events = 0
        self.multiple_person_events = 0
        self.missing_face_events = 0
        self.away_events = 0
        self.unauthorized_events = 0

        # FPS tracking
        self.prev_frame_time = time.time()
        self.fps = 0

    def log_event(self, event_type):

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

    def save_evidence(
        self,
        frame,
        event_type
    ):

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"evidence/{event_type}_{timestamp}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

    def add_risk(
        self,
        points
    ):

        self.risk_score += points

        if self.risk_score > 100:
            self.risk_score = 100

    def generate_report(self):

        os.makedirs(
            "reports",
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"reports/report_{timestamp}.txt"
        )

        with open(filename, "w") as file:

            file.write(
                "AI PROCTOR SESSION REPORT\n"
            )

            file.write(
                "========================\n\n"
            )

            file.write(
                f"Risk Score: {self.risk_score}\n\n"
            )

            file.write(
                f"Phone Events: {self.phone_events}\n"
            )

            file.write(
                f"Multiple Person Events: {self.multiple_person_events}\n"
            )

            file.write(
                f"Face Missing Events: {self.missing_face_events}\n"
            )

            file.write(
                f"Looking Away Events: {self.away_events}\n"
            )

            file.write(
                f"Unauthorized Events: {self.unauthorized_events}\n"
            )

        print(
            f"Report Saved: {filename}"
        )

    def update_fps(self):

        current_time = time.time()

        elapsed = (
            current_time
            - self.prev_frame_time
        )

        if elapsed > 0:
            self.fps = int(
                1 / elapsed
            )

        self.prev_frame_time = current_time

    def detect_phone(self, frame):

        results = self.model(
            frame,
            verbose=False
        )

        annotated_frame = results[0].plot()

        phone_detected = False
        phone_confidence = 0.0

        for box in results[0].boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = self.model.names[class_id]

            if class_name == "cell phone":

                phone_confidence = max(
                    phone_confidence,
                    confidence
                )

                if (
                    confidence >
                    self.CONFIDENCE_THRESHOLD
                ):
                    phone_detected = True

        cv2.putText(
            annotated_frame,
            f"Phone Conf: {phone_confidence:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        current_time = time.time()

        if phone_detected:

            self.phone_last_seen = current_time

            if self.phone_start is None:
                self.phone_start = current_time

            elapsed = (
                current_time
                - self.phone_start
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
                elapsed > self.PHONE_THRESHOLD
                and not self.phone_logged
            ):

                print(
                    "PHONE_DETECTED"
                )

                self.log_event(
                    "PHONE_DETECTED"
                )

                self.save_evidence(
                    annotated_frame,
                    "PHONE_DETECTED"
                )

                self.add_risk(50)

                self.phone_events += 1

                self.phone_logged = True

        else:

            if (
                self.phone_last_seen is not None
                and current_time - self.phone_last_seen < self.PHONE_GRACE_PERIOD
            ):
                pass
            else:
                self.phone_start = None
                self.phone_logged = False
                self.phone_last_seen = None

        return annotated_frame

    def detect_multiple_persons(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_detector.process(
            rgb_frame
        )

        face_count = 0

        if results.detections:
            face_count = len(
                results.detections
            )

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        if face_count > 1:

            if self.multiple_face_start is None:
                self.multiple_face_start = time.time()

            elapsed = (
                time.time()
                - self.multiple_face_start
            )

            cv2.putText(
                frame,
                f"Multi: {elapsed:.1f}s",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            if (
                elapsed >
                self.MULTIPLE_FACE_THRESHOLD
                and not
                self.multiple_face_logged
            ):

                print(
                    "MULTIPLE_PERSON_DETECTED"
                )

                self.log_event(
                    "MULTIPLE_PERSON_DETECTED"
                )

                self.save_evidence(
                    frame,
                    "MULTIPLE_PERSON_DETECTED"
                )

                self.add_risk(40)

                self.multiple_person_events += 1

                self.multiple_face_logged = True

        else:

            self.multiple_face_start = None
            self.multiple_face_logged = False

        return frame, face_count


    def detect_missing_face(
        self,
        frame,
        face_count
    ):

        if face_count == 0:

            if self.missing_face_start is None:
                self.missing_face_start = time.time()

            elapsed = (
                time.time()
                - self.missing_face_start
            )

            cv2.putText(
                frame,
                f"Missing: {elapsed:.1f}s",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2
            )

            if (
                elapsed >
                self.MISSING_FACE_THRESHOLD
                and not
                self.missing_face_logged
            ):

                print(
                    "FACE_MISSING_DETECTED"
                )

                self.log_event(
                    "FACE_MISSING_DETECTED"
                )

                self.save_evidence(
                    frame,
                    "FACE_MISSING_DETECTED"
                )

                self.add_risk(20)

                self.missing_face_events += 1

                self.missing_face_logged = True

        else:

            self.missing_face_start = None
            self.missing_face_logged = False

        return frame

    def detect_head_direction(self, frame):

        NOSE_TIP = 1
        LEFT_EYE = 33
        RIGHT_EYE = 263

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb_frame
        )

        direction = "NO_FACE"

        if results.multi_face_landmarks:

            h, w, _ = frame.shape

            landmarks = (
                results.multi_face_landmarks[0]
                .landmark
            )

            left_eye_x = int(
                landmarks[LEFT_EYE].x * w
            )

            right_eye_x = int(
                landmarks[RIGHT_EYE].x * w
            )

            nose_x = int(
                landmarks[NOSE_TIP].x * w
            )

            eye_mid = (
                left_eye_x + right_eye_x
            ) / 2

            offset = nose_x - eye_mid

            if offset > 15:
                direction = "RIGHT"
            elif offset < -15:
                direction = "LEFT"
            else:
                direction = "CENTER"

            if direction in ["LEFT", "RIGHT"]:

                if self.away_start is None:
                    self.away_start = time.time()

                elapsed = (
                    time.time()
                    - self.away_start
                )

                cv2.putText(
                    frame,
                    f"Away: {elapsed:.1f}s",
                    (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                if (
                    elapsed > self.LOOKING_AWAY_THRESHOLD
                    and not self.away_logged
                ):

                    print(
                        "LOOKING_AWAY_DETECTED"
                    )

                    self.log_event(
                        "LOOKING_AWAY_DETECTED"
                    )

                    self.save_evidence(
                        frame,
                        "LOOKING_AWAY_DETECTED"
                    )

                    self.add_risk(15)

                    self.away_events += 1

                    self.away_logged = True

            else:

                self.away_start = None
                self.away_logged = False

        if direction == "NO_FACE":

            self.away_start = None
            self.away_logged = False

        cv2.putText(
            frame,
            f"Direction: {direction}",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return frame, direction

    def detect_identity(self, frame):

        current_time = time.time()

        if (
            current_time
            - self.last_identity_check
            > 2
        ):

            self.last_identity_check = current_time

            faces = self.app.get(frame)

            self.cached_identity_status = "NO_FACE"
            self.cached_distance = None

            if len(faces) > 0:

                live_embedding = (
                    faces[0].embedding
                )

                distance = cosine(
                    self.reference_embedding,
                    live_embedding
                )

                self.cached_distance = distance

                if (
                    distance <
                    self.FACE_MATCH_THRESHOLD
                ):

                    self.cached_identity_status = (
                        "VERIFIED"
                    )

                    self.unknown_start = None
                    self.unknown_logged = False

                else:

                    self.cached_identity_status = (
                        "UNKNOWN"
                    )

                    if self.unknown_start is None:
                        self.unknown_start = time.time()

                    elapsed = (
                        time.time()
                        - self.unknown_start
                    )

                    if (
                        elapsed >
                        self.UNAUTHORIZED_THRESHOLD
                        and not
                        self.unknown_logged
                    ):

                        print(
                            "UNAUTHORIZED_PERSON_DETECTED"
                        )

                        self.log_event(
                            "UNAUTHORIZED_PERSON_DETECTED"
                        )

                        self.save_evidence(
                            frame,
                            "UNAUTHORIZED_PERSON_DETECTED"
                        )

                        self.add_risk(60)

                        self.unauthorized_events += 1

                        self.unknown_logged = True

        if self.cached_distance is not None:

            cv2.putText(
                frame,
                f"Distance: {self.cached_distance:.2f}",
                (20, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Identity: {self.cached_identity_status}",
            (20, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        return frame, self.cached_identity_status

engine = ProctorEngine()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    engine.update_fps()

    # Phone detection
    frame = engine.detect_phone(frame)

    # Multiple person detection
    frame, face_count = engine.detect_multiple_persons(frame)

    # Missing face detection
    frame = engine.detect_missing_face(
        frame,
        face_count
    )

    frame, direction = (
        engine.detect_head_direction(frame)
    )

    frame, identity_status = (
        engine.detect_identity(frame)
    )

    cv2.putText(
        frame,
        f"Risk Score: {engine.risk_score}",
        (20, 440),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {int(engine.fps)}",
        (20, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "AI Proctor Engine",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

engine.generate_report()

cap.release()

engine.face_detector.close()
engine.face_mesh.close()
# InsightFace cleanup not required

cv2.destroyAllWindows()