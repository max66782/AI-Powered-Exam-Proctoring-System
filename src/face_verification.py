import cv2
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

# Load registered embedding
registered_embedding = np.load(
    "data/embeddings/krishn.npy"
)

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# Open webcam
cap = cv2.VideoCapture(0)

# Verification threshold
THRESHOLD = 0.50

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect faces
    faces = app.get(frame)

    for face in faces:

        # Live embedding
        live_embedding = face.embedding

        # Calculate distance
        distance = cosine(
            registered_embedding,
            live_embedding
        )

        if distance < THRESHOLD:
            label = "Verified"
        else:
            label = "Unknown"

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{label} {distance:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Face Verification",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()