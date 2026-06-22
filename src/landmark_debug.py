import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

NOSE_TIP = 1
LEFT_EYE = 33
RIGHT_EYE = 263

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

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

            cv2.putText(
                frame,
                f"Direction: {direction}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            for idx in [
                NOSE_TIP,
                LEFT_EYE,
                RIGHT_EYE
            ]:

                x = int(
                    landmarks[idx].x * w
                )

                y = int(
                    landmarks[idx].y * h
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    str(idx),
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

        cv2.imshow(
            "Landmark Debug",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()