import cv2
import os

# Create folder if it doesn't exist
os.makedirs("data/students", exist_ok=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    cv2.putText(
        frame,
        "Press C to Capture | Q to Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Student Registration", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        student_name = input("Enter student name: ")

        image_path = f"data/students/{student_name}.jpg"

        cv2.imwrite(image_path, frame)

        print(f"Image saved: {image_path}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()