from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("cotton-v1.pt")  # Your custom model

# Use Mac's built-in webcam (usually index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    # Run YOLO inference on the current frame
    results = model.predict(source=frame, show=True, save=False)

    # Optional: save or process result
    for result in results:
        boxes = result.boxes
        # result.save(filename="result.jpg")  # Save if needed

    # Press 'q' to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()