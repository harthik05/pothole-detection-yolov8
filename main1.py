import cv2
from ultralytics import YOLO

def main():
    # ==============================
    # Load YOLOv8 model (fast & light)
    # ==============================
    model = YOLO("yolov8n.pt")  # auto-downloads if not present

    # ==============================
    # Choose input source
    # ==============================
    # Use 0 for webcam
    # video_source = 0

    # Use video file
    video_source = r"c:\Users\rahul\Downloads\videoplayback.mp4"

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("❌ Error: Cannot open video source")
        return

    print("✅ Video started. Press 'Q' to exit.")

    # ==============================
    # Frame-by-frame processing
    # ==============================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Video ended or frame not found")
            break

        # Resize for faster processing (important for Raspberry Pi later)
        frame = cv2.resize(frame, (640, 480))

        # ==============================
        # YOLO Inference
        # ==============================
        results = model(frame, stream=True)

        for result in results:
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Confidence score
                    confidence = float(box.conf[0])

                    # Class ID
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Label text
                    label = f"{class_name} {confidence:.2f}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        # ==============================
        # Display Output
        # ==============================
        cv2.imshow("Road Anomaly Detection (Practice)", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    # ==============================
    # Cleanup
    # ==============================
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Program stopped")

# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    main()
