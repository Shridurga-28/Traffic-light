from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO(r"runs\\detect\\train\\weights\\best.pt")

# === Video source ===
video_path = r"D:\Abaja\test.mp4"  # <-- change this
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Could not open video: {video_path}")
    exit()

# === Output video settings ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_annotated.mp4", fourcc, fps, (width, height))

print("✅ Saving annotated video as output_annotated.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video.")
        break

    # Run YOLO inference
    results = model(frame, imgsz=960, conf=0.25)

    # Annotated frame
    annotated = results[0].plot()

    # Write to output video
    out.write(annotated)

    # Optional: show live preview
    cv2.imshow("YOLOv8 Video Inference", annotated)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop early
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("🎉 Video saved successfully as output_annotated.mp4")
