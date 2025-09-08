from ultralytics import YOLO
import cv2
import os
from tkinter import Tk, filedialog

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")

# Open file dialog for image upload
Tk().withdraw()  # Hide the small Tkinter window
image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("❌ No file selected. Exiting...")
    exit()

print(f"📂 Selected file: {image_path}")

# Run YOLO model
results = model(image_path)

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Draw rectangle and label on image
        cv2.rectangle(r.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(r.orig_img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show output
cv2.imshow("Detections", r.orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite("output.jpg", r.orig_img)
print("✅ Detection complete. Output saved as output.jpg")
