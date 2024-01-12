import cv2
import numpy as np
from PIL import Image, ImageDraw
# Load YOLOv3 model
yolo_net = cv2.dnn.readNet("path/to/yolov3.weights", "path/to/yolov3.cfg")

# Load class labels
yolo_classes = []
with open('path/to/coco.names', 'r') as f:
    yolo_classes = f.read().splitlines()

# Function to perform YOLO object detection on a single image
def perform_yolo_detection(img):
    height, width, _ = img.shape  # Get image dimensions

    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo_net.setInput(blob)

    # Get YOLO output
    output_layers_names = yolo_net.getUnconnectedOutLayersNames()
    layer_outputs = yolo_net.forward(output_layers_names)

    # Parse YOLO output
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indexes = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

           
    # Draw bounding boxes and labels on the image
    font = cv2.FONT_HERSHEY_SIMPLEX  # Change font type
    thickness = 3  # Thickness of the rectangle border
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(yolo_classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = tuple(int(c) for c in colors[i])

        # Draw outer rectangle using OpenCV (unchanged)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)

        # Calculate text size for dynamic positioning (unchanged)
        (label_width, label_height), baseline = cv2.getTextSize(label + " " + confidence, font, 0.5, 2)

        # Draw inner rectangle using PIL for rounded corners
        pil_img = Image.fromarray(img)  # Convert OpenCV image to PIL image
        draw = ImageDraw.Draw(pil_img)
        draw.rounded_rectangle((x + 3, y - 20, x + label_width + 3, y + 2), radius=5, fill=color, outline=color, width=3)
        img = np.array(pil_img)  # Convert back to OpenCV image

        # Draw text using OpenCV (unchanged)
        cv2.putText(img, label + " " + confidence, (x, y - 5), font, 0.5, (255, 255, 255), 1)
    return img

# Open a live camera feed
video_capture = cv2.VideoCapture(0)  # 0 indicates the default camera, you can change it if you have multiple cameras

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Perform YOLO object detection on the frame
    yolo_detected_frame = perform_yolo_detection(frame)

    # Display the image with bounding boxes
    cv2.imshow('YOLO Object Detection', yolo_detected_frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

video_capture.release()
cv2.destroyAllWindows()
