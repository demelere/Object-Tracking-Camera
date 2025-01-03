import cv2
import numpy as np
import subprocess
import io
from PIL import Image

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt",
    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/voc/MobileNetSSD_deploy.caffemodel"
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Start libcamera process
cmd = [
    'libcamera-vid',
    '-t', '0',                # Run indefinitely
    '-n',                     # Don't save to file
    '--width', '640',         # Adjust resolution as needed
    '--height', '480',
    '--framerate', '30',
    '--codec', 'mjpeg',       # Use MJPEG
    '-o', '-'                 # Output to stdout
]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

try:
    # Buffer for the image data
    buffer = io.BytesIO()
    
    while True:
        # Read JPEG header (FF D8)
        while process.stdout.read(1) != b'\xff' or process.stdout.read(1) != b'\xd8':
            continue
            
        buffer.write(b'\xff\xd8')
        
        # Read until JPEG end (FF D9)
        while True:
            byte = process.stdout.read(1)
            buffer.write(byte)
            if byte == b'\xff' and process.stdout.read(1) == b'\xd9':
                buffer.write(b'\xd9')
                break
        
        # Convert to OpenCV format
        buffer.seek(0)
        image = Image.open(buffer)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        
        # Pass blob through network
        net.setInput(blob)
        detections = net.forward()
        
        # Get frame dimensions
        (h, w) = frame.shape[:2]
        
        # Loop over detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                
                # Compute bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw prediction
                label = f"{CLASSES[class_id]}: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Frame", frame)
        
        # Clear buffer for next frame
        buffer.seek(0)
        buffer.truncate()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up
    process.terminate()
    cv2.destroyAllWindows()