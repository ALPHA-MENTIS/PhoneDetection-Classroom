from ultralytics import YOLO
import cv2
import numpy as np

class PhoneDetector:
    def __init__(self, model_path='/home/alpha/ALPHA PY/phone-detection-app/best.pt'):
        """
        Initialize the YOLO model for phone detection
        
        Args:
            model_path: Path to your YOLOv8 model file
        """
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        
        # Class ID for 'cell-phones' in your custom model is 0
        self.phone_class_id = 0
        
        # Store last detection count for API endpoint
        self.last_phone_count = 0
        
        print("YOLO model loaded successfully!")
    
    def detect(self, frame):
        """
        Detect phones in a frame with multi-scale detection
        
        Args:
            frame: OpenCV image frame (numpy array)
            
        Returns:
            annotated_frame: Frame with bounding boxes drawn
            phone_count: Number of phones detected
        """
        # Run YOLO detection with augmentation for better detection
        # augment=True adds test-time augmentation (flips, scales the image)
        results = self.model(frame, conf=0.2, iou=0.4, augment=True, verbose=False)
        
        # Get the first result
        result = results[0]
        
        # Collect all detected phone boxes
        all_boxes = []
        
        # Loop through all detections
        for box in result.boxes:
            class_id = int(box.cls[0])
            
            if class_id == self.phone_class_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                all_boxes.append({
                    'coords': (x1, y1, x2, y2),
                    'confidence': confidence
                })
        
        # Count unique phones
        phone_count = len(all_boxes)
        
        # Draw bounding boxes on the frame
        annotated_frame = frame.copy()
        
        for box_info in all_boxes:
            x1, y1, x2, y2 = box_info['coords']
            confidence = box_info['confidence']
            
            # Draw red rectangle around phone
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Add label with confidence
            label = f'Phone {confidence:.2f}'
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Add phone count to top of frame
        if phone_count > 0:
            count_text = f'Phones Detected: {phone_count}'
            cv2.putText(annotated_frame, count_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Store count for API
        self.last_phone_count = phone_count
        
        return annotated_frame, phone_count