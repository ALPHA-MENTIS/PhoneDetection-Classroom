import cv2
import time
import threading
import numpy as np
from detector import YOLOProcessor
from filters import calculate_entropy, detect_specular_highlight
import os

class VideoCamera(object):
    def __init__(self, source='video.mp4', model_path='../../ALPHA.pt'):
        # Open video source
        # source can be an integer (webcam index) or a string (video file/RTSP url)
        # Attempt to handle relative path for video if it's a file
        if isinstance(source, str) and not source.startswith('rtsp') and not os.path.exists(source):
             # periodic check for file in parent dir if not found (development convenience)
             if os.path.exists(os.path.join(os.path.dirname(__file__), '..', source)):
                 source = os.path.join(os.path.dirname(__file__), '..', source)

        self.video = cv2.VideoCapture(source)
        
        if not self.video.isOpened():
            raise ValueError(f"Unable to open video source: {source}")

        # Initialize Detector
        # Model path is likely in ../../data/ALPHA.pt or relative to execution
        # Adjusting path assumption based on project structure: /home/alpha/ALPHA_PY/final/ALPHA.pt
        real_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ALPHA.pt'))
        if not os.path.exists(real_model_path):
             # Fallback to current dir or specific location if needed
             real_model_path = 'ALPHA.pt'
        
        print(f"Initializing Detector with model: {real_model_path}")
        self.detector = YOLOProcessor(model_path=real_model_path, conf_thres=0.4)
        
        
        self.paused = False
        self.last_frame = None
        
        # History Memory: { track_id: {'glare_seen': bool, 'frames': int, 'buffer': []} }
        self.track_history = {}

    def __del__(self):
        self.video.release()
        
    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def get_frame(self):
        if self.paused and self.last_frame is not None:
             # Just return the previous frame to simulate pause
             # We add a small sleep to prevent busy-looping cpu spike
             time.sleep(0.03)
             return self.last_frame
             
        success, image = self.video.read()
        if not success:
            # Loop video for testing purposes if it's a file
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, image = self.video.read()
            if not success:
                return None
        
        # --- DETECTION PIPELINE ---
        # 1. Detect Objects (Now with Tracking IDs)
        # Returns: [x1, y1, x2, y2, conf, cls, track_id]
        detections = self.detector.detect(image)
        class_names = self.detector.names
        
        # 2. Process Detections
        for x1, y1, x2, y2, conf, cls, track_id in detections:
            # ROI (Region of Interest)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within image bounds
            h, w, _ = image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Identify Class
            label_name = class_names.get(cls, str(cls)).lower()
            
            # --- SPECIAL CASE: PERSON ---
            if 'person' in label_name or 'human' in label_name:
                # Just draw a Blue box for people, no math analysis needed.
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue
                cv2.putText(image, f"Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                continue
                
            roi = image[y1:y2, x1:x2]
            
            # 3. Apply Filters (The "Calculator Test") with MEMORY
            entropy = calculate_entropy(roi)
            glare_score, has_glare = detect_specular_highlight(roi)
            is_shiny_now = glare_score > 0.005
            
            # Update History if we have a valid ID
            if track_id != -1:
                if track_id not in self.track_history:
                    self.track_history[track_id] = {'glare_seen': False, 'frames': 0}
                
                # If we see glare NOW, perform a "Latch" -> It's a phone forever (or for a long time)
                if is_shiny_now:
                    self.track_history[track_id]['glare_seen'] = True
                
                self.track_history[track_id]['frames'] += 1
                
                # Use Memory for decision
                # "Once a Phone (Shiny), Always a Phone" (within this session)
                is_confirmed_phone = self.track_history[track_id]['glare_seen']
            else:
                is_confirmed_phone = is_shiny_now

            # Logic Refined with Time:
            # It is a calculator ONLY IF:
            # 1. High Entropy (Buttons)
            # 2. AND NEVER seen glare (Matte)
            
            is_high_entropy = entropy > 5.5
            
            is_likely_calculator = is_high_entropy and not is_confirmed_phone
            
            if is_likely_calculator:
                color = (0, 0, 255) # Red for Calculator
                # Debug info
                label = f"#{track_id} Calc E:{entropy:.1f}"
            else:
                color = (0, 255, 0) # Green for Phone
                # Show why we think it's a phone
                reason = "G-Mem" if is_confirmed_phone else "L-Ent"
                label = f"#{track_id} {label_name} {reason} E:{entropy:.1f}"
            
            # Draw Box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Encode for streaming
        ret, jpeg = cv2.imencode('.jpg', image)
        self.last_frame = jpeg.tobytes()
        return self.last_frame
