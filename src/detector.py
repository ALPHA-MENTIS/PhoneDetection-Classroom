from ultralytics import YOLO
import torch
import cv2

class YOLOProcessor:
    def __init__(self, model_path, conf_thres=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLOv8 model from {model_path} to {self.device}...")
        
        try:
            self.model = YOLO(model_path)
            # Warmup
            # self.model.warmup(imgsz=(640, 640))
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
            
        self.conf_thres = conf_thres
    
    @property
    def names(self):
        return self.model.names

    def detect(self, frame):
        """
        Runs inference (Tracking) on a single frame.
        Returns list of detections: [x1, y1, x2, y2, conf, cls, track_id]
        """
        # Run inference with Tracking
        # persist=True is crucial for video tracking
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.conf_thres, verbose=False)
        
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Get Track ID if available
                id = -1
                if box.id is not None:
                    id = int(box.id[0].cpu().numpy())
                
                detections.append([x1, y1, x2, y2, conf, cls, id])
                
        return detections
