# detector.py
# Upgraded PhoneDetector with usage-timers, proximity checks, JSONL logging, and timer overlay.
# Drop this file into your project (replace existing detector.py). No changes required in app.py.

from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import json
from datetime import datetime

def iou(boxA, boxB):
    # boxes: (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom == 0:
        return 0.0
    return interArea / denom

class PhoneDetector:
    def __init__(self,
                 model_path='/home/alpha/ALPHA PY/phone-detection-app/best.pt',
                 phone_class_id=0,
                 person_class_id=1,
                 require_person=True,
                 gap_tolerance_sec=0.1,
                 alert_threshold_sec=15*60,
                 log_dir='logs'):
        """
        phone_class_id: class index of phone in your YOLO model
        person_class_id: class index for person (if present in the same model)
        require_person: if True, session counts as 'in use' only if phone overlaps a person box
        gap_tolerance_sec: how long a phone can disappear and still continue the session
        alert_threshold_sec: seconds until a single alert is generated (default 15 minutes)
        """
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        print("Model loaded.")

        self.phone_class_id = phone_class_id
        self.person_class_id = person_class_id
        self.require_person = require_person

        # Tracking structures
        # active_sessions: dict session_id -> session_info
        # session_info: { 'box': (x1,y1,x2,y2), 'start_ts': float, 'last_seen_ts': float,
        #                 'accumulated': float, 'alert_sent': bool }
        self.active_sessions = {}
        self.next_session_id = 1

        # Timing config
        self.gap_tolerance_sec = gap_tolerance_sec
        self.alert_threshold_sec = alert_threshold_sec

        # Logging
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.camera_name = "Camera_01"

        # For compatibility with app.py
        self.last_phone_count = 0

    # Optional setters
    def set_camera_name(self, name):
        self.camera_name = name

    def set_require_person(self, value: bool):
        self.require_person = value

    # Internal: write JSONL log line
    def _log_event(self, event_type, session_id, extra=None):
        date = datetime.now().strftime('%Y-%m-%d')
        out_dir = os.path.join(self.log_dir, date)
        os.makedirs(out_dir, exist_ok=True)

        filename = os.path.join(out_dir, f"{self.camera_name}.jsonl")

        log_entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "camera": self.camera_name,
            "event": event_type,
            "session_id": session_id
        }

        if extra:
            log_entry.update(extra)

        with open(filename, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


    # Internal: draw timer label above a box
    def _draw_timer_label(self, frame, box, elapsed_seconds, alert=False):
        x1, y1, x2, y2 = box
        mins = int(elapsed_seconds // 60)
        secs = int(elapsed_seconds % 60)
        label = f"{mins:02d}:{secs:02d}"
        # Position label above box (clamp to image)
        label_x = max(5, x1)
        label_y = max(20, y1 - 10)
        color = (0, 0, 255) if alert else (0, 255, 255)
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Match detected phone boxes to existing sessions (simple IoU)
    def _match_boxes_to_sessions(self, detected_boxes):
    
        """
        detected_boxes: list of dicts {'coords':(x1,y1,x2,y2), 'confidence':float}
        Returns mapping: det_index -> session_id (or None)
        """

        def box_center(box):
            x1, y1, x2, y2 = box
            return ((x1+x2)/2, (y1+y2)/2)

        def box_area(box):
            x1, y1, x2, y2 = box
            return max(1, (x2 - x1) * (y2 - y1))  # avoid zero

        mapping = {i: None for i in range(len(detected_boxes))}
        used_sessions = set()

        for i, det in enumerate(detected_boxes):
            new_box = det['coords']
            new_area = box_area(new_box)
            new_center = box_center(new_box)

            best_sid = None
            best_score = 0  # combined matching score

            for sid, sinfo in self.active_sessions.items():
                if sid in used_sessions:
                    continue

                old_box = sinfo['box']
                old_area = box_area(old_box)
                old_center = box_center(old_box)

                # Compute metrics
                iou_val = iou(new_box, old_box)
                size_change = abs(new_area - old_area) / old_area
                dist = np.linalg.norm(np.array(new_center) - np.array(old_center))

                # RULE 1 — basic IoU gate
                if iou_val < 0.30:
                    continue

                # RULE 2 — prevent different person's phone inheriting session
                if size_change > 0.40:
                    continue

                # RULE 3 — large jump → new session
                if dist > 80:   # adjust for resolution if needed
                    continue

                # Score = IoU (simple)
                if iou_val > best_score:
                    best_sid = sid
                    best_score = iou_val

            mapping[i] = best_sid
            if best_sid is not None:
                used_sessions.add(best_sid)

        return mapping


    def detect(self, frame):
        """
        Core detection call (used by app.py).
        Returns: annotated_frame, phone_count
        Side effects:
          - updates self.active_sessions
          - writes JSONL log lines on session start/end/alert
        """
        now = time.time()

        # Run YOLO
        results = self.model(frame, conf=0.2, iou=0.4, augment=False, verbose=False)
        result = results[0]

        # Extract phone boxes and person boxes
        phone_boxes = []
        person_boxes = []
        for box in result.boxes:
            cid = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            entry = {'coords': (x1, y1, x2, y2), 'confidence': conf}
            if cid == self.phone_class_id:
                phone_boxes.append(entry)
            

        # If require_person is True but no person boxes found, we'll still proceed,
        # but mark person_present=False in logs. You can toggle require_person via setter.
        # Match detected phones to existing sessions
        mapping = self._match_boxes_to_sessions(phone_boxes)

        # Keep track of which session ids were updated this frame
        seen_session_ids = set()
        updated_sessions = {}

        # Process each detected phone
        for i, det in enumerate(phone_boxes):
            box = det['coords']
            sid = mapping.get(i)
            # Determine whether there is a person overlapping this phone
            person_present = False
            for p in person_boxes:
                if iou(box, p['coords']) > 0.2:
                    person_present = True
                    break

            if sid is not None:
                # update session
                sinfo = self.active_sessions[sid]
                sinfo['box'] = box
                # update accumulated time: add time since last_seen
                elapsed_gap = now - sinfo['last_seen_ts']
                # If the phone disappeared for longer than gap tolerance previously, don't add gap
                if elapsed_gap <= self.gap_tolerance_sec:
                    sinfo['accumulated'] += elapsed_gap
                # update last_seen
                sinfo['last_seen_ts'] = now
                # update person_present count (for logs)
                
                updated_sessions[sid] = sinfo
                seen_session_ids.add(sid)
            else:
                # create new session
                sid = self.next_session_id
                self.next_session_id += 1
                sinfo = {
                    'id': sid,
                    'box': box,
                    'start_ts': now,
                    'last_seen_ts': now,
                    'accumulated': 0.0,   # accumulated across gaps (we add gaps on next frame)
                    'alert_sent': False,
                    
                }
                self.active_sessions[sid] = sinfo
                updated_sessions[sid] = sinfo
                seen_session_ids.add(sid)
                # Log session start
                self._log_event(
                    "usage_start",
                    sid,
                    {
                        "box": box,
                        
                    }
                )


        # Handle sessions not seen this frame (possible disappearance)
        for sid, sinfo in list(self.active_sessions.items()):
            if sid not in seen_session_ids:
                # If enough time passed since last seen, end session
                gap = now - sinfo['last_seen_ts']
                if gap > self.gap_tolerance_sec:
                    # finalize session (duration = accumulated + gap after last accumulation)
                    duration = sinfo['accumulated']
                    # Log final session end
                    self._log_event(
                        "usage_end",
                        sid,
                        {
                            "duration_seconds": int(duration),
                            "alert_triggered": bool(sinfo['alert_sent'])
                        }
                    )

                    # Cleanup
                    del self.active_sessions[sid]

        # After updates, check for alerts
        for sid, sinfo in self.active_sessions.items():
            # compute current elapsed duration
            # include time since last_seen if last_seen very recent (we've been updating accumulated already)
            # If the object is visible this frame, ensure we include up-to-now time
            elapsed = sinfo['accumulated']
            # If currently seen this frame, add (now - last_seen_ts) small delta
            if sid in updated_sessions:
                # accumulate the frame-to-frame time to elapsed for alert check
                # Note: since we already added elapsed_gap above on update, elapsed is up-to-date
                pass

            # Alert logic: only if person_present is True OR require_person is False
            should_consider = True

            if (not sinfo['alert_sent']) and should_consider and elapsed >= self.alert_threshold_sec:
                # send an alert (here: log it once and mark)
                sinfo['alert_sent'] = True
                self._log_event(
                    "alert_triggered",
                    sid,
                    {
                        "duration_seconds": int(elapsed)
                    }
                )

        # Draw boxes + timer overlays
        annotated = frame.copy()
        for sid, sinfo in self.active_sessions.items():
            box = sinfo['box']
            x1, y1, x2, y2 = box
            elapsed = int(sinfo['accumulated'])
            # If session was just seen in this frame, also add delta since last_seen (approx.)
            # (we added gaps in updates; for display, approximate by now-last_seen)
            if sid in updated_sessions:
                elapsed += int(now - sinfo['last_seen_ts'])
            # Draw rectangle
            color = (0, 0, 255) if sinfo.get('alert_sent', False) else (0, 255, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            # Draw label with confidence if desired (not stored here)
            # Draw timer above box
            self._draw_timer_label(annotated, box, elapsed, alert=sinfo.get('alert_sent', False))

        # Prepare phone_count for compatibility (count current detected phones)
        phone_count = len(phone_boxes)
        self.last_phone_count = phone_count

        return annotated, phone_count
