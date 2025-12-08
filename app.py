from flask import Flask, render_template, Response, request, jsonify
import cv2
from detector import PhoneDetector
import json

app = Flask(__name__)

# Initialize the detector (will load YOLO model)
detector = PhoneDetector(model_path='/home/alpha/ALPHA PY/phone-detection-app/best.pt')

# Global variables to store current source
current_source = None
current_source_type = None  # 'camera' or 'video'

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start detection from camera or video file"""
    global current_source, current_source_type
    
    data = request.json
    source_type = data.get('source_type')  # 'camera' or 'video'
    source_path = data.get('source_path', 0)  # camera index or video path
    
    try:
        if source_type == 'camera':
            current_source = cv2.VideoCapture(int(source_path))
            current_source_type = 'camera'
        else:  # video
            current_source = cv2.VideoCapture(source_path)
            current_source_type = 'video'
        
        if not current_source.isOpened():
            return jsonify({'success': False, 'error': 'Could not open source'})
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop current detection"""
    global current_source
    
    if current_source is not None:
        current_source.release()
        current_source = None
    
    return jsonify({'success': True})

def generate_frames():
    """Generator function to yield frames with detection"""
    global current_source
    
    while current_source is not None and current_source.isOpened():
        success, frame = current_source.read()
        
        if not success:
            # If video ended, loop it or stop
            if current_source_type == 'video':
                current_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        # Run detection on the frame
        detected_frame, phone_count = detector.detect(frame)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', detected_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame with phone count in headers
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'X-Phone-Count: ' + str(phone_count).encode() + b'\r\n\r\n'
               + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_phone_count')
def get_phone_count():
    """Get current phone count (polled by frontend)"""
    if current_source is not None and current_source.isOpened():
        # Return last detection count (you might want to store this globally)
        return jsonify({'count': detector.last_phone_count, 'detecting': True})
    return jsonify({'count': 0, 'detecting': False})

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)