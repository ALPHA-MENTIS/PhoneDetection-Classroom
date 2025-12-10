from flask import Flask, render_template, Response, request, jsonify
import cv2
from detector import PhoneDetector
import json
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
print(">>> LOADED APP FROM:", __file__)
print(">>> TEMPLATE FOLDER:", app.template_folder)


# Initialize the detector (will load YOLO model)
detector = PhoneDetector(model_path='/home/alpha/ALPHA_PY/phone-detection-app/ALPHA.pt')

# Global variables to store current source
current_source = None
current_source_type = None  # 'camera' or 'video'

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template("index.html")

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global current_source, current_source_type
    
    data = request.json
    source_type = data.get('source_type')
    source_path = data.get('source_path', 0)

    print("==== DEBUG START_DETECTION ====")
    print("Raw request JSON:", data)
    print("Detected source_type:", source_type)
    print("Detected source_path:", source_path)
    print("Type of source_path:", type(source_path))
    print("================================")

    try:
        if source_type == 'camera':
            source_path = int(source_path)
            current_source = cv2.VideoCapture(source_path)
            current_source_type = 'camera'
        else:
            current_source = cv2.VideoCapture(source_path)
            current_source_type = 'video'
        
        if not current_source.isOpened():
            print("ERROR: OpenCV could not open:", source_path)
            return jsonify({'success': False, 'error': 'Could not open source'})
        
        return jsonify({'success': True})
    except Exception as e:
        print("Exception in start_detection:", e)
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


@app.route('/generate_report', methods=['POST'])
def generate_report():
    payload = request.json
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    camera = payload.get("camera", "Camera_01")

    # Read logs between start_date and end_date
    log_entries = []
    log_dir = "logs"

    import os
    from datetime import datetime

    def to_date(d):
        return datetime.strptime(d, "%Y-%m-%d")

    start_d = to_date(start_date)
    end_d = to_date(end_date)

    for folder in os.listdir(log_dir):
        folder_path = os.path.join(log_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            folder_date = to_date(folder)
        except:
            continue

        if start_d <= folder_date <= end_d:
            file_path = os.path.join(folder_path, f"{camera}.jsonl")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            log_entries.append(json.loads(line))
                        except:
                            pass

    if not log_entries:
        return jsonify({"success": False, "message": "No logs found"})

    # Format logs into a compact structure for LLM
    formatted = json.dumps(log_entries, indent=2)

    # Ask local LLM (Ollama) to summarize
    import requests
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": f"Summarize classroom phone usage from the following events:\n\n{formatted}\n\nGive a clear summary, with totals, peak times, patterns, anomalies and overall insight.",
            "stream": False
        }
    )

    summary = response.json().get("response", "")

    return jsonify({
        "success": True,
        "summary": summary
    })

@app.route('/ask_logs', methods=['POST'])
def ask_logs():
    import requests
    data = request.json
    question = data.get("question", "")

    if not question.strip():
        return jsonify({"success": False, "error": "Ask something!"})

    try:
        # Send user's question directly to the model
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": question,
                "stream": False
            },
            timeout=60
        ).json()

        answer = response.get("response", "No response generated.")
        return jsonify({"success": True, "answer": answer})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})





@app.route("/report")
def report_page():
    return render_template("report.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")


if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)