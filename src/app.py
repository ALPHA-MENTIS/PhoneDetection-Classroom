from flask import Flask, render_template, Response
import cv2
import time
from camera import VideoCamera

app = Flask(__name__)

# Initialize camera (using a dummy source or the file initially for testing)
# In production, this will use the RTSP stream or optimized capture
video_camera = None

def get_camera():
    global video_camera
    if video_camera is None:
        # Defaulting to the local video file for development
        video_camera = VideoCamera(source='video.mp4')
    return video_camera

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            # If video ends or fails, maybe restart or yield blank
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(get_camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    camera = get_camera()
    is_paused = camera.toggle_pause()
    return {'paused': is_paused}

if __name__ == '__main__':
    # HACK: Using 0.0.0.0 to be accessible, debug=True for dev
    app.run(host='0.0.0.0', port=5000, debug=True)
