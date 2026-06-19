# api/server.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import threading

class VideoStreamer:
    def __init__(self):
        self.app = FastAPI()
        self.current_frame = None
        self.lock = threading.Lock()
        
        @self.app.get("/")
        def read_root():
            return {"System": "CogniGuard API Active", "Endpoint": "/video_feed"}

        @self.app.get("/video_feed")
        def video_feed():
            return StreamingResponse(self.generate_bytes(), media_type="multipart/x-mixed-replace; boundary=frame")

    def update_frame(self, frame):
        with self.lock:
            self.current_frame = frame.copy()

    def generate_bytes(self):
        while True:
            with self.lock:
                if self.current_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', self.current_frame)
                frame_bytes = buffer.tobytes()
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')