# utils/visualizer.py
import cv2

class UI_Visualizer:
    @staticmethod
    def draw_sift(frame, keypoints):
        return cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 255))
        
    @staticmethod
    def overlay_status(frame, status_text, motion_pct):
        # 1. Create a transparent dark overlay for the HUD background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 2. Dynamic Color Coding based on State
        if status_text == "DEEP FOCUS":
            color = (0, 255, 0)   # Green
        elif status_text == "FATIGUE DETECTED":
            color = (0, 165, 255) # Orange
        else:
            color = (0, 0, 255)   # Red
            
        # 3. Draw the exact Real-Time Metrics
        cv2.putText(frame, f"SYSTEM STATE: {status_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                    
        cv2.putText(frame, f"LIVE MOTION: {motion_pct:.2f}%", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame