# main.py
import cv2
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import SystemConfig
from geometry.calibration import CameraCalibrator
from features.sift_tracker import DocumentTracker
from detection.wronskian_gmm import MotionAnalytics
from models.fusion import SentryViT
from utils.visualizer import UI_Visualizer

from db.models import SessionLogger
from analytics.report_gen import AnalyticsEngine

def main():
    print("[BOOT] Initializing SentryViT Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init Subsystems
    calibrator = CameraCalibrator()
    doc_tracker = DocumentTracker()
    motion_sys = MotionAnalytics()
    visuals = UI_Visualizer()
    
    # Init SentryViT Deep Learning Core
    brain = SentryViT().to(device)
    brain.eval()

    # Initialize Database Logger & Analytics Engine
    db_logger = SessionLogger()
    analytics = AnalyticsEngine()

    cap = cv2.VideoCapture('demo.mp4')
    frame_buffer = []
    prev_gray = None
    
    current_state = "SCANNING DESK..."
    motion_percentage = 0.0

    print("[SYSTEM] SentryViT Active. Press 'q' on the video window to terminate.")
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("[INFO] Video feed ended or file closed.")
            break
        
        # 1. Camera Geometry Pipeline
        warped_desk = calibrator.affine_desk_warp(frame)
        gray = cv2.cvtColor(warped_desk, cv2.COLOR_BGR2GRAY)
        
        # 2. Motion Tracking Pipeline (GMM)
        gmm_mask = motion_sys.get_gmm_mask(warped_desk)
        if prev_gray is not None:
            _ = motion_sys.wronskian_micro_movement(prev_gray, gray)
        prev_gray = gray
        
        # 3. Feature Matching Pipeline (SIFT)
        kp, _ = doc_tracker.extract_sift(warped_desk)
        vis_frame = visuals.draw_sift(warped_desk, kp)
        
        # 4. Deep Learning Object Classification Inference Pipeline
        vit_img = cv2.resize(warped_desk, SystemConfig.RESIZE_DIM).transpose(2, 0, 1) / 255.0
        frame_buffer.append(vit_img)
        
        if len(frame_buffer) == SystemConfig.SEQUENCE_LENGTH:
            # Clean noise from the GMM mask
            _, clean_mask = cv2.threshold(gmm_mask, 254, 255, cv2.THRESH_BINARY)
            motion_level = np.sum(clean_mask) / 255.0 
            total_pixels = clean_mask.shape[0] * clean_mask.shape[1]
            motion_percentage = (motion_level / total_pixels) * 100
            
            # Action Trigger: Scan only if motion moves past background thresholds
            if motion_percentage > 0.5:
                img_tensor = torch.tensor(vit_img).unsqueeze(0).to(device, dtype=torch.float32)
                detected_object = brain(img_tensor)
                detected_object = detected_object.split(',')[0].upper()
                
                # Sentry Contraband Logic Rules
                contraband_items = [
                    "CELLULAR TELEPHONE", "CELLPHONE", "IPOD", 
                    "LAPTOP", "NOTEBOOK", "HAND-HELD COMPUTER",
                    "CALCULATOR", "DIGITAL WATCH"
                ]
                
                if any(item in detected_object for item in contraband_items):
                    current_state = f"THREAT: {detected_object}"
                else:
                    current_state = f"VERIFIED SAFE: {detected_object}"
                    
                print(f"[SENTRY SCAN] GMM Motion: {motion_percentage:.2f}% -> {current_state}")
                db_logger.log_state(current_state)
            else:
                current_state = "MONITORING SECURE DESK..."
                
            frame_buffer.clear()

        # UI Window Overlays
        vis_frame = visuals.overlay_status(vis_frame, current_state, motion_percentage)
        
        cv2.imshow('SentryViT: Master Proctor Vision', vis_frame)
        cv2.imshow('SentryViT: GMM Motion Isolation', gmm_mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up hooks and run analytical rollup
    cap.release()
    cv2.destroyAllWindows()
    
    # Generate the security audit summary report
    analytics.generate_terminal_report()

if __name__ == "__main__":
    main()