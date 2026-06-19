# detection/wronskian_gmm.py
import cv2
import numpy as np
from config import SystemConfig

class MotionAnalytics:
    def __init__(self):
        self.gmm = cv2.createBackgroundSubtractorMOG2(
            history=SystemConfig.GMM_HISTORY, varThreshold=16)

    def get_gmm_mask(self, frame):
        return self.gmm.apply(frame)

    def wronskian_micro_movement(self, f_t1, f_t2):
        """ Linear algebra spatial context for micro-changes """
        f1 = f_t1.astype(np.float32) + 1e-5
        f2 = f_t2.astype(np.float32)
        ratio = np.divide(f2, f1)
        variance = np.square(ratio) - ratio
        change_mask = (np.abs(variance) > SystemConfig.WRONSKIAN_THRESHOLD).astype(np.uint8) * 255
        return change_mask