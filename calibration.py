# geometry/calibration.py
import cv2

class CameraCalibrator:
    @staticmethod
    def affine_desk_warp(frame, angle=0, tx=0, ty=0, scale=1.0):
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        return cv2.warpAffine(frame, M, (w, h))