# features/sift_tracker.py
import cv2

class DocumentTracker:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.hog = cv2.HOGDescriptor()

    def extract_sift(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, None)

    def extract_hog(self, image):
        resized = cv2.resize(image, (64, 128))
        return self.hog.compute(resized)