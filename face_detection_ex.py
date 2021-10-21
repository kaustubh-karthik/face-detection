import cv2
from face_detection_utils import detector

cap = cv2.VideoCapture(1) # Check for error

# Choose one out of 'hands, pose, face_mesh'
detect = detector('holistic')
details = detect.run_details()

detect.runtime(cap, details, detect)
