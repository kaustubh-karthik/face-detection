import cv2
import mediapipe as mp
import numpy as np
import time
class detector():
    def __init__(self, detection, static_img_mode = False, detection_conf = 0.5, track_conf = 0.5):

        detector_object = {
            'hands': [mp.solutions.hands, mp.solutions.hands.Hands],
            'left_hand': [mp.solutions.hands, mp.solutions.hands.Hands],
            'right_hand': [mp.solutions.hands, mp.solutions.hands.Hands],
            'pose': [mp.solutions.pose, mp.solutions.pose.Pose],
            'face_mesh': [mp.solutions.face_mesh, mp.solutions.face_mesh.FaceMesh],
            'holistic': [mp.solutions.holistic, mp.solutions.holistic.Holistic],
            'face_detection': [mp.solutions.face_detection, mp.solutions.face_detection.FaceDetection]
        }

        detector_type = {
            'hands': 'draw_landmarks',
            'left_hand': 'draw_landmarks',
            'right_hand': 'draw_landmarks',
            'pose': 'draw_landmarks',
            'face_mesh': 'draw_landmarks',
            'holistic': 'triple_draw',
            'face_detection': 'results_only'
        }

        self.object_draw_details = {
            'hands': ['results.multi_hand_landmarks', 'self.mp_object.HAND_CONNECTIONS'],
            'left_hand': ['results.left_hand_landmarks', 'self.mp_object.HAND_CONNECTIONS'],
            'right_hand': ['results.right_hand_landmarks', 'self.mp_object.HAND_CONNECTIONS'],
            'pose': ['results.pose_landmarks', 'self.mp_object.POSE_CONNECTIONS'],
            'face_mesh': ['results.face_landmarks', 'self.mp_object.FACEMESH_TESSELATION']
        }

        self.object_iteration = {
            'hands': 'results.multi_hand_landmarks',
            'left_hand': '[results.left_hand_landmarks]',
            'right_hand': '[results.right_hand_landmarks]',
            'pose': '[results.pose_landmarks]',
            'face_mesh': 'results.face_landmarks'
        }

        self.detection = detection

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_object = detector_object[detection][0]
        self.object = detector_object[detection][1]()

        if detector_type[detection] == 'draw_landmarks':
            
            self.results_check = self.object_draw_details[detection][0]
            self.connection_details = self.object_draw_details[detection][1]
            self.iteration = self.object_iteration[detection]

        self.prev_time = 0

    @staticmethod
    def get_rgb_img(img):

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return rgb_img

    def get_results(self, img):

        results = self.object.process(img)

        return results

    def draw_box(self, img, draw = True):

        results = self.get_results(img)

        box_coords = []

        if results.detections:

            for detection in results.detections:
                
                bound_box = detection.location_data.relative_bounding_box
                height, width, channels = img.shape

                box_coords = [int(bound_box.xmin * width), int(bound_box.ymin * height), int(bound_box.width * width), int(bound_box.height * height)]

                if draw:
                    cv2.rectangle(img, box_coords, (255, 0, 0), 2)
            
        return img, box_coords

    def draw_holistic(self, img, detectors):

        rgb_img = detector.get_rgb_img(img)

        results = self.get_results(rgb_img)

        lm_list = []

        for detect in detectors:
            
            self.mp_draw.draw_landmarks(img, eval(detect.results_check), eval(detect.connection_details))

        return img


    def find_lms(self, img, draw = True, lm_id = None):

        rgb_img = detector.get_rgb_img(img)

        results = self.object.process(rgb_img)

        lm_list = []

        if eval(self.results_check):

            for lms in eval(self.iteration):

                self.mp_draw.draw_landmarks(img, lms, eval(self.connection_details))

                for id, lm in enumerate(lms.landmark):
                    
                    height, width, channel = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    lm_list.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0))

        if lm_id != None:
            print(lm_list[lm_id])

        return img, lm_list

    def run_details(self):
        if self.detection == 'holistic':

            detectors = detector('pose'), detector('face_mesh'), detector('left_hand'), detector('right_hand')

            return detectors

    def runtime(self, capture, details, detect):
        if details:
            while True:
                success, img = capture.read()

                img = detect.draw_holistic(img, details)   
                #img, lm_list = detect.find_lms(img)

                detect.display_img(img)

        elif self.detection == 'face_detection':

             while True:
                success, img = capture.read()

                img, box_coords = detect.draw_box(img)

                detect.display_img(img)
                
        else:
            while True:
                success, img = capture.read()

                img, lm_list = detect.find_lms(img, draw=False)

                detect.display_img(img)

        


    def display_img(self, img, fps = True):

        if fps:
            curr_time = time.time()
            fps = 1/(curr_time - self.prev_time)
            self.prev_time = curr_time


        cv2.putText(
            img,
            text = str(int(fps)),
            org = (10, 70),
            fontFace = cv2.FONT_HERSHEY_COMPLEX,
            fontScale = 3,
            color = (150, 150, 150),
            thickness = 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


def main():

    # Choose one out of 'hands, pose, face_mesh'
    detect = detector('holistic')
    details = detect.run_details()

    cap = cv2.VideoCapture(1) # Check for error

    detect.runtime(cap, details, detect)

if __name__ == '__main__':
    main()
