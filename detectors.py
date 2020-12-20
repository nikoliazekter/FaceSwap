import os

import dlib
import yaml

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'
from tddfa_v2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from tddfa_v2.TDDFA_ONNX import TDDFA_ONNX


class DlibDetector:

    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_detector = dlib.shape_predictor('dlib_data/shape_predictor_68_face_landmarks.dat')

    def __call__(self, image_data):
        image = image_data.original
        rects = self.face_detector(image, 0)  # don't upscale image
        if not rects:
            return

        rect = rects[0]
        detection = self.landmark_detector(image, rect)
        landmarks = []
        for i in range(detection.num_parts):
            point = detection.part(i)
            landmarks.append((point.x, point.y))

        image_data.set_detection([(rect.left(), rect.top()), (rect.right(), rect.bottom())], landmarks)


class TddfaDetector:

    def __init__(self):
        cfg = yaml.load(open('tddfa_v2/configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        self.face_detector = FaceBoxes_ONNX()
        self.tddfa = TDDFA_ONNX(**cfg)

    def __call__(self, image_data):
        image = image_data.original
        rects = self.face_detector(image)
        if not rects:
            return

        rect = rects[0]
        param_lst, roi_box_lst = self.tddfa(image, [rect])
        vertices = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
        landmarks = []
        for i in range(vertices.shape[1]):
            x = round(vertices[0, i])
            y = round(vertices[1, i])
            landmarks.append((x, y))

        image_data.set_detection([(round(rect[i * 2]), round(rect[i * 2 + 1])) for i in range(2)], landmarks)


dlib_detector = DlibDetector()
tddfa_detector = TddfaDetector()


def get_detector(detector_name):
    if detector_name == 'dlib':
        return dlib_detector
    if detector_name == '3ddfa_v2':
        return tddfa_detector
