import cv2


class ImageData:

    def __init__(self, image, width=700, height=700):
        # resize image for better detection
        h, w = image.shape[:2]
        image_ratio = w / h
        box_ratio = width / height
        if box_ratio > image_ratio:  # box is wider than image -> fit height
            new_h = height
            new_w = w * height // h
        else:  # image is wider than box -> fit width
            new_w = width
            new_h = h * width // w
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self.original = image
        self.swapped = image
        self.face_rect = None
        self.landmarks = None
        self.warp_data = {}

    def set_detection(self, rect, landmarks):
        h, w = self.original.shape[:2]
        for x, y in landmarks:
            if not ((0 <= x < w) and (0 <= y < h)):
                return
        self.face_rect = rect
        self.landmarks = landmarks
