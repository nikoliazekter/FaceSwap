import cv2
import numpy as np

from config import Config
from detectors import get_detector
from image_data import ImageData
from keycodes import Keycode
from warps import get_warp

WINDOW_HEIGHT = 720
WINDOW_WIDTH = 1280
WINDOW_NAME = 'Face swap'


def apply_scale_offset(points, scale, offset_x, offset_y):
    if not points:
        return

    if type(points[0][0]) == int:  # list of points
        return [(round(x * scale + offset_x), round(y * scale + offset_y)) for x, y in points]
    else:  # list of lists of points
        return [[(round(x * scale + offset_x), round(y * scale + offset_y)) for x, y in ps] for ps in points]


def resize_to_fit(image, width, height):
    h, w = image.shape[:2]

    image_ratio = w / h
    box_ratio = width / height
    if box_ratio > image_ratio:  # box is wider than image -> fit height
        new_h = height
        new_w = w * height // h
        offset_x = (width - new_w) // 2
        offset_y = 0
        scale = height / h
    else:  # image is wider than box -> fit width
        new_w = width
        new_h = h * width // w
        offset_x = 0
        offset_y = (height - new_h) // 2
        scale = width / w

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    new_image = np.zeros((height, width, 3), np.uint8)
    new_image[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_image
    return new_image, offset_x, offset_y, scale


def draw_rectangle(screen, rect, scale, offset_x, offset_y, color=(0, 0, 255), thickness=2):
    if rect is None:
        return

    p1, p2 = apply_scale_offset(rect, scale, offset_x, offset_y)
    cv2.rectangle(screen, p1, p2, color=color, thickness=thickness)


def draw_landmarks(screen, landmarks, scale, offset_x, offset_y, color=(255, 0, 0), thickness=3):
    if landmarks is None:
        return

    points = apply_scale_offset(landmarks, scale, offset_x, offset_y)
    for p in points:
        cv2.circle(screen, p, radius=1, color=color, thickness=thickness)


def draw_triangles(screen, triangles, scale, offset_x, offset_y, color=(255, 255, 255)):
    vertices = apply_scale_offset(triangles, scale, offset_x, offset_y)
    for v1, v2, v3 in vertices:
        cv2.line(screen, v1, v2, color=color, thickness=1)
        cv2.line(screen, v2, v3, color=color, thickness=1)
        cv2.line(screen, v3, v1, color=color, thickness=1)


def redraw_screen(screen_width, screen_height, first_data, second_data, config):
    screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    image1 = first_data.swapped if config.swap_first else first_data.original
    image2 = second_data.swapped if config.swap_second else second_data.original

    width1 = screen_width // 2
    image1, offset_x1, offset_y1, scale1 = resize_to_fit(image1, width1, screen_height)
    screen[:, :width1, :] = image1

    width2 = screen_width - width1
    image2, offset_x2, offset_y2, scale2 = resize_to_fit(image2, width2, screen_height)
    offset_x2 += width1
    screen[:, -width2:, :] = image2

    if config.show_rectangles:
        rect1 = first_data.face_rect
        draw_rectangle(screen, rect1, scale1, offset_x1, offset_y1)
        rect2 = second_data.face_rect
        draw_rectangle(screen, rect2, scale2, offset_x2, offset_y2)

    if config.show_landmarks:
        landmarks1 = first_data.landmarks
        draw_landmarks(screen, landmarks1, scale1, offset_x1, offset_y1)
        landmarks2 = second_data.landmarks
        draw_landmarks(screen, landmarks2, scale2, offset_x2, offset_y2)

    if config.warp == 'affine' and config.show_triangles:
        if 'triangles' in first_data.warp_data:
            triangles1 = first_data.warp_data['triangles']
            draw_triangles(screen, triangles1, scale1, offset_x1, offset_y1)
        if 'triangles' in second_data.warp_data:
            triangles2 = second_data.warp_data['triangles']
            draw_triangles(screen, triangles2, scale2, offset_x2, offset_y2)

    cv2.imshow(WINDOW_NAME, screen)


if __name__ == "__main__":
    config = Config(image_dir='images')
    print(f'Detector: {config.detector}')
    print(f'Warp: {config.warp}')

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    changed_window = True
    prev_window_width = -1
    prev_window_height = -1

    capture = None
    first_data = None
    second_data = None
    detector = None
    warp = None

    while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) > 0:
        window_rect = cv2.getWindowImageRect(WINDOW_NAME)
        window_width, window_height = window_rect[2], window_rect[3]
        if window_height <= 0:
            break

        # check for window size changes
        if window_width != prev_window_width or window_height != prev_window_height:
            prev_window_width = window_width
            prev_window_height = window_height
            changed_window = True

        # turn camera on/off
        if config.changed['mode']:
            if config.mode == 'camera':
                capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                config.changed['first_image'] = True
                if capture is not None:
                    capture.release()

        # read a frame from camera
        if config.mode == 'camera':
            ret, frame = capture.read()
            first_data = ImageData(frame)
            config.changed['first_image'] = True

        # read images from directory
        if config.mode == 'image' and config.changed['first_image']:
            first_data = ImageData(cv2.imread(config.first_image_id))
        if config.changed['second_image']:
            second_data = ImageData(cv2.imread(config.second_image_id))

        # change detector
        if config.changed['detector']:
            detector = get_detector(config.detector)

        # detect face rects and landmarks
        if config.changed['detector'] or config.changed['first_image']:
            detector(first_data)
        if config.changed['detector'] or config.changed['second_image']:
            detector(second_data)

        # change warp
        if config.changed['warp']:
            warp = get_warp(config.warp)

        # do warp preprocessing
        if config.changed['detector'] \
                or config.changed['warp'] \
                or config.changed['first_image']:
            warp.preprocess(first_data)
        if config.changed['detector'] \
                or config.changed['warp'] \
                or config.changed['second_image']:
            warp.preprocess(second_data)

        # swap faces
        if config.swap_first and (config.changed['detector']
                                  or config.changed['warp']
                                  or config.changed['first_image']
                                  or config.changed['second_image']
                                  or config.changed['swap_first']):
            warp(second_data, first_data)
        if config.swap_second and (config.changed['detector']
                                   or config.changed['warp']
                                   or config.changed['first_image']
                                   or config.changed['second_image']
                                   or config.changed['swap_second']):
            warp(first_data, second_data)

        if changed_window \
                or config.changed['first_image'] \
                or config.changed['second_image'] \
                or config.changed['mode'] \
                or config.changed['detector'] \
                or config.changed['warp'] \
                or config.changed['show_rectangles'] \
                or config.changed['show_landmarks'] \
                or config.changed['show_triangles'] \
                or config.changed['swap_first'] \
                or config.changed['swap_second']:
            redraw_screen(window_width, window_height, first_data, second_data, config)

        config.clear_changed()
        changed_window = False

        # handle key presses
        pressed_key = cv2.waitKeyEx(1)
        if pressed_key in Keycode.C:
            config.next_mode()
        elif pressed_key in Keycode.D:
            config.next_detector()
            print(f'Detector: {config.detector}')
        elif pressed_key in Keycode.W:
            config.next_warp()
            print(f'Warp: {config.warp}')
        elif pressed_key in Keycode.R:
            config.toggle_rectangles()
        elif pressed_key in Keycode.L:
            config.toggle_landmarks()
        elif pressed_key in Keycode.T:
            config.toggle_triangles()
        elif pressed_key in Keycode.NUM1:
            config.toggle_swap_first()
        elif pressed_key in Keycode.NUM2:
            config.toggle_swap_second()
        elif pressed_key in Keycode.ARROW_DOWN:
            if config.mode == 'image':
                config.prev_first_image()
        elif pressed_key in Keycode.ARROW_UP:
            if config.mode == 'image':
                config.next_first_image()
        elif pressed_key in Keycode.ARROW_LEFT:
            config.prev_second_image()
        elif pressed_key in Keycode.ARROW_RIGHT:
            config.next_second_image()

    cv2.destroyAllWindows()
