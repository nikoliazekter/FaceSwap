import os


class Config:
    modes = ['image', 'camera']
    detectors = ['dlib', '3ddfa_v2']
    warps = ['affine', 'thin_plate_spline']

    def __init__(self, *, image_dir):
        self.image_paths = [f'{image_dir}/{path}' for path in os.listdir(image_dir)
                            if path[-4:] in {'.jpg', '.jpeg', '.png'}]
        if len(self.image_paths) == 0:
            raise RuntimeError(f'{image_dir} does not contain images')

        self._mode = 0
        self._detector = 0
        self._warp = 0
        self._first_image_id = 0
        self._second_image_id = 1 % len(self.image_paths)
        self.show_rectangles = False
        self.show_landmarks = False
        self.show_triangles = False
        self.swap_first = False
        self.swap_second = False

        self.changed = {'mode': True, 'detector': True, 'warp': True,
                        'first_image': True, 'second_image': True,
                        'show_rectangles': True, 'show_landmarks': True,
                        'show_triangles': True, 'swap_first': True, 'swap_second': True}

    @property
    def mode(self):
        return Config.modes[self._mode]

    @property
    def detector(self):
        return Config.detectors[self._detector]

    @property
    def warp(self):
        return Config.warps[self._warp]

    @property
    def first_image_id(self):
        return self.image_paths[self._first_image_id]

    @property
    def second_image_id(self):
        return self.image_paths[self._second_image_id]

    def next_mode(self):
        self.changed['mode'] = True
        self._mode = (self._mode + 1) % len(Config.modes)
        return self.mode

    def next_detector(self):
        self.changed['detector'] = True
        self._detector = (self._detector + 1) % len(Config.detectors)
        return self.detector

    def next_warp(self):
        self.changed['warp'] = True
        self._warp = (self._warp + 1) % len(Config.warps)
        return self.warp

    def prev_first_image(self):
        self.changed['first_image'] = True
        self._first_image_id = (self._first_image_id - 1) % len(self.image_paths)
        return self.first_image_id

    def next_first_image(self):
        self.changed['first_image'] = True
        self._first_image_id = (self._first_image_id + 1) % len(self.image_paths)
        return self.first_image_id

    def prev_second_image(self):
        self.changed['second_image'] = True
        self._second_image_id = (self._second_image_id - 1) % len(self.image_paths)
        return self.second_image_id

    def next_second_image(self):
        self.changed['second_image'] = True
        self._second_image_id = (self._second_image_id + 1) % len(self.image_paths)
        return self.second_image_id

    def toggle_rectangles(self):
        self.changed['show_rectangles'] = True
        self.show_rectangles = not self.show_rectangles

    def toggle_landmarks(self):
        self.changed['show_landmarks'] = True
        self.show_landmarks = not self.show_landmarks

    def toggle_triangles(self):
        self.changed['show_triangles'] = True
        self.show_triangles = not self.show_triangles

    def toggle_swap_first(self):
        self.changed['swap_first'] = True
        self.swap_first = not self.swap_first

    def toggle_swap_second(self):
        self.changed['swap_second'] = True
        self.swap_second = not self.swap_second

    def clear_changed(self):
        for key in self.changed:
            self.changed[key] = False
