import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.spatial.distance import pdist, cdist, squareform


def bounding_rect(points):
    xs = [x for x, y in points]
    ys = [y for x, y in points]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return min_x, min_y, max_x, max_y


class AffineWarp:

    def __call__(self, src_data, dst_data):
        if src_data.landmarks is None or dst_data.landmarks is None:
            return

        # get corresponding triangles
        triangles1 = self._triangles_from_ids(dst_data.warp_data['vertices_ids'], src_data.landmarks)
        triangles2 = dst_data.warp_data['triangles']

        # create warped source face image
        warped = np.zeros_like(dst_data.original, dtype=np.float32)
        for t1, t2 in zip(triangles1, triangles2):
            self._warp_triangle(src_data, t1, t2, warped)

        # blend destination image and warped source face image
        mask = dst_data.warp_data['mask']
        hull_rect = dst_data.warp_data['hull_rect']
        center = ((hull_rect[0] + hull_rect[2]) // 2, (hull_rect[1] + hull_rect[3]) // 2)
        dst_data.swapped = cv2.seamlessClone(warped, dst_data.original, mask, center, cv2.NORMAL_CLONE)

    def preprocess(self, image_data):
        landmarks = image_data.landmarks
        if landmarks is None:
            return

        # perform Delaunay triangulation
        subdiv = cv2.Subdiv2D(bounding_rect(landmarks))
        for p in landmarks:
            subdiv.insert(p)

        triangles = []
        vertices_ids = []
        for t in subdiv.getTriangleList():
            vertices = [(round(t[i * 2]), round(t[i * 2 + 1])) for i in range(3)]
            triangles.append(vertices)

            # find vertex indices in landmarks list
            ids = []
            for x, y in vertices:
                for i in range(len(landmarks)):
                    px, py = landmarks[i]
                    if px == x and py == y:
                        ids.append(i)
                        break
            vertices_ids.append(ids)

        image_data.warp_data['triangles'] = triangles
        image_data.warp_data['vertices_ids'] = vertices_ids

        # find convex hull of landmark points and construct face mask
        hull_points = np.int32(
            cv2.convexHull(np.float32(landmarks), returnPoints=True)).squeeze()
        image_data.warp_data['mask'] = cv2.fillConvexPoly(np.zeros_like(image_data.original), hull_points,
                                                          color=(255, 255, 255), lineType=cv2.LINE_AA)
        image_data.warp_data['hull_rect'] = bounding_rect(hull_points)

    def _warp_triangle(self, src_data, t1, t2, warped):
        rect1 = bounding_rect(t1)
        rect2 = bounding_rect(t2)

        # translate triangles so that min_x and min_y are 0
        t1 = np.float32([(x - rect1[0], y - rect1[1]) for x, y in t1])
        t2 = np.float32([(x - rect2[0], y - rect2[1]) for x, y in t2])

        # compute affine transform and apply it to corresponding source image rectangle
        transform = cv2.getAffineTransform(t1, t2)
        src_img = np.float32(src_data.original[rect1[1]:rect1[3] + 1, rect1[0]:rect1[2] + 1])
        warped_img = cv2.warpAffine(src_img, transform, (rect2[2] - rect2[0] + 1, rect2[3] - rect2[1] + 1),
                                    borderMode=cv2.BORDER_REFLECT_101)

        # mask warped image so that only triangle area is visible
        mask = cv2.fillConvexPoly(np.zeros_like(warped_img), np.int32(t2), color=(255, 255, 255),
                                  lineType=cv2.LINE_AA) / 255
        warped_img *= mask

        # add warped triangle to the whole warped face
        warped[rect2[1]:rect2[3] + 1, rect2[0]:rect2[2] + 1] *= (1 - mask)
        warped[rect2[1]:rect2[3] + 1, rect2[0]:rect2[2] + 1] += warped_img

    def _triangles_from_ids(self, vertices_ids, landmarks):
        triangles = []
        for ids in vertices_ids:
            t = [landmarks[i] for i in ids]
            triangles.append(t)
        return triangles


class ThinPlateSplineWarp:

    def __call__(self, src_data, dst_data):
        landmarks1 = src_data.landmarks
        landmarks2 = dst_data.landmarks
        if landmarks1 is None or landmarks2 is None:
            return

        src_points = np.array(landmarks1)
        x_src = src_points[:, 0]
        y_src = src_points[:, 1]
        dst_points = np.array(landmarks2)

        # compute spline parameters
        T = dst_data.warp_data['T']
        x_src_aug = np.concatenate([x_src, np.zeros(3)])
        y_src_aug = np.concatenate([y_src, np.zeros(3)])
        cx = np.linalg.lstsq(T, x_src_aug, rcond=None)[0]
        cy = np.linalg.lstsq(T, y_src_aug, rcond=None)[0]

        warped = np.zeros_like(dst_data.original, dtype=np.float32)
        hull_rect = dst_data.warp_data['hull_rect']

        # get coordinates of all points that lie inside hull_rect
        x = np.arange(hull_rect[0], hull_rect[2] + 1)
        y = np.arange(hull_rect[1], hull_rect[3] + 1)
        x, y = np.meshgrid(x, y)
        xs = x.flatten()
        ys = y.flatten()

        # compute coordinates of corresponding points in source image
        pgLift = self._liftPts(np.stack([xs, ys], 1), dst_points)
        xgt = np.dot(pgLift, cx)
        ygt = np.dot(pgLift, cy)
        img_shape1 = src_data.original.shape
        xgt = xgt.clip(0, img_shape1[1] - 1)
        ygt = ygt.clip(0, img_shape1[0] - 1)

        # create warped face image by interpolation
        warped[ys, xs, 0] = map_coordinates(src_data.original[:, :, 0], [ygt, xgt])
        warped[ys, xs, 1] = map_coordinates(src_data.original[:, :, 1], [ygt, xgt])
        warped[ys, xs, 2] = map_coordinates(src_data.original[:, :, 2], [ygt, xgt])

        # blend destination image and warped source face image
        mask = dst_data.warp_data['mask']
        warped *= mask / 255
        center = ((hull_rect[0] + hull_rect[2]) // 2, (hull_rect[1] + hull_rect[3]) // 2)
        dst_data.swapped = cv2.seamlessClone(warped, dst_data.original, mask, center, cv2.NORMAL_CLONE)

    def preprocess(self, image_data):
        if image_data.landmarks is None:
            return

        image_data.warp_data['T'] = self._makeT(np.array(image_data.landmarks))

        # find convex hull of landmark points and construct face mask
        hull_points = np.int32(
            cv2.convexHull(np.float32(image_data.landmarks), returnPoints=True)).squeeze()
        image_data.warp_data['mask'] = cv2.fillConvexPoly(np.zeros_like(image_data.original), hull_points,
                                                          color=(255, 255, 255), lineType=cv2.LINE_AA)
        image_data.warp_data['hull_rect'] = bounding_rect(hull_points)

    # construct matrix used for calculation of spline parameters
    def _makeT(self, control_points):
        # control_points: [K x 2]
        # T: [(K+3) x (K+3)]
        K = control_points.shape[0]
        T = np.zeros((K + 3, K + 3))
        T[:K, 0] = 1
        T[:K, 1:3] = control_points
        T[K, 3:] = 1
        T[K + 1:, 3:] = control_points.T
        R = squareform(pdist(control_points, metric='euclidean'))
        R = R * R
        R[R == 0] = 1  # a trick to make R ln(R) 0
        R = R * np.log(R)
        np.fill_diagonal(R, 0)
        T[:K, 3:] = R
        return T

    # construct matrix used for transforming points according to spline parameters
    def _liftPts(self, input_points, control_points):
        # input_points: [N x 2]]
        # control_points: [K x 2]]
        # pLift: [N x (3+K)], lifted input points
        N, K = input_points.shape[0], control_points.shape[0]
        pLift = np.zeros((N, K + 3))
        pLift[:, 0] = 1
        pLift[:, 1:3] = input_points
        R = cdist(input_points, control_points, 'euclidean')
        R = R * R
        R[R == 0] = 1
        R = R * np.log(R)
        pLift[:, 3:] = R
        return pLift


affine_warp = AffineWarp()
tps_warp = ThinPlateSplineWarp()


def get_warp(warp_name):
    if warp_name == 'affine':
        return affine_warp
    if warp_name == 'thin_plate_spline':
        return tps_warp
