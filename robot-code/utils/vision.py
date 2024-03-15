from __future__ import annotations
import cv2
import numpy as np
from rich import print
from utils.opencv_utils import putBText
from scipy.spatial.transform import Rotation
from scipy import optimize
from enum import Enum
from utils.utils import boundary
from utils.utils import load_config


class Vision:
    def __init__(self, camera_matrix, dist_coeffs, cam_config, robotgeometries) -> None:

        # hsv color ranges for green/red circles detection
        self.red_lower = np.array([60, 40, 150])
        self.red_upper = np.array([190, 190, 255])
        self.green_lower = np.array([10, 40, 35])
        self.green_upper = np.array([80, 255, 255])

        # get aruco stuff from cv
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_det = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.aruco_params)

        self.aruco_size = robotgeometries.aruco_size  # measured in meters
        self.h_r2c = robotgeometries.h_r2c
        self.v_r2c = robotgeometries.v_r2c
        self.X_rotation = robotgeometries.x_angle
        self.Y_rotation = robotgeometries.y_angle
        self.Z_rotation = robotgeometries.z_angle

        pos_r2c = np.array([self.h_r2c, 0, self.v_r2c])  # y_c = 0

        # rotation angles with intrinsic convention from robot to camera
        r = Rotation.from_euler(
            "ZYX", [np.radians(self.Z_rotation), np.radians(self.Y_rotation), np.radians(self.X_rotation)], degrees=False)
        rot_matrix_r2c = r.as_matrix()

        # rotation matrix from robot to camera
        self.t_r2c = np.identity(4, dtype=float)
        self.t_r2c[:3, :3] = rot_matrix_r2c
        self.t_r2c[:3, 3] = pos_r2c

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.cam_config = cam_config

        extrinics = np.linalg.inv(self.t_r2c)
        K = np.zeros((3, 4))
        K[:3, :3] = self.camera_matrix

        P = K @ extrinics

        # source: Zisserman pg. 157, Multiple View Geometry in Computer Vision Second Edition

        M = P[:3, :3]
        p_4 = P[:3, 3]  # column 4

        self.M_inv = np.linalg.inv(M)

        self.C_tilde = -self.M_inv @ p_4

    # function to transform from one system of reference to another

    def to_tf(self, rvec, tvec, order="xyz"):
        tf = np.identity(4, dtype=float)
        r = Rotation.from_euler(order, rvec, degrees=False)
        rot_matrix = r.as_matrix()
        tf[:3, :3] = rot_matrix
        tf[:3, 3] = tvec
        return tf

    def img_to_world(self, x_img):
        x_tilde = self.M_inv @ x_img
        mu = float(- self.C_tilde[2]/x_tilde[2])  # this solves for Z=0
        X = np.squeeze(mu * x_tilde) + self.C_tilde
        return X

    def detections(self, img: np.ndarray, draw_img: np.ndarray, x: tuple, kind: str = "aruco") -> tuple:

        corners, ids, _ = self.aruco_det.detectMarkers(img)

        if ids is not None:
            # get rvecs and tvecs of aruco wrt robot

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.aruco_size, self.camera_matrix, self.dist_coeffs)

            # vector of displacement from robot to camera (floor is z=0)

            x_r2m = np.zeros(len(ids))
            y_r2m = np.zeros(len(ids))
            z_r2m = np.zeros(len(ids))

            for i, aruco_id in enumerate(ids):
                # rotation matrix from camera to marker
                t_c2m = self.to_tf(rvecs[i], tvecs[i])
                # dot product to get rotation matrix from robot to marker
                t_r2m = np.dot(self.t_r2c, t_c2m)
                x_r2m[i], y_r2m[i], z_r2m[i] = t_r2m[:3, 3]

            # convert to alpha and rho
            # do not take into account the z component
            ids = ids.flatten().tolist()
            landmark_rs = np.sqrt(x_r2m ** 2 + y_r2m ** 2)
            landmark_alphas = np.arctan2(y_r2m, x_r2m)
            landmark_positions = np.zeros_like(landmark_rs)

        else:
            # print("No markers detected")
            # except:
            ids = []
            landmark_rs = np.zeros(0)
            landmark_alphas = np.zeros(0)
            landmark_positions = np.zeros(0)

        # look for circles
        ids_circles, pos_circles_world = self.detect_circles(img)
        # ids_circles = np.asarray(ids_circles)
        print('id:', ids, type(ids))
        print('id circ:', ids_circles, type(ids_circles))
        # updated ids, landmark,....
        if len(ids_circles) > 0:
            x_r2circle = np.zeros(len(ids_circles))
            y_r2circle = np.zeros(len(ids_circles))
            z_r2circle = np.zeros(len(ids_circles))
            ids = np.concatenate((ids, ids_circles))
            for i, pos in enumerate(pos_circles_world):
                x_r2circle[i] = pos[0]
                y_r2circle[i] = pos[1]

            landmark_rs_circles = np.sqrt(x_r2circle**2+y_r2circle**2)
            landmark_alphas_circles = np.arctan2(y_r2circle, x_r2circle)
            landmark_positions_circles = np.zeros_like(landmark_rs_circles)
        else:
            landmark_rs_circles = np.zeros(0)
            landmark_alphas_circles = np.zeros(0)
            landmark_positions_circles = np.zeros(0)
            # pass
            # landmark_rs_circles=np.

        # print(landmark_rs.shape)
        # print(type(landmark_rs))
        # print(landmark_rs_circles.shape)

        # print(type(landmark_rs_circles))
        landmark_rs = np.concatenate((landmark_rs, landmark_rs_circles))
        landmark_alphas = np.concatenate(
            (landmark_alphas, landmark_alphas_circles))
        landmark_positions = np.concatenate(
            (landmark_positions, landmark_positions_circles))
        # print(landmark_rs)
        # print(type(landmark_rs))
        # print(landmark_rs_circles)
        # print(type(landmark_rs_circles))
        # ids, landmark_rs, landm ark_alphas, landmark_positions = [], [], [], []
        ids = np.asarray(ids, np.uint16)
        return ids, landmark_rs, landmark_alphas, landmark_positions

    def find_centroids(self, img, range_lower, range_upper):
        """Takes an input images and returns a list containing the x (left to right) and y (top to bottom)"""

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, range_lower, range_upper)

        circles = cv2.bitwise_and(img, img, mask=mask)

        circles_gray = cv2.cvtColor(circles, cv2.COLOR_BGR2GRAY)
        circles_gray = cv2.GaussianBlur(circles_gray, (5, 5), 0)
        _, circles_binary_image = cv2.threshold(
            circles_gray, 100, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        circles_binary_image = cv2.morphologyEx(
            circles_binary_image, cv2.MORPH_OPEN, kernel, iterations=3)
        circles_binary_image = cv2.morphologyEx(
            circles_binary_image, cv2.MORPH_CLOSE, kernel, iterations=3)

        output = cv2.connectedComponentsWithStats(
            circles_binary_image, 8, cv2.CV_32S)  # each pixel in the binary image is assigned a label
        # representing the connected component it belongs to.
        # as many random colors as labels
        (numLabels, labels, stats, centroids) = output

        return numLabels, centroids, stats

    def detect_arucos(self, img: np.ndarray):
        pass

    def detect_circles(self, img: np.ndarray):

        output_red = self.find_centroids(img, self.red_lower, self.red_upper)
        (numLabels_red, centroids_red, stats_red) = output_red
        output_green = self.find_centroids(
            img, self.green_lower, self.green_upper)
        (numLabels_green, centroids_green, stats_green) = output_green

        ids = []
        world_coordinates = []

        if numLabels_red > 0:  # there is something to do with the red circles
            counter = 1000
            for i in range(1, numLabels_red):
                x = centroids_red[i, cv2.CC_STAT_LEFT]
                y = centroids_red[i, cv2.CC_STAT_TOP]
                ids.append(counter)
                counter = counter+2
                world_coordinates.append(self.img_to_world([x, y, 1]))

        if numLabels_green > 0:  # there is something to do with the green circles
            counter = 1001
            for i in range(1, numLabels_green):
                x = centroids_green[i, cv2.CC_STAT_LEFT]
                y = centroids_green[i, cv2.CC_STAT_TOP]
                ids.append(counter)
                counter = counter+2
                world_coordinates.append(self.img_to_world([x, y, 1]))

        return ids, world_coordinates
