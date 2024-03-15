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
        """Returns transformation matrix from camera to marker"""
        
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

    def detections(self, img: np.ndarray, draw_img: None, x: tuple, kind: str = "all") -> tuple:

        # detect arucos and circles
        ids, x_r2m, y_r2m = self.detect_arucos(img, draw_img)
        if kind=="all":
            ids_circles, x_r2circle, y_r2circle = self.detect_circles(
                img, draw_img)
        else:
            ids_circles=[]
            x_r2circle=[]
            y_r2circle=[]

        # concatenate all ids and get all x and y coords
        ids = np.concatenate((ids, ids_circles)).astype(np.int16)
        x_r2landmarks = np.asarray(x_r2m+x_r2circle)
        y_r2landmarks = np.asarray(y_r2m+y_r2circle)

        # coordinates of landmarks wrt to world coordinates
        x_w2landmarks = x_r2landmarks+x[0]
        y_w2landmarks = y_r2landmarks+x[1]

        # if landmarks were found
        if len(ids) > 0:
            landmark_rs = np.sqrt(x_r2landmarks**2 + y_r2landmarks**2)
            landmark_alphas = np.arctan2(y_r2landmarks, x_r2landmarks)
            landmark_positions = np.vstack([x_w2landmarks, y_w2landmarks]).T
        else:
            landmark_rs = []
            landmark_alphas = []
            landmark_positions = []

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

        # each pixel in the binary image is assigned a label
        output = cv2.connectedComponentsWithStats(
            circles_binary_image, 8, cv2.CV_32S)
        # representing the connected component it belongs to.
        # as many random colors as labels
        (numLabels, labels, stats, centroids) = output

        return numLabels, centroids, stats

    def detect_arucos(self, img: np.ndarray, draw_img=None):
        # TODO: check to_tf and frame coordinated system used
        # call to cv2
        corners, ids, _ = self.aruco_det.detectMarkers(img)

        if draw_img is not None:
            cv2.aruco.drawDetectedMarkers(draw_img, corners, ids, (0, 255, 0))

        world_coord_x = []
        world_coord_y = []
        # only if something was found
        if ids is not None:
            ids = ids.flatten().tolist()
            num_ids = len(ids)

            # get rvecs and tvecs of aruco wrt robot
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.aruco_size, self.camera_matrix, self.dist_coeffs)

            for i in range(num_ids):
                # rotation matrix from camera to marker
                t_c2m = self.to_tf(rvecs[i], tvecs[i])
                # dot product to get rotation matrix from robot to marker
                t_r2m = np.dot(self.t_r2c, t_c2m)
                # save only relevant cordinates
                x, y = t_r2m[:2, 3]
                world_coord_x.append(x)
                world_coord_y.append(y)
        else:
            ids = []
        return ids, world_coord_x, world_coord_y

    def detect_circle(self, img: np.ndarray, color, lower, upper, draw_img=None):
        output = self.find_centroids(img, lower, upper)
        (numLabels, centroids, stats) = output

        ids = []
        world_coord_x = []
        world_coord_y = []

        if color == 'green':
            offset = 1
        else:
            offset = 0

        if numLabels > 0:
            counter = 1002+offset
            for i in range(1, numLabels):
                ids.append(counter)
                x = centroids[i, cv2.CC_STAT_LEFT]
                y = centroids[i, cv2.CC_STAT_TOP]
                if draw_img is not None:
                    left = stats[i, cv2.CC_STAT_LEFT]
                    top = stats[i, cv2.CC_STAT_TOP]
                    height = stats[i, cv2.CC_STAT_HEIGHT]
                    width = stats[i, cv2.CC_STAT_WIDTH]
                    draw_img = cv2.rectangle(
                        draw_img, (left, top), (left+width, top+height),  (0, 255, 0), 2)
                counter = counter+2
                x, y, _ = self.img_to_world([x, y, 1])
                world_coord_x.append(x)
                world_coord_y.append(y)

        return ids, world_coord_x, world_coord_y

    def detect_circles(self, img: np.ndarray, draw_img=None):

        ids_r, w_c_r_x, w_c_r_y = self.detect_circle(
            img, 'red', self.red_lower, self.red_upper, draw_img)
        ids_g, w_c_g_x, w_c_g_y = self.detect_circle(
            img, 'green', self.green_lower, self.green_upper, draw_img)

        # join the lists of ids and coordinates of circles
        ids = ids_r+ids_g
        world_coordinates_x = w_c_r_x+w_c_g_x
        world_coordinates_y = w_c_r_y+w_c_g_y

        return ids, world_coordinates_x, world_coordinates_y
