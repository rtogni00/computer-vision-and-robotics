from __future__ import annotations
import cv2
import numpy as np
from rich import print
from utils.opencv_utils import putBText
from scipy.spatial.transform import Rotation
from scipy import optimize
from enum import Enum
from utils.utils import boundary


class Vision:
    def __init__(self, camera_matrix, dist_coeffs, cam_config) -> None:

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.cam_config = cam_config

    def detections(self, img: np.ndarray, draw_img:np.ndarray, robot_pose: tuple, kind: str = "aruco") -> tuple:
        pass
        ids, landmark_rs, landmark_alphas, landmark_positions = [], [], [], []
        return ids, landmark_rs, landmark_alphas, landmark_positions



