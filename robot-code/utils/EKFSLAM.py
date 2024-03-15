import numpy as np
from rich import print
from timeit import default_timer as timer


class EKFSLAM:
    def __init__(self,
        WHEEL_RADIUS,
        WIDTH,

        MOTOR_STD,
        DIST_STD,
        ANGLE_STD,

        init_state: np.ndarray = np.zeros(3),
        init_covariance: np.ndarray = np.zeros((3,3))
    ):
        self.WHEEL_RADIUS = WHEEL_RADIUS
        self.WIDTH = WIDTH

        self.mu = init_state.copy()
        self.Sigma = init_covariance.copy()
        self.ids = []

        self.DIST_STD = DIST_STD
        self.ANGLE_STD = np.radians(ANGLE_STD)
        return

    def predict(self, l, r):
        pass

    def add_landmark(self, position: tuple, measurement: tuple, id: str):
        pass

    def correction(self, landmark_position_measured: tuple, id: int):
        pass

    def get_robot_pose(self):
        pass
        robot_x, robot_y, robot_theta, robot_stdev = 0, 0, 0, [0, 0]
        return robot_x, robot_y, robot_theta, robot_stdev

    def get_landmark_poses(self):
        pass
        landmark_estimated_positions, landmark_estimated_stdevs = [], []
        return landmark_estimated_positions, landmark_estimated_stdevs
    
    def get_error_ellipse(self, covariance):
        pass

    def get_landmark_ids(self):
        return np.array(self.ids)
