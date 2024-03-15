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
                 init_covariance: np.ndarray = np.zeros((3, 3))
                 ):
        self.WHEEL_RADIUS = WHEEL_RADIUS
        self.WIDTH = WIDTH

        # Initialize state (mean and covariance)
        self.mu = init_state.copy()
        self.mu_prime = init_state.copy()

        self.Sigma = init_covariance.copy()
        self.Sigma_prime = init_covariance.copy()
        
        self.ids = []
        self.SIGMA_SQUARED_X = 100
        self.SIGMA_SQUARED_Y = 100

        self.DIST_STD = DIST_STD
        self.ANGLE_STD = np.radians(ANGLE_STD)
        return

    def predict(self, l, r):
        x, y, theta, std = self.get_robot_pose()
        w = self.WIDTH
        alpha = (r-l)/w
        R = l/alpha
        R_w_2 = R+(w/2)
        sin_theta = np.sin(theta)
        sin_theta_rlw = np.sin(theta+alpha)
        cos_theta_rlw = np.cos(theta+alpha)

        cos_theta = np.cos(theta)
        const_AB = (w*r)/(r-l)**2
        const_CD = -(w*l)/(r-l)**2
        const_ABCD = (r+l)/(2*(r-l))
        l_alpha_w_2 = (l/alpha+w/2)

        # prediction for state vector
        x_prime = x + R_w_2*(np.sin(theta+alpha)-sin_theta)
        y_prime = y+R_w_2*(-np.cos(theta+alpha)+cos_theta)
        theta_prime = (theta+alpha) % (2*np.pi)

        self.mu_prime = np.array([x_prime, y_prime, theta_prime])

        # prediction for covariance matrix
        if l == r:
            G = np.array([[1, 0, -l*sin_theta],
                          [0, 1, l*cos_theta],
                          [0, 0, 1]])
            A = 0.5*(cos_theta+l/w*sin_theta)
            B = 0.5*(sin_theta-l/w*cos_theta)
            C = 0.5*(cos_theta-l/w*sin_theta)
            D = 0.5*(sin_theta+l/w*cos_theta)
            
            V = np.array([[A, C],
                          [B, D],
                          [-1/w, 1/w]])
        else:
            G = np.array([[1, 0, l_alpha_w_2*(cos_theta_rlw-cos_theta)],
                          [0, 1, l_alpha_w_2*(sin_theta_rlw-sin_theta)],
                          [0, 0, 1]])
            
            A = const_AB*(sin_theta_rlw-sin_theta)-const_ABCD*cos_theta_rlw
            B = const_AB*(-cos_theta_rlw+cos_theta)-const_ABCD*sin_theta_rlw
            C = const_CD*(sin_theta_rlw-sin_theta)+const_ABCD*cos_theta_rlw
            D = const_CD*(-cos_theta_rlw+cos_theta)+const_ABCD*sin_theta_rlw

            V = np.array([[A, C],
                          [B, D],
                          [-1/w, 1/w]])
        
        Sigma_u = np.array([[self.SIGMA_SQUARED_X, 0],[0, self.SIGMA_SQUARED_Y]])

        # TODO: change dimensions of G and V according to number of landmark added
        self.Sigma_prime = G @ self.Sigma @ G.T + V @ Sigma_u @ V.T


    def add_landmark(self, position: tuple, measurement: tuple, id: str):
        if int(id) < 1000:
            x, y = position

            # extend self.mu and self.Sigma with the new landmark
            self.mu = np.append(self.mu, [x, y])

            dim = self.Sigma.shape
            Sigma = np.zeros(np.add(dim, 2))
            Sigma[:dim[0], :dim[1]] = self.Sigma
            Sigma[-2, -2] = self.SIGMA_SQUARED_X
            Sigma[-1, -1] = self.SIGMA_SQUARED_Y

            self.Sigma = Sigma

            # add the landmark id to self.ids
            self.ids.append(id)

    def correction(self, landmark_position_measured: tuple, id: int):
        pass

    def get_robot_pose(self):
        # pass
        # robot_x, robot_y, robot_theta, robot_stdev = 0, 0, 0, [0, 0]
        # read out the robot position and angle from mu variable
        # read out robot error from Sigma
        robot_x, robot_y, robot_theta = self.mu[:3].copy()
        sigma = self.Sigma[:2, :2]
        var_x, var_y, _ = self.get_error_ellipse(sigma)
        robot_stdev = [np.sqrt(var_x), np.sqrt(var_y)]

        return robot_x, robot_y, robot_theta, robot_stdev

    def get_landmark_poses(self):
        pass
        landmark_estimated_positions, landmark_estimated_stdevs = [], []
        return landmark_estimated_positions, landmark_estimated_stdevs

    def get_error_ellipse(self, covariance):
        # get eigenvalues and eigenvectors of covariance matrix
        # check the first eigenvalue is the largest
        eigen_vals, eigen_vec = np.linalg.eig(covariance)

        if eigen_vals[0] >= eigen_vals[1]:
            i = 0
            j = 1
        else:
            i = 1
            j = 0

        angle = np.arctan2(eigen_vec[i][1], eigen_vec[i][0])
        return np.sqrt(eigen_vals[i]), np.sqrt(eigen_vals[j]), angle

    def get_landmark_ids(self):
        return np.array(self.ids)
