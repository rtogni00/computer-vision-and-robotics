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
        # self.mu_prime = init_state.copy()

        self.Sigma = init_covariance.copy()
        # self.Sigma_prime = init_covariance.copy()

        self.ids = []
        self.SIGMA_SQUARED_X = 100
        self.SIGMA_SQUARED_Y = 100

        self.DIST_STD = DIST_STD
        self.ANGLE_STD = np.radians(ANGLE_STD)

        # initial uncertainty on mark position
        self.Sigma_u = np.array(
            [[self.SIGMA_SQUARED_X, 0], [0, self.SIGMA_SQUARED_Y]])

        # uncertainty on measurements
        self.Q = np.array([[self.DIST_STD**2, 0], [0, self.ANGLE_STD**2]])

        return

    def predict(self, l, r):
        # get current poition in world coordinates
        x, y, theta, std = self.get_robot_pose()

        w = self.WIDTH  # robot width from center of the two wheels
        alpha = (r-l)/w
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_theta_rlw = np.sin(theta+alpha)
        cos_theta_rlw = np.cos(theta+alpha)

        # prediction for robot coordinates only
        # new position and covariance matrix
        if l == r:
            x_prime = x + l*(cos_theta)
            y_prime = y + l*(sin_theta)
            theta_prime = theta

            G = np.array([[1, 0, -l*sin_theta],
                          [0, 1, l*cos_theta],
                          [0, 0, 1]])
            A = 0.5*(cos_theta+l/w*sin_theta)
            B = 0.5*(sin_theta-l/w*cos_theta)
            C = 0.5*(cos_theta-l/w*sin_theta)
            D = 0.5*(sin_theta+l/w*cos_theta)

        else:
            R = l/alpha
            R_w_2 = R+(w/2)
            x_prime = x + R_w_2*(np.sin(theta+alpha)-sin_theta)
            y_prime = y + R_w_2*(-np.cos(theta+alpha)+cos_theta)
            theta_prime = (theta+alpha) % (2*np.pi)

            const_AB = (w*r)/(r-l)**2
            const_CD = -(w*l)/(r-l)**2
            const_ABCD = (r+l)/(2*(r-l))
            l_alpha_w_2 = (l/alpha+w/2)

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

        # faster computation for new covariance matrix
        sigma_xx = self.Sigma[0:3, 0:3]
        sigma_xm = self.Sigma[0:3, 3:]
        sigma_mm = self.Sigma[3:, 3:]

        Sigma_top_left = G@sigma_xx@G.T + V @ self.Sigma_u @ V.T
        Sigma_top_right = G@sigma_xm
        Sigma_low_left = (G@sigma_xm).T

        # save updated position and covariance matrix
        mu_prime = self.mu.copy()
        mu_prime[0:3] = np.array([x_prime, y_prime, theta_prime])
        self.mu_prime = mu_prime
        self.Sigma_prime = np.block([[Sigma_top_left, Sigma_top_right],
                                     [Sigma_low_left, sigma_mm]])

    def add_landmark(self, position: tuple, id: str):
        # measurement: tuple, id: str): # changed to a shorter signature of the function
        if int(id) < 1000:
            x, y = position

            # extend self.mu and self.Sigma with the new landmark
            self.mu = np.append(self.mu, [x, y])
            self.mu_prime = self.mu.copy()

            dim = self.Sigma.shape
            Sigma = np.zeros(np.add(dim, 2))
            Sigma[:dim[0], :dim[1]] = self.Sigma
            Sigma[-2, -2] = self.SIGMA_SQUARED_X
            Sigma[-1, -1] = self.SIGMA_SQUARED_Y

            self.Sigma = Sigma
            self.Sigma_prime = Sigma

            # add the landmark id to self.ids
            self.ids.append(id)

    def correction(self, landmark_position_measured: tuple, id: int):
        # # index of identified aruco
        # index = self.ids.index(id)

        # # current world position of robot
        # x, y, theta, _ = self.get_robot_pose()

        # # the measured position fo the aruco from the camera
        # r_i, beta_i = landmark_position_measured

        # # the old position of the markers
        # x_m, y_m = self.mu[3+index:3+index+2]

        # # define the return values of h
        # r = np.sqrt((x_m-x)**2+(y_m-y)**2)
        # beta = self.subtract(np.arctan2((y_m-y), (x_m-x)), theta)

        # # define the entries of jacobian
        # r_x = -(x_m-x)/r
        # r_y = -(y_m-y)/r
        # r_theta = 0
        # r_x_m = -r_x
        # r_y_m = -r_y
        # beta_x = (y_m-y)/r**2
        # beta_y = -(x_m-x)/r*2
        # beta_theta = -1
        # beta_x_m = -beta_x
        # beta_y_m = -beta_y

        # H_s_robot = np.array([[r_x, r_y, r_theta,],
        #                      [beta_x, beta_y, beta_theta,]])
        # H_s_marker = np.array([[r_x_m, r_y_m],
        #                       [beta_x_m, beta_y_m]])

        # n = 2*index
        # m = max(3+2*len(self.ids)-n-5, 0)

        # H = np.block([H_s_robot, np.zeros((2, n)),
        #              H_s_marker, np.zeros((2, m))])
        # H_s = np.block([H_s_robot, H_s_marker])

        # inf = 3+2*index
        # sup = 3+2*index+2

        # sigma_s_top_left = self.Sigma[0:3, 0:3]
        # sigma_s_low_right = self.Sigma[inf:sup, inf:sup]
        # sigma_s_top_right = self.Sigma[0:3, inf:sup]
        # sigma_s_low_left = self.Sigma[inf:sup, 0:3]

        # sigma_s = np.block([[sigma_s_top_left, sigma_s_top_right],
        #                     [sigma_s_low_left, sigma_s_low_right]])

        # Z = H_s@sigma_s@(H_s.T)+self.Q

        # K = self.Sigma_prime@(H.T)@(np.linalg.inv(Z))
        # self.mu = self.mu_prime + \
        #     K@(np.array([r_i-r, self.subtract(beta_i, beta)]).T)
        # self.Sigma = (np.eye(3+2*len(self.ids))-K@H)@self.Sigma_prime
        pass

    def get_robot_pose(self):
        # read out the robot position and angle from mu variable
        # read out robot error from Sigma
        robot_x, robot_y, robot_theta = self.mu[:3].copy()
        sigma = self.Sigma[:2, :2]
        var_x, var_y, _ = self.get_error_ellipse(sigma)
        robot_stdev = [np.sqrt(var_x), np.sqrt(var_y)]

        return robot_x, robot_y, robot_theta, robot_stdev

    def get_landmark_poses(self):
        pass
        landmark_estimated_positions, landmark_estimated_stdevs = [1], [1]
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

    # return the list of all known ids
    def get_landmark_ids(self):
        return np.array(self.ids)

    # helper function to get correct difference of two angle
    def subtract(self, theta_1, theta_2):
        diff = (theta_1-theta_2) % (2*np.pi)
        if diff > np.pi:
            diff = diff-(2*np.pi)
        return diff
