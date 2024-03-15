import numpy as np
from rich import print
from utils.tempo import *
import pickle
from numba import jit


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
        self.Sigma = init_covariance.copy()

        self.ids = []
        self.index_to_ids = {}
        self.n_ids = 0
        self.SIGMA_SQUARED_X = 100
        self.SIGMA_SQUARED_Y = 100
        self.error_l = MOTOR_STD
        self.error_r = MOTOR_STD
        self.std_lr = MOTOR_STD**2
        self.diag = np.eye(2)*self.std_lr

        self.DIST_STD = DIST_STD
        self.ANGLE_STD = np.radians(ANGLE_STD)

        # initial uncertainty on mark position
        self.Sigma_u = np.array(
            [[self.error_l, 0], [0, self.error_r]])

        # uncertainty on measurements
        self.Q = np.array([[self.DIST_STD**2, 0], [0, self.ANGLE_STD**2]])

        return

    @timeit
    def predict(self, l, r):
        x, y, theta, _ = self.get_robot_pose(fastmode=True)
        alpha = (r-l)/self.WIDTH

        # difference in left and right angle  0.1 deg
        # if abs(l - r) <= np.deg2rad(0.1) * WHEEL_RADIUS:
        if np.abs(l-r) <= 0.00004:

            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_theta_rlw = np.sin(theta+alpha)
            cos_theta_rlw = np.cos(theta+alpha)

            l = (l+r)/2
            x += l*(cos_theta)
            x += l*(sin_theta)

            G = np.array([[1, 0, -l*sin_theta],
                          [0, 1, l*cos_theta],
                          [0, 0, 1]], dtype=np.float64)
            A = 0.5*(cos_theta+l/self.WIDTH*sin_theta)
            B = 0.5*(sin_theta-l/self.WIDTH*cos_theta)
            C = 0.5*(cos_theta-l/self.WIDTH*sin_theta)
            D = 0.5*(sin_theta+l/self.WIDTH*cos_theta)

        else:
            R = l/alpha
            R_w_2 = R+(self.WIDTH/2)
            x += R_w_2*(np.sin(theta+alpha)-np.sin(theta))
            y += R_w_2*(-np.cos(theta+alpha)+np.cos(theta))

            theta = (theta+alpha)

            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_theta_rlw = np.sin(theta+alpha)
            cos_theta_rlw = np.cos(theta+alpha)

            const_AB = (self.WIDTH*r)/(r-l)**2
            const_CD = -(self.WIDTH*l)/(r-l)**2
            const_ABCD = (r+l)/(2*(r-l))
            l_alpha_w_2 = (R+self.WIDTH/2)

            G = np.array([[1, 0, l_alpha_w_2*(cos_theta_rlw-cos_theta)],
                          [0, 1, l_alpha_w_2*(sin_theta_rlw-sin_theta)],
                          [0, 0, 1]], dtype=np.float64)

            A = const_AB*(sin_theta_rlw-sin_theta)-const_ABCD*cos_theta_rlw
            B = const_AB*(-cos_theta_rlw+cos_theta)-const_ABCD*sin_theta_rlw
            C = const_CD*(sin_theta_rlw-sin_theta)+const_ABCD*cos_theta_rlw
            D = const_CD*(-cos_theta_rlw+cos_theta)+const_ABCD*sin_theta_rlw

        V = np.array([[A, C],
                      [B, D],
                      [-1/self.WIDTH, 1/self.WIDTH]])

        N = self.n_ids
        if N > 0:
            G = np.block([[G, np.zeros((3, 2*N))],
                         [np.zeros((2*N, 3)), np.eye(2*N)]])
            V = np.append(V, np.zeros((2*N, 2)), axis=0)

        self.Sigma = np.dot(np.dot(G, self.Sigma), G.T) + \
            np.dot(np.dot(V, self.diag), V.T)

        self.mu[0], self.mu[1], self.mu[2] = x, y, theta

    @timeit
    def add_landmark(self, position: tuple, id: str):
        if int(id) <= 1000 and int(id) > 0:
            x, y = position  # array with x and y

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
            self.index_to_ids[id] = self.n_ids
            self.n_ids += 1
            print(f"Landmark with id {id} added")

    @timeit
    def correction(self, landmark_position_measured: tuple, id: int):
        # index of identified aruco
        index = self.index_to_ids[id]

        # current world position of robot
        x, y, theta, _ = self.get_robot_pose(fastmode=True)

        # the measured position fo the aruco from the camera
        r_i, beta_i = landmark_position_measured

        # the old position of the markers
        # mu = [0,0,0, id1, id1, id2, id2, ..., idn, idn]
        x_m, y_m = self.mu[3+2*index:3+2*index+2]

        # using jit acceleration to speed up the heavy lifting
        delta_mu, Sigma_ = correction_acc(x_m, y_m,
                                          x, y,
                                          theta,
                                          r_i, beta_i,
                                          self.n_ids,
                                          index,
                                          self.Q, self.Sigma)

        self.mu += np.squeeze(delta_mu)
        self.Sigma = Sigma_

    # read out the robot position and angle from mu
    # read out robot error from Sigma
    @timeit
    def get_robot_pose(self, fastmode=False):
        robot_x, robot_y, robot_theta = self.mu[:3].copy()
        if not fastmode:
            sigma = self.Sigma[:2, :2]
            error = self.get_error_ellipse(sigma)
        else:
            error = [0, 0, 0]

        return robot_x, robot_y, robot_theta % (2*np.pi), error  # wrt world

    @timeit
    def get_landmark_poses(self, fastmode=False):
        landmark_estimated_positions = []
        landmark_estimated_stdevs = []
        if not fastmode:
            for i in range(self.n_ids):
                sigma_xy = self.Sigma[3+2*i:3+2*i+2, 3+2*i:3+2*i+2]
                landmark_error = self.get_error_ellipse(sigma_xy)
                landmark_estimated_stdevs.append(landmark_error)

            landmark_estimated_positions = self.mu[3:].reshape(self.n_ids, 2)
        else:

            landmark_estimated_positions = self.mu[3:].reshape(self.n_ids, 2)
            landmark_estimated_stdevs = [[0, 0, 0]] * self.n_ids
        return landmark_estimated_positions, landmark_estimated_stdevs

    # important for graphical representations
    @timeit
    def get_error_ellipse(self, covariance):
        # check the first eigenvalue is the largest
        eigen_vals, eigen_vec = np.linalg.eig(covariance)

        if eigen_vals[0] >= eigen_vals[1]:
            i = 0
            j = 1
        else:
            i = 1
            j = 0
        # get eigenvalues and eigenvectors of covariance matrix
        eigvec_x, eigvec_y = np.real(eigen_vec[:, i])
        angle = np.arctan2(eigvec_y, eigvec_x)
        return np.sqrt(np.abs(eigen_vals[i])), np.sqrt(np.abs(eigen_vals[j])), angle

    # return the list of all known ids
    @timeit
    def get_landmark_ids(self):
        return self.ids

    # return the sigma matrix of uncertainties
    def get_sigma(self):
        return self.Sigma

    # return the mu matrix of robot and acos positions
    def get_mu(self):
        return self.mu

    # save state on file
    def dump(self):
        print('DUMPING STATE ON FILE')
        data = {"ids": self.ids,
                "index_to_ids": self.index_to_ids,
                "n_ids": self.n_ids,
                "mu": self.mu,
                "sigma": self.Sigma}
        with open('SLAM_DUMP.pickle', 'wb') as pickle_file:  # dump of all the robot has recorded
            pickle.dump(data, pickle_file)
        print("DUMPING FINISHED")

    # injecct the loaded map tp the EKFSLAM object
    def load_map(self, ids, index_to_ids, n_ids, mu, sigma):
        self.ids = ids
        self.index_to_ids = index_to_ids
        self.n_ids = n_ids
        self.mu = mu
        self.Sigma = sigma
        self.Sigma[3:, 3:] = 0

# jit for heavy lifting


@timeit
@jit(nopython=True, cache=True)
def correction_acc(x_m: float, y_m: float,
                   x: float, y: float,
                   theta: float,
                   r_i: float, beta_i: float,
                   n_ids: int,
                   index: int,
                   Q, Sigma):
    # helper function for the difference between 2 angles
    def subtract(theta_1: float, theta_2: float) -> np.float64:
        diff = (theta_1-theta_2) % (2*np.pi)
        diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
        return diff

    # define the return values of h
    r = np.sqrt((x_m-x)**2+(y_m-y)**2)
    beta = subtract(np.arctan2((y_m-y), (x_m-x)), theta)

    # define the entries of jacobian
    r_x = -(x_m-x)/r
    r_y = -(y_m-y)/r
    r_theta = 0
    r_x_m = -r_x
    r_y_m = -r_y
    beta_x = (y_m-y)/r**2
    beta_y = -(x_m-x)/r**2
    beta_theta = -1
    beta_x_m = -beta_x
    beta_y_m = -beta_y

    H_s_robot = np.array([[r_x, r_y, r_theta,],
                          [beta_x, beta_y, beta_theta,]])
    H_s_marker = np.array([[r_x_m, r_y_m],
                           [beta_x_m, beta_y_m]])

    n = 2*index
    m = max(3+2*n_ids-n-5, 0)
    H = np.concatenate((H_s_robot, np.zeros((2, n)),
                       H_s_marker, np.zeros((2, m))), axis=1)

    H_s = np.concatenate((H_s_robot, H_s_marker), axis=1)

    inf = 3+2*index
    sup = 3+2*index+2

    sigma_s_top_left = Sigma[0:3, 0:3]
    sigma_s_low_right = Sigma[inf:sup, inf:sup]
    sigma_s_top_right = Sigma[0:3, inf:sup]
    sigma_s_low_left = Sigma[inf:sup, 0:3]

    sigma_s_top = np.concatenate((sigma_s_top_left, sigma_s_top_right), axis=1)
    sigma_s_bottom = np.concatenate(
        (sigma_s_low_left, sigma_s_low_right), axis=1)
    sigma_s = np.concatenate((sigma_s_top, sigma_s_bottom), axis=0)

    Z = H_s@sigma_s@(H_s.T)+Q

    K = Sigma@H.T@np.linalg.inv(Z)
    angle = subtract(beta_i, beta)
    temp = np.zeros((2, 1))
    temp[0, 0] = r_i-r
    temp[1, 0] = angle
    delta_mu = np.dot(K, temp)
    Sigma = (np.eye(3+2*n_ids)-K@H)@Sigma

    return delta_mu, Sigma
