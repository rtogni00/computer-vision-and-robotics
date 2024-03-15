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

        self.Sigma = init_covariance.copy()

        self.ids = []
        self.SIGMA_SQUARED_X = 100
        self.SIGMA_SQUARED_Y = 100
        self.error_l = MOTOR_STD
        self.error_r = MOTOR_STD

        self.DIST_STD = DIST_STD
        self.ANGLE_STD = np.radians(ANGLE_STD)

        # initial uncertainty on mark position
        self.Sigma_u = np.array(
            [[self.error_l, 0], [0, self.error_r]])

        # uncertainty on measurements
        # TODO cancel Q if not used
        self.Q = np.array([[self.DIST_STD**2, 0], [0, self.ANGLE_STD**2]])

        return

    def predict(self, l ,r):
        x, y, theta, std = self.get_robot_pose()
        self.mu, self.Sigma = self._predict(l, r, 
                                            self.WIDTH, self.WHEEL_RADIUS,
                                            x, y, theta, 
                                            self.error_l, self.error_r,
                                            self.ids, 
                                            self.mu, self.Sigma)

    
    def _predict(self, l, r , WIDTH, WHEEL_RADIUS, x, y, theta, error_l, error_r, ids, mu, Sigma):

        w = WIDTH  # robot width from center of the two wheels
        alpha = (r-l)/w
        
        # prediction for robot coordinates only
        # new position and covariance matrix
        if abs(l- r) <= np.radians(1) * WHEEL_RADIUS: # difference in left and right angle  1°
            
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_theta_rlw = np.sin(theta+alpha)
            cos_theta_rlw = np.cos(theta+alpha)
            
            l = np.mean(np.array([l,r]))
            x += l*(cos_theta)
            x += l*(sin_theta)

            G = np.array([[1, 0, -l*sin_theta],
                          [0, 1, l*cos_theta],
                          [0, 0, 1]], dtype=np.float64)
            A = 0.5*(cos_theta+l/w*sin_theta)
            B = 0.5*(sin_theta-l/w*cos_theta)
            C = 0.5*(cos_theta-l/w*sin_theta)
            D = 0.5*(sin_theta+l/w*cos_theta)

        else:
            R = l/alpha
            R_w_2 = R+(w/2)
            x += R_w_2*(np.sin(theta+alpha)-np.sin(theta))
            y += R_w_2*(-np.cos(theta+alpha)+np.cos(theta))
            
            # TODO: check if theta in -pi pi is needed
            theta = (theta+alpha) % (2*np.pi)
            if theta > np.pi:
                theta = theta-2*np.pi

            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_theta_rlw = np.sin(theta+alpha)
            cos_theta_rlw = np.cos(theta+alpha)
        
            const_AB = (w*r)/(r-l)**2
            const_CD = -(w*l)/(r-l)**2
            const_ABCD = (r+l)/(2*(r-l))
            l_alpha_w_2 = (R+w/2)

            G = np.array([[1, 0, l_alpha_w_2*(cos_theta_rlw-cos_theta)],
                          [0, 1, l_alpha_w_2*(sin_theta_rlw-sin_theta)],
                          [0, 0, 1]], dtype=np.float64)
            
            A = const_AB*(sin_theta_rlw-sin_theta)-const_ABCD*cos_theta_rlw
            B = const_AB*(-cos_theta_rlw+cos_theta)-const_ABCD*sin_theta_rlw
            C = const_CD*(sin_theta_rlw-sin_theta)+const_ABCD*cos_theta_rlw
            D = const_CD*(-cos_theta_rlw+cos_theta)+const_ABCD*sin_theta_rlw

        V = np.array([[A, C],
                      [B, D],
                      [-1/w, 1/w]])
        

        N = len(ids)
        if N > 0:
            G = np.block([[G, np.zeros((3,2*N))], [np.zeros((2*N,3)), np.identity(2*N)]])
            # G = np.block(((G, np.zeros((3,2*N))), (np.zeros((2*N,3)), np.identity(2*N))))
            V = np.append(V, np.zeros((2*N,2)), axis=0)

        diag = np.diag(np.array([np.power(error_l,2), np.power(error_r,2)]))
        Sigma = np.dot(np.dot(G, Sigma), G.T) + np.dot(np.dot(V, diag), V.T)

        # faster computation for new covariance matrix
        # sigma_xx = self.Sigma[0:3, 0:3]
        # sigma_xm = self.Sigma[0:3, 3:]
        # sigma_mm = self.Sigma[3:, 3:]

        # Sigma_top_left = G@sigma_xx@G.T + V @ self.Sigma_u @ V.T  # final dim 3x3
        # Sigma_top_right = G@sigma_xm
        # Sigma_low_left = (G@sigma_xm).T

        # save updated position and covariance matrix
        # mu_prime = self.mu.copy()
        # mu_prime[0:3] = np.array([x_prime, y_prime, theta_prime])
        # self.mu_prime = mu_prime
        # self.Sigma_prime = np.block([[Sigma_top_left, Sigma_top_right],
        #                              [Sigma_low_left, sigma_mm]])
        mu[0], mu[1], mu[2] = x, y, theta

        return mu, Sigma

    def add_landmark(self, position: tuple, id: str):
        # measurement: tuple, id: str): # changed to a shorter signature of the function
        if int(id) <= 1000:
            x, y = position  # array with x and y

            # extend self.mu and self.Sigma with the new landmark
            self.mu = np.append(self.mu, [x, y])
            # self.mu_prime = np.append(self.mu_prime, [x, y])

            dim = self.Sigma.shape
            # dim_prime = self.Sigma_prime.shape
            Sigma = np.zeros(np.add(dim, 2))
            # Sigma_prime = np.zeros(np.add(dim_prime, 2))

            Sigma[:dim[0], :dim[1]] = self.Sigma
            Sigma[-2, -2] = self.SIGMA_SQUARED_X
            Sigma[-1, -1] = self.SIGMA_SQUARED_Y

            # Sigma_prime[:dim[0], :dim[1]] = self.Sigma_prime
            # Sigma_prime[-2, -2] = self.SIGMA_SQUARED_X
            # Sigma_prime[-1, -1] = self.SIGMA_SQUARED_Y

            self.Sigma = Sigma
            # self.Sigma_prime = Sigma_prime

            # add the landmark id to self.ids
            self.ids.append(id)
            print(f"Landmark with id {id} added")

    def correction(self, landmark_position_measured: tuple, id: int):
        # index of identified aruco
        index = self.ids.index(id)
        print("Index for correction:", index)

        # current world position of robot
        x, y, theta, _ = self.get_robot_pose()

        # the measured position fo the aruco from the camera
        r_i, beta_i = landmark_position_measured

        # the old position of the markers
        # mu = [0,0,0, id1, id1, id2, id2, ..., idn, idn]
        x_m, y_m = self.mu[3+2*index:3+2*index+2]

        # define the return values of h
        r = np.sqrt((x_m-x)**2+(y_m-y)**2)
        beta = self.subtract(np.arctan2((y_m-y), (x_m-x)), theta)

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
        m = max(3+2*len(self.ids)-n-5, 0)

        H = np.block([H_s_robot, np.zeros((2, n)),
                     H_s_marker, np.zeros((2, m))])
        H_s = np.block([H_s_robot, H_s_marker])

        inf = 3+2*index
        sup = 3+2*index+2

        sigma_s_top_left = self.Sigma[0:3, 0:3]
        sigma_s_low_right = self.Sigma[inf:sup, inf:sup]
        sigma_s_top_right = self.Sigma[0:3, inf:sup]
        sigma_s_low_left = self.Sigma[inf:sup, 0:3]

        sigma_s = np.block([[sigma_s_top_left, sigma_s_top_right],
                            [sigma_s_low_left, sigma_s_low_right]])

        Z = H_s@sigma_s@(H_s.T)+self.Q

        K = self.Sigma@((H.T)@(np.linalg.inv(Z)))
        
        self.mu = self.mu + \
            K@(np.array([r_i-r, self.subtract(beta_i, beta)]))
        self.Sigma = (np.identity(3+2*len(self.ids))-K@H)@self.Sigma 


    def get_robot_pose(self):
        # read out the robot position and angle from mu variable
        # read out robot error from Sigma
        robot_x, robot_y, robot_theta = self.mu[:3].copy()
        sigma = self.Sigma[:2, :2]
        error = self.get_error_ellipse(sigma)

        return robot_x, robot_y, robot_theta, error  # wrt world

    def get_landmark_poses(self):
        landmark_estimated_positions = []
        landmark_estimated_stdevs = []
        for i, id in enumerate(self.get_landmark_ids()):
            landmark_xy = self.mu[3+2*i: 3+2*i+2]
            sigma_xy = self.Sigma[3+2*i:3+2*i+2, 3+2*i:3+2*i+2]
            landmark_error = self.get_error_ellipse(sigma_xy)

            landmark_estimated_positions.append(landmark_xy)
            landmark_estimated_stdevs.append(landmark_error)

        return landmark_estimated_positions, landmark_estimated_stdevs
        # return [[0,0], [1,1]],[[0.5,0.5,0],[0.2,0.7,0]]

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
        eigvec_x, eigvec_y = eigen_vec[:, i]
        angle = np.arctan2(eigvec_y, eigvec_x)
        print(eigen_vals)
        return np.sqrt(np.real(eigen_vals[i])), np.sqrt(np.real(eigen_vals[j])), angle

    # return the list of all known ids
    def get_landmark_ids(self):
        return np.array(self.ids)

    # helper function to get correct difference of two angle
    def subtract(self, theta_1, theta_2):
        diff = (theta_1-theta_2) % (2*np.pi)
        if diff > np.pi:
            diff = diff-(2*np.pi)
        return diff

    def get_sigma(self):
        return self.Sigma