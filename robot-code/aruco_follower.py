from cycler import V
import ev3_dc as ev3
import time
from utils import camera
from utils import vision
import utils.utils
import numpy as np
import sys


config = utils.utils.load_config("config.yaml")
cam = camera.Camera(config.camera.exposure_time,
                    config.camera.gain)
vis = vision.Vision(cam.CAMERA_MATRIX, cam.DIST_COEFFS,
                    config.camera, config.geometries)

# _, raw_img, cam_fps, img_created = cam.read()

# time.sleep(0.1)


def findRadius(x1, y1, x2, y2, x3, y3):
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
                              ((y31) * (x12) - (y21) * (x13))))

    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
         (2 * ((x31) * (y12) - (x21) * (y13))))

    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1)

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c

    # r is the radius
    r = np.sqrt(sqr_of_r)
    return r

time.sleep(0.1)
print('Starting')

with ev3.TwoWheelVehicle(
    0.028,  # radius_wheel
    0.137,  # tread
    protocol=ev3.USB
) as vehicle:
    while True:
        try:
            _, raw_img, cam_fps, img_created = cam.read()
            ids, rho, alpha, coords = vis.detections(raw_img, None, [0, 0], kind='aruco')
            coefficient = 0.2

            # print(ids)
            index = np.argsort(rho)
            for i, ids in enumerate(ids[index]):
                print('id:', ids, coords[i][0], 'y:', coords[i][1])
            first = index[0]
            second = index[1]
            third = index[1]

            # TODO find 3 closest points
            x_first = coords[first][1]
            y_first = coords[first][0]

            x_second = coords[second][1]
            y_second = coords[second][0]

            x_third = coords[third][1]
            y_third = coords[third][0]

            # TODO find coords of middle point

            x_m = np.mean([x_first, x_second, x_third])
            y_m = np.mean([y_first, y_second, y_third])

            # TODO find radius of circumference passing through middle point, last point and robot and make robot move by angle
            r = findRadius(0, 0, x_m, y_m, x_third, y_third)
            distance = np.sqrt(x_m**2+y_m**2)
            # angle btw robot and last point
            beta = np.rad2deg(2*np.arcsin(distance/(2*r)))
            print(r, beta)
            if y_m<0:
                rotation=1
            else:
                rotation=-1
            if vehicle.busy:
                vehicle.stop()
            if r < 5:
                vehicle.drive_turn(rotation*beta*coefficient, r).start(thread=False)
            else:
                vehicle.drive_straight(distance*coefficient).start(thread=False)

                # idx_i = np.argmin(rho)
                # distance=rho[idx_i]
                # heading=np.rad2deg(alpha[idx_i])

            time.sleep(1)
        except KeyboardInterrupt:
            print('\n' + '**** lost connection ****')
            vehicle.stop()
            sys.exit()



class Wanderer:
    def __init__(self,robot):
        self.robot=robot
        self.speed = 0
        self.turn = 0
        self.config = utils.utils.load_config("config.yaml")
        self.cam = camera.Camera(self.config.camera.exposure_time,
                    self.config.camera.gain)
        self.vis = vision.Vision(self.cam.CAMERA_MATRIX, self.cam.DIST_COEFFS,
                    self.config.camera, self.config.geometries)
        

    def choose_traj(self, img):
        ids, rho, alpha, coords = vis.detections(img, None, [0, 0], kind='aruco') # In the future we can get all this info directly from ekf_slam
        inner_ids=(ids%3==0)& (ids<1000) # left
        outer_ids=(ids%3==2)& (ids<1000) # right

        x_inner=coords[inner_ids][0]
        y_inner=coords[inner_ids][1]

        x_outer=coords[outer_ids][0]
        y_outer=coords[outer_ids][1]
        
        # Closer aruco on the left

        



        # Closer aruco on the right

    def lost_wanderer(self):
        self.robot.move(self.speed, self.turn)

    def tramp(self, img):
        # analyze the image to get aruco positions

        # choose the movement
        
        # keep track of the movement status

        # issue the movement command
        self.robot.move(self.speed, self.turn)