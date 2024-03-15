from cycler import V
import ev3_dc as ev3
import time
from utils import camera
from utils import vision
import utils.utils
import numpy as np
import sys
from rich import print


class Wanderer:
    def __init__(self, robot):
        self.robot = robot
        self.side = 1
        self.speed = 40
        self.turn = 0
        self.speed_lost = 10
        self.speed_cruise = 40
        self.changed = False

    def choose_traj(self, img):
        # TODO In the future we can get all this info directly from ekf_slam
        ids, rho, alpha, coords = self.robot.vision.detections(
            img, None, [0, 0], kind='aruco')

        arrive_ids = (ids < 100) & (ids > 0)
        inner_ids = (ids % 3 == 0) & (ids < 1000) & (ids >= 100)  # left
        outer_ids = (ids % 3 == 2) & (ids < 1000) & (ids >= 100)  # right
        num_arrive_ids = np.sum(arrive_ids)
        num_inner_ids = np.sum(inner_ids)
        num_outer_ids = np.sum(outer_ids)
        print('OUTER IDS: ',num_outer_ids)
        # set starting side: right
        # drive according to right border markers
        # drive according to left border
        if self.side == 1:
            self.drive_border(num_outer_ids, outer_ids, rho, alpha, coords)
        elif self.side == -1:
            self.drive_border(num_inner_ids, inner_ids, rho, alpha, coords)

        # when we get to arrival switch to inner border
        if num_arrive_ids > 2 and self.changed == False:
            self.side = -1
            self.changed = True

    def drive_border(self, num_border, ids, rho, alpha, coords):
        if num_border >= 2:
            self.speed = self.speed_cruise
            rho_ids = rho[ids]
            alpha_ids = alpha[ids]

            x = coords[ids][:, 0]
            y = coords[ids][:, 1]

            # Closer aruco on the left
            # try:
            index = np.argsort(rho_ids)
            first = index[0]
            second = index[1]
            x_2 = x[second]
            x_1 = x[first]

            y_2 = y[second]
            y_1 = y[first]

            # if self.side == 0:
            #     if y_1 > 0:
            #         self.side = -1  # left
            #     else:
            #         self.side = 1  # right

            target_x = x_2 - x_1
            target_y = y_2 - y_1
            # print("Target x:", target_x)
            # print("Target y:", target_y)

            theta = np.rad2deg(np.arctan2(target_y, target_x))
            direction = 1 if theta >= 0 else -1
            print("Theta:", theta)
            if abs(theta) < 10:
                self.turn = 0
                # print("Going straight")
            elif abs(theta) < 30:
                self.turn = 10
                # print("Turning 40 degrees")
            elif abs(theta) < 45:
                self.turn = 20
                # print("Turning 80 degrees")
            else:
                self.turn = 80
                # print("Turning 120 degrees")

            if abs(rho_ids[first]*np.sin(alpha_ids[first])) < 0.13 and (rho_ids[first]*np.cos(alpha_ids[first])) < 0.4:
                self.turn = int(self.turn+5)*direction * self.side
                print(":warning:[bright_red]Too close")

            if rho_ids[first] > 0.35:
                self.turn = int(self.turn * direction/2)
                print(":warning:[bright_red]Far arucos")
            else:
                self.turn = int(self.turn*direction)
        else:
            print(':warning:[bright_red]I can\'t see a thing')
            self.turn = -200*self.side
            self.speed = self.speed_lost

    def tramp(self, img):
        self.choose_traj(img)
        # issue the movement command
        # self.robot.move(self.speed, self.turn)
        return self.speed, self.turn
