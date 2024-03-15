from cycler import V
import ev3_dc as ev3
import time
from utils import camera
from utils import vision
import utils.utils
import numpy as np
import sys

class Wanderer:
    def __init__(self, robot):
        self.robot = robot
        self.speed = 25
        self.turn = 0

    def choose_traj(self, img):
        # TODO In the future we can get all this info directly from ekf_slam
        ids, rho, alpha, coords = self.robot.vision.detections(
            img, None, [0, 0], kind='aruco')
        # print(ids, coords)
        # inner_ids=(ids%3==0)& (ids<1000) # left
        outer_ids = (ids % 3 == 2) & (ids < 1000)  # right
        # print("Outer ids:", outer_ids)
        # rho_inner_ids=rho[inner_ids]

        if len(outer_ids) >= 2:
            rho_outer_ids = rho[outer_ids]
            alpha_outer_ids=alpha[outer_ids]
            # x_inner=coords[inner_ids][0]
            # y_inner=coords[inner_ids][1]

            x_outer = coords[outer_ids][:, 0]
            y_outer = coords[outer_ids][:, 1]

            # Closer aruco on the left
            # try:
            index_outer = np.argsort(rho_outer_ids)
            # print(ids[index_outer])
            first_outer = index_outer[0]
            second_outer = index_outer[1]
            # print("First and second outer:", ids[first_outer], ids[second_outer])
            x_2_outer = x_outer[second_outer]
            x_1_outer = x_outer[first_outer]

            y_2_outer = y_outer[second_outer]
            y_1_outer = y_outer[first_outer]

            target_x = x_2_outer - x_1_outer
            target_y = y_2_outer - y_1_outer
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

            if rho_outer_ids[first_outer] > 0.35:
                self.turn = int(self.turn * direction/2)
                print("[green]I see arucos far away! Correcting direction[/green]")
            else:
                self.turn = int(self.turn*direction)

            if abs(rho_outer_ids[first_outer]*np.sin(alpha_outer_ids[first_outer])) < 0.13 and (rho_outer_ids[first_outer]*np.cos(alpha_outer_ids[first_outer])) < 0.4:
                self.turn = int(self.turn+5)*direction
                print("[green]Too close!!! Correcting direction[/green]")

        else:
            print('I can\'t see a thing')
            self.turn = 200

            # Closer aruco on the right

    def tramp(self, img):
        self.choose_traj(img)
        # issue the movement command
        # self.robot.move(self.speed, self.turn)
        return self.speed, self.turn
