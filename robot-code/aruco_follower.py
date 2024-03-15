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
        self.speed = 20
        self.turn = 0
        self.speed_lost = 10
        self.speed_cruise = 20
        self.which_side = 'outer'  # by default start following the outer perimeter

    def choose_traj_outer(self, data):
        ids, rho, alpha = data.landmark_ids, data.landmark_rs, data.landmark_alphas
        side = 1

        if self.which_side == 'outer':
            ids = (ids % 3 == 2) & (ids < 1000) & (ids > 100)  # outer ids
            side = 1
        else:
            ids = (ids % 3 == 0) & (ids < 1000) & (
                ids > 100)  # inner ids follow
            side = -1

        if np.sum(ids) >= 2:
            self.speed = self.speed_cruise
            rho_ids = rho[ids]
            alpha_ids = alpha[ids]

            x = rho * np.cos(alpha)
            y = rho * np.sin(alpha)

            # Closer aruco on the left
            index = np.argsort(rho_ids)
            first = index[0]
            second = index[1]
            x_2 = x[second]
            x_1 = x[first]

            y_2 = y[second]
            y_1 = y[first]

            target_x = x_2 - x_1
            target_y = y_2 - y_1

            theta = np.rad2deg(np.arctan2(target_y, target_x))
            direction = 1 if theta >= 0 else -1
            if abs(theta) < 10:
                self.turn = 0
            elif abs(theta) < 30:
                self.turn = 10
            elif abs(theta) < 45:
                self.turn = 20
            else:
                self.turn = 80

            if abs(rho_ids[first]*np.sin(alpha_ids[first])) < 0.13 and (rho_ids[first]*np.cos(alpha_ids[first])) < 0.4:
                self.turn = int(self.turn+5)*direction * side
                print(":warning:[bright_red]Too close")

            if rho_ids[first] > 0.35:
                self.turn = int(self.turn * direction/2)
                print(":warning:[bright_red]Far arucos")
            else:
                self.turn = int(self.turn*direction)

        else:
            print(':warning:[blue]I can\'t see a thing')
            self.turn = -150*side
            self.speed = self.speed_lost

    def tramp(self, data):
        ids_seen = np.array(data.landmark_estimated_ids)
        landmark_estimated_positions = np.array(data.landmark_estimated_positions)
        robot_pose = data.robot_position
        # i have seen the finish line in the past
        finish_line_ids_mask = (ids_seen < 100) & (ids_seen > 0)
        if np.sum(finish_line_ids_mask) >= 1:
            if self.which_side == 'outer':
                # ids_finish_line = ids_seen[finish_line_ids_mask]
                pos_1 = landmark_estimated_positions[finish_line_ids_mask][0]
                if distance(robot_pose, pos_1) < 0.4:
                    self.which_side = 'inner'

        # if self.which_side=='outer':
        self.choose_traj_outer(data)
        # else:
        # self.choose_traj_inner(data)
        # pass

        return self.speed, self.turn


def distance(pos_1, pos_2):
    return np.sqrt((pos_1[0]-pos_2[0])**2 + (pos_1[1]-pos_2[1])**2)
