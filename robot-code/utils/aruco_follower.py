from cycler import V
import ev3_dc as ev3
import time
from timeit import default_timer as timer
from utils import camera
from utils import vision
import utils.utils
import numpy as np
import sys
from rich import print
from utils.tempo import *

class Wanderer:
    def __init__(self, config):
        self.speed = None
        self.turn = 0
        self.speed_lost = config.wanderer.speed_lost
        self.speed_cruise = config.wanderer.speed_cruise
        self.which_side = 'inner'  # by default start following the inner perimeter
        self.end_reached=0

    @timeit
    def choose_traj(self, data):
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

            # arucos too close
            if abs(rho_ids[first]*np.sin(alpha_ids[first])) < 0.13 and (rho_ids[first]*np.cos(alpha_ids[first])) < 0.4:
                self.turn = int(self.turn+10)*direction * side

            if rho_ids[first] > 0.35: # far arucos
                self.turn = int(self.turn * direction/2)
            else:
                self.turn = int(self.turn*direction)

        else: # does not see anything
            self.turn = -150*side
            self.speed = self.speed_lost

    @timeit
    def tramp(self, data):
        robot_pose = data.robot_position
        landmark_esitmated_ids = np.array(data.landmark_estimated_ids)
        landmark_estimated_positions = np.array(
            data.landmark_estimated_positions)
        # mask on arrival arucos
        mask_arrival_ids = (landmark_esitmated_ids < 100) & (
            landmark_esitmated_ids > 0)
        # if at leat 2 arrival arucos were seen
        if np.sum(mask_arrival_ids) > 1:
            pos_arrival_ids = landmark_estimated_positions[mask_arrival_ids]
            finish_line = np.mean(pos_arrival_ids, axis=0)
            dist_from_arrival = np.sqrt(np.sum((finish_line-robot_pose)**2))

            # if closer than 0.6m then switch side or stop
            if dist_from_arrival < 0.5:
                if self.which_side == 'inner':
                    print('Switch to outer side')
                    self.which_side = 'outer'
                    # self.time0 = timer()
                elif data.theta_gyro <- 7/2*np.pi:
                    self.end_reached+=1
                #timer()-self.time0 > self.grace_time:
                    print('.'*self.end_reached)
                    return 0, 0, self.end_reached

        self.choose_traj(data)

        return self.speed, self.turn, False
