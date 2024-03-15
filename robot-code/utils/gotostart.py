from the_maze_runner import MazeRunner
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
from magic import ordinator
# import math


class GoToStart:
    def __init__(self, data):
        self.positions = data.landmark_estimated_positions
        self.ids = data.landmark_estimated_ids
        self.target = None

    def find_extremes(self):
        positions_array = np.array(self.positions)
        mask_start_line = (self.ids < 100) & (self.ids > 0)
        positions_start_line = positions_array[mask_start_line]
        pos_start_line_ordered = np.array(
            ordinator(positions_start_line, max_distance=0.3))
        first = pos_start_line_ordered[0]
        last = pos_start_line_ordered[-1]

        radius = np.sqrt((first[0]-last[0])**2+(last[1]-first[1])**2)

        return (first, last, radius)

    def get_centers(self, first, last, r):
        x1, y1 = first
        x2, y2 = last
        r1 = r2 = r
        centerdx = x1 - x2
        centerdy = y1 - y2
        R = np.sqrt(centerdx**2 + centerdy**2)
        if not (abs(r1 - r2) <= R and R <= r1 + r2):
            """ No intersections """
            return []

        """ intersection(s) should exist """
        R2 = R**2
        R4 = R2**2
        a = (r1**2 - r2**2) / (2 * R2)
        r2r2 = r1**2 - r2**2
        c = np.sqrt(2 * (r1**2 + r2**2) / R2 - (r2r2**2) / R4 - 1)

        fx = (x1 + x2) / 2 + a * (x2 - x1)
        gx = c * (y2 - y1) / 2
        ix1 = fx + gx
        ix2 = fx - gx

        fy = (y1 + y2) / 2 + a * (y2 - y1)
        gy = c * (x1 - x2) / 2
        iy1 = fy + gy
        iy2 = fy - gy

        return [np.array([ix1, iy1]), np.array([ix2, iy2])]

    def get_arrival_coords(self):
        intersect_1, intersect_2 = self.get_centers(self.find_extremes())
        d1 = np.sqrt(np.sum((self.robot_pose-intersect_1)**2))
        d2 = np.sqrt(np.sum((self.robot_pose-intersect_2)**2))

        if d1 < d2:
            self.target = intersect_1
        else:
            self.target = intersect_2

    def run(self, data, vehicle):
        # get current position and orientation
        robot_pose = data.robot_position
        robot_angle = data.robot_theta
        start_line_reached = False

        distance_to_target = np.sqrt(np.sum((self.robot_pose-self.target)**2))
        target_angle = np.arccos((self.target[1]-robot_pose[1])/distance_to_target)

        while(target_angle - robot_angle < 0.1):
            self.move = -200

