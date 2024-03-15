# import the_maze_runner.MazeRunner as MazeRunner
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
# from the_maze_runner import *
from utils.the_maze_runner import MazeRunner

class Runner:
    def __init__(self, data, start, end):
        # self.robot = robot
        self.speed = 15
        self.turn = 0
        # TODO choose start and end in the_maze_runner
        print('Computing the path')
        self.path = MazeRunner(data).create_path(start, end) # in meters
        print('Path computed, ready to run the race')



    @timeit
    def run(self, data):
        # get current poistion and orientation
        robot_pose=data.robot_position
        robot_angle=data.robot_theta
        end_reached=False
        
        return self.speed, self.turn, end_reached
