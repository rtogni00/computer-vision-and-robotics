# import the_maze_runner.MazeRunner as MazeRunner
import ev3_dc as ev3
import time
from timeit import default_timer as timer
import numpy as np
import sys
from rich import print
from utils.tempo import *
# from the_maze_runner import *
from utils.the_maze_runner import MazeRunner
import math
from utils.pid_controller import PIDController


class Runner:
    def __init__(self, data, start, end):
        # self.robot = robot
        self.speed = 20
        self.turn = 0
        # TODO choose start and end in the_maze_runner
        print('Computing the path')
        self.path = MazeRunner(data).create_path(start, end)  # in meters
        np.savetxt('path_coords.txt', self.path)
        for p in self.path:
            print(p)
        np.savetxt('robot_pose.txt', data.robot_position)
        # np.savetxt("robot_angle.txt", np.array(data.robot_theta))
        print('Path computed, ready to run the race')
        # kp_initial, ki_initial and kd_initial are hyperparameters to be tuned
        self.pid_turn = PIDController(80, 0, 0.0)
        # self.pid_turn

    def compute_desired_direction(self, current_pose, target_coordinate):
        # Calculate the direction vector from the current pose to the target coordinate
        dx = target_coordinate[0] - current_pose[0]
        dy = target_coordinate[1] - current_pose[1]
        desired_direction = math.atan2(dy, dx)
        return desired_direction

    # def compute_actual_direction(self, current_pose):
    #     # Assuming the robot's orientation is given by the third element of the pose tuple
    #     return current_pose[2]

    def compute_error_angle(self, desired_direction, actual_direction):

        desired_direction = (desired_direction + np.pi) % (2 * np.pi) - np.pi
        actual_direction = (actual_direction + np.pi) % (2 * np.pi) - np.pi
        # Calculate the error angle between the desired and actual directions
        error_angle = desired_direction - actual_direction
        # Normalize the error angle to be within [-pi, pi]
        error_angle = (error_angle + np.pi) % (2 * np.pi) - np.pi
        return error_angle

    def reached_target_coordinate(self, target, robot_pose):
        """Check if the robot has reached its target"""
        print("DISTANCE TO TARGET:", np.sqrt(np.sum((target-robot_pose)**2)))
        # if closer than 5 cm form the checkpoint
        if np.sqrt(np.sum((target-robot_pose)**2)) < 0.10:
            return True
        else:
            return False

    @timeit
    def run(self, data, dt):
        # get current poistion and orientation
        robot_pose = data.robot_position
        robot_angle = data.robot_theta

        desired_direction = self.compute_desired_direction(
            robot_pose, self.path[0])
        # self.compute_actual_direction(robot_pose)
        actual_direction = robot_angle

        error_angle = self.compute_error_angle(
            desired_direction, actual_direction)

        # correction =
        # check if it + or - correction
        self.turn = int(self.pid_turn.update(error_angle, dt=0))

        if self.reached_target_coordinate(self.path[0], robot_pose):
            print('Removed point, going to next')
            self.pid_turn.reset()
            # self.turn = 0
            if(len(self.path) > 0):
                self.path.pop(0)
            else:
                print("End reached!")
                return self.speed, self.turn, True


        print("*"*100)
        print(robot_pose, self.path[0],)
        print("---->", robot_angle, desired_direction, error_angle)
        print(self.turn, )
        print("*"*100)
        return self.speed, self.turn, False
