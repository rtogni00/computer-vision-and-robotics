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
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, data, start, end, config):
        # self.robot = robot
        self.speed = config.runner.speed
        kp = config.pid.kp
        ki = config.pid.ki
        kd = config.pid.kd
        self.turn = 0
        self.removal_distance = config.pid.removal_distance  # 0.25 m
        # TODO choose start and end in the_maze_runner
        print('Computing the path')
        self.path = MazeRunner(data, config).create_path(start, end)  # in meters
        # np.savetxt('path_coords.txt', self.path)
        # np.savetxt('robot_pose.txt', data.robot_position)
        if config.maze.save_fig:
            self.save_path_info(data)
        # np.savetxt("robot_angle.txt", np.array(data.robot_theta))
        print('Path computed, ready to run the race')
        # kp_initial, ki_initial and kd_initial are hyperparameters to be tuned
        self.pid_turn = PIDController(kp, ki, kd)

    @timeit
    def compute_desired_direction(self, current_pose, target_coordinate):
        # Calculate the direction vector from the current pose to the target coordinate
        dx = target_coordinate[0] - current_pose[0]
        dy = target_coordinate[1] - current_pose[1]
        desired_direction = math.atan2(dy, dx)
        return desired_direction

    @timeit
    def compute_error_angle(self, desired_direction, actual_direction):

        desired_direction = (desired_direction + np.pi) % (2 * np.pi) - np.pi
        actual_direction = (actual_direction + np.pi) % (2 * np.pi) - np.pi
        # Calculate the error angle between the desired and actual directions
        error_angle = desired_direction - actual_direction
        # Normalize the error angle to be within [-pi, pi]
        error_angle = (error_angle + np.pi) % (2 * np.pi) - np.pi
        return error_angle

    @timeit
    def reached_target_coordinate(self, target, robot_pose):
        """Check if the robot has reached its target"""
        print("DISTANCE TO TARGET:", np.sqrt(np.sum((target-robot_pose)**2)))
        # if closer than 5 cm form the checkpoint
        if np.sqrt(np.sum((target-robot_pose)**2)) < self.removal_distance:
            return True
        else:
            return False

    @timeit
    def run_path(self, data, dt):
        # get current poistion and orientation
        robot_pose = data.robot_position
        robot_angle = data.robot_theta
        if len(self.path) == 0:
            print("End reached")
            return 0, 0, True

        desired_direction = self.compute_desired_direction(
            robot_pose, self.path[0])
        # self.compute_actual_direction(robot_pose)
        actual_direction = robot_angle

        error_angle = self.compute_error_angle(
            desired_direction, actual_direction)

        # correction =
        # check if it + or - correction
        self.turn = int(self.pid_turn.update(error_angle, dt))

        if self.reached_target_coordinate(self.path[0], robot_pose):
            print('Removed point, going to next')
            # self.pid_turn.reset()
            if (len(self.path) > 0):
                self.path.pop(0)

        # print("*"*10)
        # print(robot_pose, self.path[0],)
        print("---->", robot_angle, desired_direction, error_angle)
        # print(self.turn, )
        # print("*"*10)
        return self.speed, self.turn, False

    def save_path_info(self, data):
        ids = data.landmark_estimated_ids
        positions = data.landmark_estimated_positions
        robot_pose = data.robot_position
        for pos_zip in zip(positions, ids):
            col = pos_zip[1] % 3
            if col == 0:
                col2 = 'green'
            elif col == 1:
                col2 = 'red'
            elif col == 2:
                col2 = 'blue'
            if pos_zip[1] < 100:
                col2 = 'black'
            plt.scatter(*pos_zip[0], color=col2)

        for point in self.path:
            plt.scatter(*point, color="purple")

        plt.scatter(*self.path[0], c='red', label='start path')
        plt.scatter(*self.path[-1], c='blue', label='end end')
        plt.scatter(*self.path[3], c='coral', label='intermediate path')
        # plt.plot(self.path[:,0], self.path[:,1], c='purple')
        plt.scatter(*robot_pose, color='orange', label='robot pose')
        plt.gca().set_aspect('equal')
        plt.legend(bbox_to_anchor=(1.05,  1.0), loc='upper left')
        plt.tight_layout()
        # plt.legend()
        plt.savefig('path.png')
        print('Fig saved')
        # plt.show()
