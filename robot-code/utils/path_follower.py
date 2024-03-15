import numpy as np
from rich import print
from utils.tempo import *
from utils.the_maze_runner import MazeRunner
from utils.pid_controller import PIDController
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, data, start, end, config):
        self.speed = config.runner.speed

        # load parameters for PID controller
        kp = config.pid.kp
        ki = config.pid.ki
        kd = config.pid.kd
        self.turn = 0
        self.removal_distance = config.pid.removal_distance  # 0.25 m

        # start the computationof the path
        print('Computing the path')
        self.path = MazeRunner(data, config).create_path(
            start, end)  # in meters

        # save the image, but takes some seconds to complete
        if config.maze.save_fig:
            self.save_path_info(data)

        print('Path computed, ready to run the race')

        self.pid_turn = PIDController(kp, ki, kd)

    # Calculate the direction vector from the current pose to the target coordinate
    @timeit
    def compute_desired_direction(self, current_pose, target_coordinate):
        dx = target_coordinate[0] - current_pose[0]
        dy = target_coordinate[1] - current_pose[1]
        desired_direction = np.arctan2(dy, dx)
        return desired_direction

    # calculate the angle between the current orientation and the desired one
    @timeit
    def compute_error_angle(self, desired_direction, actual_direction):

        desired_direction = (desired_direction + np.pi) % (2 * np.pi) - np.pi
        actual_direction = (actual_direction + np.pi) % (2 * np.pi) - np.pi
        # Calculate the error angle between the desired and actual directions
        error_angle = desired_direction - actual_direction
        # Normalize the error angle to be within [-pi, pi]
        error_angle = (error_angle + np.pi) % (2 * np.pi) - np.pi
        return error_angle

    # Check if the robot has reached its target
    @timeit
    def reached_target_coordinate(self, target, robot_pose):
        print("DISTANCE TO TARGET:", np.sqrt(np.sum((target-robot_pose)**2)))
        # if closer than 5 cm form the checkpoint
        if np.sqrt(np.sum((target-robot_pose)**2)) < self.removal_distance:
            return True
        else:
            return False

    # actually compute what to do at every cycle
    @timeit
    def run_path(self, data, dt):
        # get current position and orientation
        robot_pose = data.robot_position
        robot_angle = data.robot_theta
        if len(self.path) == 0:
            print("End reached")
            return 0, 0, True

        desired_direction = self.compute_desired_direction(
            robot_pose, self.path[0])
        actual_direction = robot_angle

        error_angle = self.compute_error_angle(
            desired_direction, actual_direction)

        self.turn = int(self.pid_turn.update(error_angle, dt))

        if self.reached_target_coordinate(self.path[0], robot_pose):
            print('Removed point, going to next')
            if (len(self.path) > 0):
                self.path.pop(0)

        print(f"---->\nrobot angle: {robot_angle}\ndesired direction: {desired_direction}\nerror angle: {error_angle}\nturn: {self.turn}")

        return self.speed, self.turn, False

    # save map and computed path
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
        plt.scatter(*robot_pose, color='orange', label='robot pose')
        plt.gca().set_aspect('equal')
        plt.legend(bbox_to_anchor=(1.05,  1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig('path.png')
        print('Figure saved')