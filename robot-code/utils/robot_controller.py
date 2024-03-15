from __future__ import annotations
from types import SimpleNamespace
import ev3_dc as ev3
import numpy as np
from rich import print

from utils.camera import Camera
from utils.vision import Vision
from utils.EKFSLAM import EKFSLAM
from utils.recorder import Recorder
from utils.robot_dummy import DummyVehicle

from timeit import default_timer as timer

from utils.tempo import *


class RobotController:
    def __init__(self, config) -> None:

        self.config = config
        self.dt = config.robot.delta_t

        self.__ev3_obj__ = None
        self.vehicle = None

        self.camera = None
        self.vision = None

        self.recorder = Recorder(self.dt)

        self.slam = EKFSLAM(
            config.robot.wheel_radius,
            config.ekf_slam.robot_width,
            MOTOR_STD=config.ekf_slam.motor_std,
            DIST_STD=config.ekf_slam.dist_std,
            ANGLE_STD=config.ekf_slam.angle_std
        )

        self.old_l, self.old_r = 0, 0

        self.past_ids=[]

        self.radius = self.config.robot.wheel_radius


    def __enter__(self) -> RobotController:

        self.camera = Camera(self.config.camera.exposure_time,
                             self.config.camera.gain)
        self.vision = Vision(self.camera.CAMERA_MATRIX, self.camera.DIST_COEFFS,
                             self.config.camera, self.config.robotgeometries)

        try:
            self.__ev3_obj__ = ev3.EV3(protocol=ev3.USB, sync_mode="STD")
        except Exception as e:
            print("error:", e)

        if self.__ev3_obj__:
            self.vehicle = ev3.TwoWheelVehicle(
                self.config.robot.wheel_radius,  # radius wheel
                self.config.robot.width,  # middle-to-middle tread measured
                speed=10,
                ev3_obj=self.__ev3_obj__
            )
            self.gyro = ev3.Gyro(
                port=ev3.PORT_3,
                ev3_obj=self.__ev3_obj__
            )

            print("[green]***CONNECTED TO REAL VEHICLE***[/green]")
        else:
            print("[red]***USING Dummy VEHICLE***[/red]")
            self.vehicle = DummyVehicle()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.recorder.is_recording:
            self.recorder.save_recording()

        if self.vehicle:
            self.vehicle.stop(brake=False)
            self.vehicle.__exit__(exc_type, exc_val, exc_tb)

        if self.__ev3_obj__:
            self.__ev3_obj__.__exit__(exc_type, exc_val, exc_tb)

        if self.camera:
            self.camera.close()
        
        # save a dump of all the knowledge of the robot
        self.slam.dump()

    @timeit
    def move(self, speed, turn, img=None):
        self.vehicle.move(speed, turn)
        self.recorder.save_step(img, speed, turn)

    # compute the l,r,theta_gyro for the ev3
    @timeit
    def get_motor_movement(self) -> tuple:
        motor_pose = self.vehicle.motor_pos

        alpha_l = np.deg2rad(motor_pose.left)
        alpha_r = np.deg2rad(motor_pose.right)

        l = alpha_l*self.radius
        r = alpha_r*self.radius
        theta_gyro = np.deg2rad(self.gyro.angle)

        return (l, r, theta_gyro)

    @timeit
    def run_ekf_slam(self, img, draw_img=None, fastmode=False):
        # movements is what is refered to as u = (l, r) in the document
        l, r, theta_gyro = self.get_motor_movement()
        movements = l - self.old_l, r - self.old_r
        self.old_l, self.old_r = l, r
        if movements[0] != 0.0 or movements[1] != 0.0:
            self.slam.predict(*movements)

        robot_pose=self.slam.get_robot_pose()
        # detect only aruco for faster computation
        ids, landmark_rs, landmark_alphas, landmark_positions = self.vision.detections(
            img, draw_img, robot_pose, kind='aruco')

        robot_x, robot_y, robot_theta, robot_stdev = robot_pose
        landmark_estimated_ids = self.slam.get_landmark_ids()

        # fastmode = only perform prediction step
        n_ids=len(ids)
        if not fastmode:
            # only perform on two thirds of detected landmarks
            limit_ids=2*n_ids//3
            for i, id in enumerate(ids[0:limit_ids]):
                if id in self.past_ids:
                    if id not in landmark_estimated_ids:
                        self.slam.add_landmark(
                            landmark_positions[i], id)
                    else:
                        # correct each detected landmark that is already added
                        self.slam.correction(
                            (landmark_rs[i], landmark_alphas[i]), id)
        elif fastmode: # fastmode true
            # only perform correction on max 3 landmarks
            if n_ids < 4:    
                limit_ids=n_ids
            else:
                limit_ids=3
            for i, id in enumerate(ids[0:limit_ids]):
                if id in self.past_ids:
                    if id in landmark_estimated_ids:
                        # correct each detected landmark that is already added
                        self.slam.correction(
                            (landmark_rs[i], landmark_alphas[i]), id)

            
        self.past_ids=ids

        landmark_estimated_positions, landmark_estimated_stdevs = self.slam.get_landmark_poses(fastmode)
    
        data = SimpleNamespace()
        data.landmark_ids = ids
        data.landmark_rs = landmark_rs
        data.landmark_alphas = landmark_alphas
        data.landmark_positions = landmark_positions

        data.landmark_estimated_ids = landmark_estimated_ids
        data.landmark_estimated_positions = landmark_estimated_positions
        data.landmark_estimated_stdevs = landmark_estimated_stdevs

        data.robot_position = np.array([robot_x, robot_y])
        data.robot_theta = robot_theta
        data.robot_stdev = robot_stdev
        data.theta_gyro = theta_gyro

        return data
