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

        self.detected_ids = set()


    def __enter__(self) -> RobotController:

        self.camera = Camera(self.config.camera.exposure_time,
                             self.config.camera.gain)
        self.vision = Vision(self.camera.CAMERA_MATRIX, self.camera.DIST_COEFFS,
                             self.config.camera)

        try:
            self.__ev3_obj__ = ev3.EV3(protocol=ev3.USB, sync_mode="STD")
        except Exception as e:
            print("error:", e)

        if self.__ev3_obj__:
            self.vehicle = ev3.TwoWheelVehicle(
                self.config.robot.wheel_radius, # radius wheel
                self.config.robot.width, # middle-to-middle tread measured
                speed = 10,
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

    def move(self, speed, turn, img=None):
        self.vehicle.move(speed, turn)
        self.recorder.save_step(img, speed, turn)

    def get_motor_movement(self) -> tuple:
        pass
        l = 0
        r = 0
        return (l, r)

    def run_ekf_slam(self, img, draw_img=None, fastmode=False):
        # movements is what is refered to as u = (l, r) in the document
        l,r = self.get_motor_movement()
        movements = l - self.old_l, r - self.old_r
        self.old_l, self.old_r = l, r
        if movements[0] != 0.0 or movements[1] != 0:
            self.slam.predict(*movements)

        ids, landmark_rs, landmark_alphas, landmark_positions = self.vision.detections(img, draw_img, self.slam.get_robot_pose())

        robot_x, robot_y, robot_theta, robot_stdev = self.slam.get_robot_pose()
        landmark_estimated_ids = self.slam.get_landmark_ids()
        landmark_estimated_positions, landmark_estimated_stdevs = self.slam.get_landmark_poses()

        
        for i, id in enumerate(ids):
            if id not in self.slam.get_landmark_ids():
                self.slam.add_landmark(landmark_positions[i], (landmark_rs[i], landmark_alphas[i]), id)
                print(f"Landmark with id {id} added")
            else:
                # correct each detected landmark that is already added
                self.slam.correction((landmark_rs[i], landmark_alphas[i]), id)

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

        return data
