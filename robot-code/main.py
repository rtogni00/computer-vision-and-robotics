import cv2
import time
import numpy as np
import sys
import jsonpickle
import pickle
from message import Message
from timeit import default_timer as timer
from numba import typed

from utils.robot_controller import RobotController

from publisher import Publisher
from utils.keypress_listener import KeypressListener
from rich import print
from utils.utils import load_config



from enum import Enum
class TaskPart(Enum):
    Manual = 0
    Exploration = 1
    ToStartLine = 2
    Race = 3
    Load = 4


class Main():
    def __init__(self) -> None:

        self.config = load_config("config.yaml")

        self.robot = RobotController(self.config)
        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()

        self.DT = self.config.robot.delta_t # delta time in seconds

        self.speed = 0
        self.turn = 0
        self.new_speed = 0
        self.new_turn = 0

        self.manualMode = False
        self.is_running = True

        self.map = None

        self.mode = TaskPart.Manual

        self.run_loop()

    def run_loop(self):
        print("starting...")

        # control vehicle movement and visualize it
        with self.keypress_listener, self.publisher, self.robot:
            print("starting EKF SLAM...")

            print("READY!")
            print("[green]MODE: Manual")

            count = 0

            while self.is_running:
                time0 = timer()
                self.run(count, time0)

                elapsed_time = timer() - time0
                if elapsed_time <= self.DT:
                    dt = self.DT - elapsed_time
                    time.sleep(dt) # moves while sleeping
                else:
                    print(f"[red]Warning! dt = {elapsed_time}")

                count += 1

            print("*** END PROGRAM ***")

    def run(self, count, time0):


        if not self.robot.recorder.playback:
            # read webcam and get distance from aruco markers
            _, raw_img, cam_fps, img_created = self.robot.camera.read() # BGR color

            speed = self.speed
            turn = self.turn
        else:
            cam_fps = 0
            raw_img, speed, turn = next(self.robot.recorder.get_step)

        if raw_img is None:
            print("[red]image is None!")
            return

        if self.mode == TaskPart.Race:
            draw_img = raw_img
            data = self.robot.run_ekf_slam(raw_img, fastmode = True)
        else:
            draw_img = raw_img.copy()
            data = self.robot.run_ekf_slam(raw_img, draw_img)

        self.parse_keypress()

        if self.mode == TaskPart.Manual:
            self.robot.move(self.speed, self.turn)

        if self.mode == TaskPart.Exploration:
           pass

        if self.mode == TaskPart.ToStartLine:
            pass

        if self.mode == TaskPart.Race:
            pass

        if self.mode == TaskPart.Load:
            pass

        msg = Message(
            id = count,
            timestamp = time0,
            start = True,

            landmark_ids = data.landmark_ids,
            landmark_rs = data.landmark_rs,
            landmark_alphas = data.landmark_alphas,
            landmark_positions = data.landmark_positions,

            landmark_estimated_ids = data.landmark_estimated_ids,
            landmark_estimated_positions = data.landmark_estimated_positions,
            landmark_estimated_stdevs = data.landmark_estimated_stdevs,

            robot_position = data.robot_position,
            robot_theta = data.robot_theta,
            robot_stdev = data.robot_stdev,

            text = f"cam fps: {cam_fps}"
        )

        msg_str = jsonpickle.encode(msg)
        self.publisher.publish_img(msg_str, draw_img)


    def save_state(self, data):
        with open("SLAM.pickle", 'wb') as pickle_file:
            pass
        
        pass

    def load_and_localize(self):
        with open("SLAM.pickle", 'rb') as f:
            pass

        pass

    def parse_keypress(self):
        char = self.keypress_listener.get_keypress()

        turn_step = 40
        speed_step = 5

        if char == "a":
            if self.turn >= 0:
                self.new_turn = self.turn + turn_step
            else:
                self.new_turn = 0
            self.new_turn = min(self.new_turn, 200)
        elif char == "d":
            if self.turn <= 0:
                self.new_turn = self.turn - turn_step
            else:
                self.new_turn = 0
            self.new_turn = max(self.new_turn, -200)
        elif char == "w":
            if self.speed >= 0:
                self.new_speed = self.speed + speed_step
            else:
                self.new_speed = 0
            self.new_speed = min(self.new_speed, 100)
        elif char == "s":
            if self.speed <= 0:
                self.new_speed = self.speed - speed_step
            else:
                self.new_speed = 0
            self.new_speed = max(self.new_speed, -100)
        elif char == "c":
            self.new_speed = 0
            self.new_turn = 0
        elif char == "q":
            self.new_speed = 0
            self.new_turn = 0
            self.is_running = False
        elif char == "m":
            self.new_speed = 0
            self.new_turn = 0
            self.mode = TaskPart.Manual
            print("[green]MODE: Manual")
        elif char == "r":
            self.mode = TaskPart.Race
            print("[green]MODE: Race")
        elif char == "l":
            self.mode = TaskPart.Load
            print("[green]MODE: Load map")
        elif char == "p":
            self.mode = TaskPart.ToStartLine
            print("[green]MODE: To start line")
        elif char == "e":
            self.mode = TaskPart.Exploration
            print("[green]MODE: Exploration")

        if self.speed != self.new_speed or self.turn != self.new_turn:
            self.speed = self.new_speed
            self.turn = self.new_turn
            print("speed:", self.speed, "turn:", self.turn)


if __name__ == '__main__':

    main = Main()

