import cv2
import time
import numpy as np
import jsonpickle
from message import Message
from camera import Camera
from publisher import Publisher
from keypress_listener import KeypressListener

class Main():
    def __init__(self) -> None:

        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()
        self.camera = Camera()

        self.run()

    def run(self):
        # counter
        count = 0
        start = True
        while True:
            try:
                print("count:", count)
                # maybe we want to get keypresses in the terminal, for debugging
                self.parse_keypress()

                # get image from the camera
                _, raw_img = self.camera.read()
                
                # get the time
                time0 = time.time()

                # imaginary processing
                time.sleep(0.05)

                # create message
                msg = Message(
                    id = count,
                    timestamp = time0,
                    start = start,
                    landmark_ids = [],
                    landmark_rs = [],
                    landmark_alphas = [],
                    landmark_xs = [],
                    landmark_ys = [],
                    landmark_stdevs = [],
                    x = 0.0,
                    y = 0.0,
                    theta = 0.0,
                    stdev = [0.5, 0.5, 0.5]
                )
                    
                # pickle message
                msg_str = jsonpickle.encode(msg)

                # publish message and image
                self.publisher.publish_img(msg_str, raw_img)

                count += 1
                start = False

            except KeyboardInterrupt:
                # tidy up
                self.close()
                break
        
        # tidy up
        self.close()

    def close(self):
        self.keypress_listener.close()
        self.camera.close()

    def parse_keypress(self):
        char = self.keypress_listener.get_keypress()
        if char is not None:
            print("char:", char)

if __name__ == '__main__':
    main = Main()