# Computer Vision Course - Robot Viewer

## Installation

Requires Python 3.9. Instsall the following packages:
```bash
pip install opencv-contrib-python Pillow numpy imagezmq jsonpickle pyqt5
```

## Running

In `subscriber.py` set the correct IP address of your robot.

```python
self.image_hub = imagezmq.ImageHub(open_port='tcp://100.64.0.3:5555', REQ_REP=False)
```

In `viewer.py` you can change the size of your world by editing these parameters:
```python
self.world_extents = (8.0, 8.0) # world extents, in meters
self.max_scanner_range = 2.0 # range of camera, in meters
```

Run the viewer with:
```bash
python viewer.py
```

## Running the Robot Code

Copy the folder `robot-code` to your raspberry pi.

Run with:
```bash
python main.py
```

## Message Format

The message should be of the following format, where m is the number of landmarks measured by the camera, and n the number of landmarks estimated by SLAM:

- landmark_ids: shape (m,)
- landmark_rs: shape (m,) - distance to landmark
- landmark_alphas: shape (m,) - angle to landmark
- landmark_positions: shape (m, 2) - list of (x, y)
- landmark_estimated_ids: shape (n,)
- landmark_estimated_positions: shape (n, 2) - list of (x, y)
- landmark_estimated_stdevs: shape (n, 3) - list of (sqrt(lambda1), sqrt(lambda2), alpha)
- robot_position: shape (2,) - (x, y)
- robot_theta: float
- robot_stdev: shape (3,) - list of (sqrt(lambda1), sqrt(lambda2), alpha)
