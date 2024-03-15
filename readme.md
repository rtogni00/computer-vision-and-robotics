# Computer Vision Course - Robot Viewer

## Installation

Requires Python 3.9. Instsall the following packages:
```bash
pip install opencv-contrib-python Pillow numpy imagezmq jsonpickle pyqt5
```

## Running

In `subscriber.py` set the correct IP address of your robot.

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