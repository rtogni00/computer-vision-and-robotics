import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
from a_star import *
from magic import ordinator
from scipy import ndimage


class MazeRunner():
    def __init__(self, data):
        self.spatial_step = 0.03
        # get data from data
        self.positions = data.positions
        self.ids = data.ids
        self.robot_pose = data.robot_pose
        self.path = []
        self.n_x = 0
        self.n_y = 0
        self.start = ()
        self.end = ()

    def matrix_to_meter(self):
        pass

    def meter_to_matrix(self):
        pass

    def create_mask(self):
        # distinguish internal external obstable and start markers
        positions_array = np.array(self.positions)

        mask_external = (self.ids % 3 == 2) & (
            self.ids >= 100) & (self.ids <= 1000)
        mask_internal = (self.ids % 3 == 0) & (
            self.ids >= 100) & (self.ids <= 1000)
        mask_obstacle = (self.ids % 3 == 1) & (
            self.ids >= 100) & (self.ids <= 1000)
        mask_start_line = (self.ids < 100)

        # ids_external = ids[mask_external]
        # ids_internal = ids[mask_internal]

        positions_external = positions_array[mask_external]
        positions_internal = positions_array[mask_internal]
        positions_obstacle = positions_array[mask_obstacle]
        positions_start_line = positions_array[mask_start_line]

        pos_list_ext = [pos for pos in positions_external]
        pos_list_int = [pos for pos in positions_internal]

        external_ordered = np.array(ordinator(pos_list_ext, max_distance=0.6))
        internal_ordered = np.array(ordinator(pos_list_int))

        max_x = np.max(external_ordered[:, 0])
        max_y = np.max(external_ordered[:, 1])
        min_x = np.min(external_ordered[:, 0])
        min_y = np.min(external_ordered[:, 1])
        delta_x = max_x-min_x
        delta_y = max_y-min_y
        # center_x = min_x+delta_x/2
        # center_y = min_y+delta_y/2

        self.n_x = int(delta_x/self.spatial_step/1.5)
        self.n_y = int(delta_y/self.spatial_step/1.5)

        ext_ordered_poly = Polygon(external_ordered)
        int_ordered_poly = Polygon(internal_ordered)

        x = np.linspace(min(np.concatenate([external_ordered[:, 0], internal_ordered[:, 0]])),
                        max(np.concatenate(
                            [external_ordered[:, 0], internal_ordered[:, 0]])),
                        self.n_x)
        y = np.linspace(min(np.concatenate([external_ordered[:, 1], internal_ordered[:, 1]])),
                        max(np.concatenate(
                            [external_ordered[:, 1], internal_ordered[:, 1]])),
                        self.n_y)

        # X, Y = np.meshgrid(x, y)

        # Create a mask where the pixels inside the first circle and outside the second circle are set to 1, and all others are set to 0
        mask_internal = np.array([[int_ordered_poly.contains(Point(x[i], y[j]))
                                   for i in range(self.n_x)]
                                  for j in range(self.n_y)])
        mask_external = np.array([[not ext_ordered_poly.contains(Point(x[i], y[j]))
                                   for i in range(self.n_x)]
                                  for j in range(self.n_y)])

        mask_internal = ndimage.binary_dilation(mask_internal, iterations=2)
        mask = mask_internal + mask_external

        mask_line_obs = np.zeros_like(mask, dtype=bool)
        for x_i, y_i in np.vstack((positions_start_line, positions_obstacle)):
            mask_line_obs += np.array([[Point(x_i, y_i).buffer(0.1).contains(Point(x[i], y[j]))
                                        for i in range(self.n_x)] for j in range(self.n_y)])

        self.mask = mask+mask_line_obs

        # Invert x and y for map
        self.path = astar(self.mask, self.start, self.end)
        return self.path]
