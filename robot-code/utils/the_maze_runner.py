import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
from utils.a_star import *
from utils.magic import ordinator
from scipy import ndimage


class MazeRunner():
    def __init__(self, data, config):
        self.spatial_step = config.maze.spatial_step  # 5cm of the grid resolution
        # get data from data object
        self.positions = np.array(data.landmark_estimated_positions)
        self.ids = np.array(data.landmark_estimated_ids)
        self.robot_pose = data.robot_position
        self.n_x = None
        self.n_y = None
        self.start = ()
        self.end = ()
        self.min_x = None
        self.min_y = None
        self.mask = None
        self.n_dilations = config.maze.dilations
        self.radius_obstacle = config.maze.radius_obstacle
        self.radius_startline = config.maze.radius_startline

    # conversion from grid to real world
    def matrix_to_meter(self, index_x, index_y):
        x = self.spatial_step*index_x+self.min_x
        y = self.spatial_step*index_y+self.min_y
        return (x, y)

    # conversion from real world to grid
    def meter_to_matrix(self, x, y):
        index_x = int(abs(self.min_x-x)/self.spatial_step)
        index_y = int(abs(self.min_y-y)/self.spatial_step)
        return (index_x, index_y)

    # creat maze to run the a star
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

        positions_external = positions_array[mask_external]
        positions_internal = positions_array[mask_internal]
        positions_obstacle = positions_array[mask_obstacle]
        positions_start_line = positions_array[mask_start_line]

        pos_list_ext = [pos for pos in positions_external]
        pos_list_int = [pos for pos in positions_internal]

        # more flexibility on external circuit
        external_ordered = np.array(ordinator(pos_list_ext, max_distance=0.6))
        internal_ordered = np.array(ordinator(pos_list_int))

        max_x = np.max(external_ordered[:, 0])
        max_y = np.max(external_ordered[:, 1])
        self.min_x = np.min(external_ordered[:, 0])
        self.min_y = np.min(external_ordered[:, 1])

        delta_x = max_x-self.min_x
        delta_y = max_y-self.min_y

        self.n_x = int(delta_x/self.spatial_step)
        self.n_y = int(delta_y/self.spatial_step)

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

        # Create a mask where the pixels inside the first circle and outside the second circle are set to 1, and all others are set to 0
        mask_internal = np.array([[int_ordered_poly.contains(Point(x[i], y[j]))
                                   for i in range(self.n_x)]
                                  for j in range(self.n_y)])
        mask_external = np.array([[not ext_ordered_poly.contains(Point(x[i], y[j]))
                                   for i in range(self.n_x)]
                                  for j in range(self.n_y)])

        # erosion of internal region to avoid stepping over the track
        mask_internal = ndimage.binary_dilation(
            mask_internal, iterations=self.n_dilations)

        mask = mask_internal + mask_external

        # addition of obstacles markers and start line to split the track
        mask_line_obs = np.zeros_like(mask, dtype=bool)
        for x_i, y_i in positions_obstacle:
            mask_line_obs += np.array([[Point(x_i, y_i).buffer(self.radius_obstacle).contains(Point(x[i], y[j]))
                                        for i in range(self.n_x)]
                                       for j in range(self.n_y)])

        for x_i, y_i in positions_start_line:
            mask_line_obs += np.array([[Point(x_i, y_i).buffer(self.radius_startline).contains(Point(x[i], y[j]))
                                        for i in range(self.n_x)]
                                       for j in range(self.n_y)])

        self.mask = mask+mask_line_obs

    # trigger the a star computation
    def create_path(self, start, end):
        # compute the mask
        print('Start computing the maze')
        self.create_mask()

        # get the matrix path with mask
        print('Computing a star')
        path_matrix = astar(self.mask, self.meter_to_matrix(
            *start), self.meter_to_matrix(*end))
        if not path_matrix:
            print('Path not found')
            print(self.meter_to_matrix(*start), self.meter_to_matrix(*end))
            print('Dumping the mask on mask.txt')
            np.savetxt('mask.txt', self.mask)
        # return the path in geometric coordinates
        path_meter = [self.matrix_to_meter(*xy) for xy in path_matrix]

        # trim the complexity of the path
        path_meter = path_meter[::4]
        path_meter.append(path_meter[0])
        # remove the first point for our convenience: it is always bad
        path_meter = path_meter[1:]
        print("Path computed, ready to start race")
        return path_meter
