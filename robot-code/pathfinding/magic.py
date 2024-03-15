import numpy as np

# Compute distance between 2 points in np.array


def dist_nodes(a, b) -> float:
    return np.sqrt(((a-b)**2).sum())

# disentangling function for internal and external path with custum max and min distances
def ordinator(l_input, max_distance=0.5, min_dist_=100):

    orig_size = len(l_input)
    _l_input = list(l_input[:])  # copy input to avoid side effects
    new_order = []  # the output order
    last = _l_input.pop(0)  # start from the first recorded node
    new_order.append(last)
    while len(_l_input) > 0:
        min_dist = min_dist_
        index = orig_size
        for i, point in enumerate(_l_input):
            distance_temp = dist_nodes(last, point)
            if distance_temp < min_dist:
                min_dist = distance_temp
                if distance_temp < max_distance:
                    index = i
        if index == orig_size:
            # no closer candidate was found
            # part of the list will not be included
            current_len = len(new_order)
            print('Missing '+str(orig_size-current_len) +
                  ' nodes to close the polygon; min_dist = '+str(min_dist))
            return new_order[:]
        else:
            last = _l_input.pop(index)
            new_order.append(last)
    return new_order[:]

# topologically order points in a line
def line_ordinator(l_input, max_distance=0.6, min_dist_=100):
    # copy input, avoid input changing side effects
    _l_input = list(l_input[:])

    xy_aruco = np.array(_l_input)
    avg_xy = np.mean(xy_aruco, axis=0)

    # filter any false positives, namely if they are more than 0.7m far apart from the mean
    xy_aruco_filtered = [xy for xy in _l_input if dist_nodes(xy, avg_xy) < 0.7]

    xy_mid_point = np.mean(np.array(xy_aruco_filtered), axis=0)
    start = None
    max_dist = 0
    popper = None
    for i, el in enumerate(xy_aruco_filtered):
        d = dist_nodes(el, xy_mid_point)
        if d > max_dist:
            max_dist = d
            start = el
            popper = i
    # remove only if more distant than 0.3m
    xy_aruco_filtered.pop(popper)

    xy_aruco_filtered.append(start)
    new_order = ordinator(xy_aruco_filtered[::-1], max_distance, min_dist_)

    return new_order
