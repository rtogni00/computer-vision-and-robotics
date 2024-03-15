import numpy as np


def dist_nodes(a, b) -> float:
    """Compute distance between 2 points in np.array
    """
    return np.sqrt(((a-b)**2).sum())


def ordinator(l_input, max_distance=0.5, min_dist_=100):
    """Disentangling function for internal and external path
    Use proper max_distance
    """
    orig_size = len(l_input)
    _l_input = l_input[:]  # copy input, avoid input changing side effects
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
