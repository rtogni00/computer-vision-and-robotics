import numpy as np

# compute distance between 2 points in nd.array
def dist_nodes(a, b) -> float:
    return np.sqrt(((a-b)**2).sum())

# disentangling function for internal and external path
def ordinator(l_input, max_distance=0.4, min_dist=100):
    
    _l_input=l_input
    new_order = []
    point_index = 0
    last = _l_input.pop(point_index)
    new_order.append(last)

# TODO: raises error because pop out of range
    while len(_l_input)>0:
        min_dist = 100
        index = 100
        for i, point in enumerate(_l_input):
            distance_temp = dist_nodes(last, point)
            if distance_temp < min_dist and distance_temp < max_distance:
                min_dist = distance_temp
                index = i

        last = _l_input.pop(index)
        new_order.append(last)
    return new_order