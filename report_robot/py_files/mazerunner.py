def matrix_to_meter(index_x, index_y):
    x = spatial_step*index_x+min_x+spatial_step/2
    y = spatial_step*index_y+min_y+spatial_step/2
    return (x, y)

def meter_to_matrix(x, y):
    index_x = int(abs(min_x-x)/spatial_step)
    index_y = int(abs(min_y-y)/spatial_step)
    return (index_x, index_y)