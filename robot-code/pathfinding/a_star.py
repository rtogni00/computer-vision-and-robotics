import heapq
import numpy as np
import matplotlib.pyplot as plt
import math

###############################################
# newly defined heuristic
###############################################

def distance(a,b):
    return np.sqrt((a[0]-b[0])**2+ (a[1]-b[1])**2)

def angle_between_three_points(pointA, pointB, pointC):
    # Check if pointA and pointC are the same
    if np.all(pointA == pointC):
        return 0

    # Calculate vectors AB and AC
    AB = np.subtract(pointA, pointB)
    AC = np.subtract(pointA, pointC)

    # Check if AB or AC is a zero vector
    if np.linalg.norm(AB) == 0 or np.linalg.norm(AC) == 0:
        raise ValueError("One of the input points is identical to the other.")

    # Normalize the vectors
    AB_normalized = AB / np.linalg.norm(AB)
    AC_normalized = AC / np.linalg.norm(AC)

    # Calculate the dot product
    dot_product = np.dot(AB_normalized, AC_normalized)

    # Calculate the angle in radians
    angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Convert the angle to degrees and ensure it's between 0 and 360
    angle_degrees = np.degrees(angle_radians) % 360

    return angle_degrees


def heuristic_(maze, current, start):
    total_distance = 0
    row, col =maze.shape[0]-1, maze.shape[1]-1
    max_row = 0
    min_row = 0
    max_col = 0
    min_col = 0

    # print("Rows:",row,"Cols:", col)

    # Finding max and min row
    for i in range(row):
        if (maze[i,:]==0).any():
            max_row= i
            break
    for i in range(row,0,-1):
        if (maze[i,:]==0).any():
            min_row = i
            break

    # Finding max and min col
    for i in range(col):
        if (maze[:,i]==0).any():
            max_col= i
            break
    for i in range(col,0,-1):
        if (maze[:,i]==0).any():
            min_col = i
            break

    # print("Max and min rows",max_row, min_row)
    # print("Max and min cols",max_col, min_col)
    
    center_row, center_col = max_row+int((min_row-max_row)/2), max_col+int((min_col-max_col)/2)
    center = [center_row, center_col]

    # print("Center row:",center_row)
    # print("Center col:", center_col)

    radius = max(center_row, center_col)
    # print("Radius:",radius)

    # distance between center and starting point
    dist_center_start = distance(center, start)
    dist_start_circle = radius-dist_center_start

    # distance between center and current
    dist_center_current = distance(center, current)
    dist_current_circle = radius-dist_center_current

    # angle
    angle = angle_between_three_points(current, center, start)

    # arc: 2*pi*r = theta : 360
    arc = (angle*2*math.pi*radius)/360


    # total_distance = dist_current_circle + 2*r*pi-arc + dist_start_circle

    total_distance = dist_start_circle + dist_current_circle + arc


    # return center_row, center_col, radius, total_distance
    return total_distance

###############################################
# end of newly defined heuristic
###############################################
        

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(maze, start, end):
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0),
             (1, 1), (-1, -1), (-1, 1), (1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic_(maze, start, end)}
    oheap = []

    def neighbors_available(position, maze, moves):
        neighbors = []
        for move in moves:
            x_test = position[0]+move[0]
            y_test = position[1]+move[1]
            maze_test = maze[x_test, y_test]
            if x_test >= 0 and x_test < maze.shape[0]-1:
                if y_test >= 0 and y_test < maze.shape[1]-1:
                    if maze_test != 1:
                        neighbors.append((x_test, y_test))

        return neighbors

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == end:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for neighbor in neighbors_available(current, maze, moves):
            tentative_g_score = gscore[current] + 1

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic_(maze, neighbor, end)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False


def maze_plot(maze, start, end):
    path = astar(maze, start[::-1], end[::-1]) # Invert x and y for map
    path.append(start[::-1])
    plt.imshow(maze, origin='lower')

    for i in path:
        plt.scatter(i[1], i[0], color='red')
    plt.show()
    
    path = [(x[1],x[0]) for x in path]
    print(path[::-1])


def main():

    # frame = np.ones(12,12)

    maze = np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

    start = (0, 0)
    end = (10, 10)

    maze_plot(maze, start, end)


if __name__ == '__main__':
    main()
