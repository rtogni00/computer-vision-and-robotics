import heapq
import numpy as np
import matplotlib.pyplot as plt


# function to find quadrand
def quadrant(position, max_x, max_y, min_x, min_y):
    center_y = min_x+(max_y-min_y)/2
    center_x = min_y+(max_x-min_x)/2
    x, y = position[0:2]
    if x >= center_x and y < center_y:
        return 1
    elif x >= center_x and y >= center_y:
        return 2
    elif x < center_x and y >= center_y:
        return 3
    elif x < center_x and y < center_y:
        return 4

# function heuristics

def distance_from_arrival(position, arrival, max_x, max_y, min_x, min_y, visited_quadrant):
    quadrant_temp = quadrant(position, max_x, max_y, min_x, min_y)

    if quadrant_temp not in visited_quadrant:
        visited_quadrant.append(quadrant_temp)

    distance = 0

    delta_x = max_x-min_x
    delta_y = max_y-min_y

    center_y = min_x+delta_y/2
    center_x = min_y+delta_x/2

    x, y = position[0:2]
    x_arrival, y_arrival = arrival[0:2]

    if quadrant_temp == 1:
        # variable 1
        distance += abs(max_x-x)+abs(center_y-y)
        # 2 & 3
        distance += abs(delta_y)
        distance += abs(delta_x)
        # 4
        distance += abs(min_x)
        distance += abs(center_y)
    elif quadrant_temp == 2:
        # 2 variable
        distance += abs(max_y-y)+abs(x-center_x)
        # 3
        distance += delta_x/2
        distance += delta_y/2
        # 4
        distance += abs(min_x)
        distance += abs(center_y)

    elif quadrant_temp == 3:
        # 3 variable
        distance += abs(x-min_x)+abs(y-center_y)
        # 4
        distance += abs(min_x)
        distance += abs(center_y)
    elif quadrant_temp == 4:
        if 2 in visited_quadrant:
            distance += abs(x)+abs(y)
        else:
            distance += abs(x-max_x) + abs(y-center_y)

    return distance, visited_quadrant


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(maze, start, end):
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0),
             (1, 1), (-1, -1), (-1, 1), (1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, end)}
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
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, end)
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
