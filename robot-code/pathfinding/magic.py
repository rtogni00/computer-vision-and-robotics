import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, data, coords):
        self.data = data
        self.coords = coords
        self.next = None

    def __repr__(self):
        return 'node_entity<data:'+str(self.data)+', coords:'+repr(self.coords)+'>'

    def __str__(self) -> str:
        return 'id: '+str(self.data)+' -- coords: '+str(self.coords)+'\n'

    def __sub__(self, a):
        return Node(1000, self.coords-a.coords)


def dist_nodes(a: Node, b: Node) -> float:
    return np.sqrt(((a.coords-b.coords)**2).sum())


class CircularLinkedList:
    def __init__(self, nodes=None, coords_=None):
        _nodes = nodes
        _coords = coords_
        self.size = 0
        self.head = None
        if nodes is not None and coords_ is not None:
            self.size = len(nodes)
            self.first_node = Node(data=_nodes.pop(0), coords=_coords.pop(0))
            node = self.first_node
            self.head = node
            for i, elem in enumerate(nodes):
                node.next = Node(data=elem, coords=coords_[i])
                node = node.next
            node.next = self.first_node

    def __iter__(self):
        starting_point = self.head
        node = starting_point
        while node is not None and (node.next != starting_point):
            yield node
            node = node.next
        yield node

    def get(self, data):
        for node in self:
            if node.data == data:
                return node
        raise IndexError('no object found')

    def traverse(self, starting_point):
        if starting_point is None:
            starting_point = self.head
        node = starting_point
        while node is not None and (node.next != starting_point):
            yield node
            node = node.next
        yield node

    def __repr__(self):
        output = []
        for node in self:
            output.append(node.data)
        return ' -> '.join(list(map(str, output)))+' <='

    def __str__(self) -> str:
        return self.__repr__()

    def plot(self):
        x = []
        y = []
        t = []
        for node in self:
            x.append(node.coords[0])
            y.append(node.coords[1])
            t.append(node.data)
        x.append(x[0])
        y.append(y[0])
        t.append(t[0])
        plt.plot(x, y)
        plt.scatter(x, y)
        for xyt in zip(x, y, t):
            plt.text(xyt[0], xyt[1], xyt[2])
        plt.grid()
        plt.gca().set_aspect('equal')
        plt.show()

    def dump(self):
        xy = []
        ids = []
        for node in self:
            xy.append(node.coords)
            ids.append(node.data)
        return xy, ids

    def check_and_correct(self, max_distance, max_jumps=2, verbose=True):
        node = self.head
        self.plot()
        for i in range(self.size):
            # for node in self:
            dist = dist_nodes(node, node.next)
            if dist > max_distance:
                if verbose:
                    print(node.__repr__()+' and ' +
                          node.next.__repr__()+' are too far')

                next_distances = []
                next_nodes = []
                jumps = 0
                for next_node in self.traverse(node.next):
                    if jumps < max_jumps:
                        next_distances.append(dist_nodes(node, next_node))
                        next_nodes.append(next_node)
                    jumps += 1

                index_best_next_node = np.array(next_distances).argmin()
                new_best = next_nodes[index_best_next_node]
                old_next = node.next
                end_jump = new_best.next

                node.next = new_best
                new_best.next = old_next
                old_next.next = end_jump
                self.plot()
            node = node.next
