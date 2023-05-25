import time
import random
import statistics
import sys
from collections import defaultdict
from pyspark import SparkConf
from pyspark import SparkContext


class TriangleCounting:
    def __init__(self, C, R):
        self.C = C
        self.R = R

    @staticmethod
    def CountTriangles(edges):
        # Create a defaultdict to store the neighbors of each vertex
        neighbors = defaultdict(set)
        for edge in edges:
            u, v = edge
            neighbors[u].add(v)
            neighbors[v].add(u)

        # Initialize the triangle count to zero
        triangle_count = 0

        # Iterate over each vertex in the graph.
        # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
        for u in neighbors:
            # Iterate over each pair of neighbors of u
            for v in neighbors[u]:
                if v > u:
                    for w in neighbors[v]:
                        # If w is also a neighbor of u, then we have a triangle
                        if w > v and w in neighbors[u]:
                            triangle_count += 1
        # Return the total number of triangles in the graph
        return triangle_count

    @staticmethod
    def countTriangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
        # We assume colors_tuple to be already sorted by increasing colors. Just transform in a list for simplicity
        colors = list(colors_tuple)
        # Create a dictionary for adjacency list
        neighbors = defaultdict(set)
        # Create a dictionary for storing node colors
        node_colors = dict()
        for edge in edges:
            u, v = edge
            node_colors[u] = ((rand_a * u + rand_b) % p) % num_colors
            node_colors[v] = ((rand_a * v + rand_b) % p) % num_colors
            neighbors[u].add(v)
            neighbors[v].add(u)

        # Initialize the triangle count to zero
        triangle_count = 0

        # Iterate over each vertex in the graph
        for v in neighbors:
            # Iterate over each pair of neighbors of v
            for u in neighbors[v]:
                if u > v:
                    for w in neighbors[u]:
                        # If w is also a neighbor of v, then we have a triangle
                        if w > u and w in neighbors[v]:
                            # Sort colors by increasing values
                            triangle_colors = sorted((node_colors[u], node_colors[v], node_colors[w]))
                            # If triangle has the right colors, count it.
                            if colors == triangle_colors:
                                triangle_count += 1
        # Return the total number of triangles in the graph
        return triangle_count

    def MR_ApproxTCwithNodeColors(self, edges):

        t_final = []

        def hash_function(edge, p, a, b):
            h_edges = []
            h_value = {}
            for v in edge:
                h_value[v] = ((a * v + b) % p) % self.C
            if h_value[edge[0]] == h_value[edge[1]]:
                h_edges.append((h_value[edge[0]], edge))
            return h_edges

        for _ in range(self.R):
            p = 8191
            a = random.randint(1, p - 1)
            b = random.randint(0, p - 1)

            t_count = (edges.flatMap(lambda edge: hash_function(edge, p, a, b))
                       .groupByKey()
                       .map(lambda x: self.CountTriangles(x[1]))
                       .reduce(lambda x, y: x + y))

            t_final.append((self.C ** 2) * t_count)

        return statistics.median(t_final)

    def MR_ExactTC(self, edges):

        t_final = []

        for _ in range(self.R):
            p = 8191
            a = random.randint(1, p - 1)
            b = random.randint(0, p - 1)

            pairs = edges.flatMap(
                lambda x: [((tuple(sorted([(((a * x[0] + b) % p) % self.C), (((a * x[1] + b) % p) % self.C), i]))), x) for i
                           in range(self.C)])
            t_count = pairs.groupByKey().map(lambda x: self.countTriangles2(x[0], list(x[1]), a, b, p, self.C)).reduce(
                lambda x, y: x + y)

            t_final.append(t_count)

        return t_final[self.R - 1]

def main():

    assert len(sys.argv) == 5, "Usage: python G072HW2.py <C> <R> <flag> <dataset file>"

    conf = SparkConf().setAppName('Triangle Counting')
    sc = SparkContext(conf=conf)

    C = sys.argv[1]
    assert C.isdigit(), "C must be an integer"
    C = int(C)
    R = sys.argv[2]
    assert R.isdigit(), "R must be an integer"
    R = int(R)
    F = sys.argv[3]
    assert F.isdigit() and int(F) in [0, 1], "F must be 0 or 1"
    F = int(F)
    file_name = sys.argv[4]

    rawData = sc.textFile(file_name).map(lambda line: line.split(','))
    edges = rawData.map(lambda e: (int(e[0]), int(e[1]))).repartition(32).cache()

    tc = TriangleCounting(C, R)

    print('Dataset = ' + file_name)
    print('Number of Edges = ', edges.count())
    print('Number of Colors = ' + str(C))
    print('Number of Repetitions = ' + str(R))

    if F == 0:

        print('Approximation algorithm with node coloring')
        start = time.time()
        res = tc.MR_ApproxTCwithNodeColors(edges)
        end = time.time()

        print('- Number of triangles (median over {} runs) = '.format(R), res)
        print('- Running time (average over {} runs) = '.format(R), int(((end - start) * 1000) / R), 'ms')
    elif F == 1:
        print('Exact algorithm with node coloring')
        start = time.time()
        res = tc.MR_ExactTC(edges)
        end = time.time()
        print('- Number of triangles = ', res)
        print('- Running time (average over {} runs) = '.format(R), int(((end - start) * 1000) / R), 'ms')


if __name__ == '__main__':
    main()