import time
import random
import statistics
from collections import defaultdict
from pyspark import SparkConf
from pyspark import SparkContext


class TriangleCounting:
    def __init__(self, C, R):
        self.C = C
        self.R = R

    def hash_function(self, u, p, a, b):
        return ((a * u + b) % p) % self.C

    @staticmethod
    def CountTriangles(edges_array):
        neighbors = defaultdict(set)
        for edge in edges_array:
            u, v = edge
            neighbors[u].add(v)
            neighbors[v].add(u)
        triangle_count = 0

        for u in neighbors:
            for v in neighbors[u]:
                if v > u:
                    for w in neighbors[v]:
                        if w > v and w in neighbors[u]:
                            triangle_count += 1
        return triangle_count

    def MR_ApproxTCwithNodeColors(self, vertex_array):
        u_list = []
        v_list = []
        vertex_hash = {}
        t_final = []
        for _ in range(self.R):
            p = 8191
            a = random.randint(1, p - 1)
            b = random.randint(0, p - 1)

            for index in range(0, int(len(vertex_array)), 2):
                if vertex_array[index] not in vertex_hash:
                    vertex_hash[vertex_array[index]] = self.hash_function(vertex_array[index], p, a, b)

                if vertex_array[index + 1] not in vertex_hash:
                    vertex_hash[vertex_array[index + 1]] = self.hash_function(vertex_array[index + 1], p, a, b)

                if vertex_hash[vertex_array[index]] == vertex_hash[vertex_array[index + 1]]:
                    u_list.append(vertex_array[index])
                    v_list.append(vertex_array[index + 1])

            result = list(map(list, zip(u_list, v_list)))
            t_final.append((self.C ** 2) * TriangleCounting.CountTriangles(result))

        return statistics.median(t_final)

    def MR_ApproxTCwithSparkPartitions(self, rawData):
        edges_rdd = rawData.map(lambda pair: pair.split(',')).map(lambda v: (int(v[0]), int(v[1]))).cache()

        partitioned_edges = edges_rdd.map(lambda edge: (hash(edge) % self.C, edge))

        triangles_per_partition = partitioned_edges.groupByKey().mapValues(
            TriangleCounting.CountTriangles).collectAsMap()

        return self.C ** 2 * sum(triangles_per_partition.values())


if __name__ == '__main__':
    sc = SparkContext(conf=SparkConf().setAppName('Triangle Counting'))

    file_name = 'facebook_large.txt'

    color = 8
    rounds = 5

    rawData = sc.textFile(file_name)

    string_edges = rawData.collect()
    vertices = []
    for pair in string_edges:
        vertex = pair.split(',')
        vertices.append(int(vertex[0]))
        vertices.append(int(vertex[1]))

    tc = TriangleCounting(C=color, R=rounds)

    print('Dataset = ' + file_name)
    print('Number of Edges = ' + str(int(len(vertices) / 2)))
    print('Number of Colors = ' + str(color))
    print('Number of Repetitions = ' + str(rounds))
    print('Approximation through node coloring')
    start = time.time()
    res = tc.MR_ApproxTCwithNodeColors(vertex_array=vertices)
    end = time.time()
    print('- Number of triangles (median over {} runs) = '.format(rounds), res)
    print('- Running time (average over {} runs) = '.format(rounds), int((end - start) * 1000), 'ms')
    print('Approximation through Spark partitions')
    start = time.time()
    res = tc.MR_ApproxTCwithSparkPartitions(rawData=rawData)
    end = time.time()
    print('- Number of triangles = ', res)
    print('- Running time (average over = ', int((end - start) * 1000), 'ms')
