import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import numpy as np


class RootSystem:
    def __init__(self, points: np.ndarray):
        self.points = points
        self.n_points, self.dimension = np.shape(points)

    def random_valid_vector(self) -> np.ndarray:
        """
        Returns a random linear functional that is non-zero on all roots
        """
        while True:
            random_vector = np.random.rand(self.dimension, 1)
            image = np.dot(self.points, random_vector)
            # Check if there are no zeros, the margin 0.01
            # is there to prevent floating point errors
            if np.all(abs(image) > 0.02):
                return random_vector

    def positive_roots(self, vector: np.ndarray) -> np.ndarray:
        image = np.dot(self.points, vector)
        positive_indexes = [i for i in range(self.n_points) if image[i] > 0]
        return self.points[positive_indexes]

    def simple_roots(self) -> np.ndarray:
        vector = self.random_valid_vector()
        positive = self.positive_roots(vector)
        simple = self._filter_simple_roots(positive)
        return simple

    def _filter_simple_roots(self, positive: np.ndarray) -> np.ndarray:
        sums = [a + b for a in positive for b in positive]
        simple_indexes = []
        for i in range(len(positive)):
            candidate = positive[i]
            is_simple = True
            for vec in sums:
                if np.array_equal(candidate, vec):
                    is_simple = False
                    break
            if is_simple:
                simple_indexes.append(i)
        return positive[simple_indexes]

    def dynkin_diagram(self):
        simple_roots = self.simple_roots()
        dots = np.dot(simple_roots, np.transpose(simple_roots))
        n_roots = len(simple_roots)
        V = list(range(n_roots))

        def weight(i, j) -> int:
            return int(4 * (dots[i, j] ** 2) / (dots[i, i] * dots[j, j]))

        def check_edge(edge) -> bool:
            i, j, weight = edge[0], edge[1], edge[2]
            print((edge, dots[i, i], dots[j, j]))
            if i == j or weight in {0, 4}:
                return False
            if weight == 1:
                return True
            if i > j:
                return dots[i, i] >= dots[j, j]
            # if i < j:
            #     return dots[j, j] >= dots[i, i]
            return False

        all_edges = [
            (i, j, weight(i, j)) for i in range(n_roots) for j in range(n_roots)
        ]
        E = [edge for edge in all_edges if check_edge(edge)]
        print(all_edges)

        G = nx.DiGraph()
        G.add_nodes_from(V)
        G.add_weighted_edges_from(E)
        weights = nx.get_edge_attributes(G, "weight").values()
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos=pos,
            node_color="white",
            edge_color="black",
            edgecolors="black",
            width=list(map(lambda x: 1.5 * x, list(weights))),
            # node_size=3000,
        )
        weight = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weight)
        plt.tight_layout()
        plt.show()
