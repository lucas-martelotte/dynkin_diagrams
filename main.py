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
            if np.all(image):  # if it contains no zeros
                return random_vector

    def positive_roots(self, vector: np.array) -> np.ndarray:
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
            if i == j or weight in {0, 4}:
                return False
            if i > j:
                return dots[i, i] >= dots[j, j]
            if i < j:
                return dots[j, j] >= dots[i, i]
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


# a = RootSystem(
#     np.matrix([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]])
# )
# a.dynkin_diagram()

# a = RootSystem(np.matrix([[1, 0], [-1, 0], [0, 1], [0, -1]]))
# a.dynkin_diagram()


# E8 Lattice
roots: list[list[float]] = []
for i in range(8):
    for j in range(8):
        if i == j:
            continue
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                root: list[float] = [0, 0, 0, 0, 0, 0, 0, 0]
                root[i] = s1
                root[j] = s2
                if root not in roots:
                    roots.append(root)
roots.append([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
for i in range(8):
    for j in range(8):
        for k in range(8):
            for l in range(8):
                if len({i, j, k, l}) == 4:
                    root = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                    root[i] = -0.5
                    root[j] = -0.5
                    root[k] = -0.5
                    root[l] = -0.5
                    if root not in roots:
                        roots.append(root)
for i in range(8):
    for j in range(8):
        if len({i, j}) == 2:
            root = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            root[i] = -0.5
            root[j] = -0.5
            if root not in roots:
                roots.append(root)


a = RootSystem(np.matrix(roots))
a.dynkin_diagram()
