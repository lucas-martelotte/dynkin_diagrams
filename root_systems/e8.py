from .root_system import RootSystem
import numpy as np


class E8RootSystem(RootSystem):
    def __init__(self) -> None:
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
                    for r in range(8):
                        if len({i, j, k, r}) == 4:
                            root = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                            root[i] = -0.5
                            root[j] = -0.5
                            root[k] = -0.5
                            root[r] = -0.5
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
        super().__init__(np.matrix(roots))
