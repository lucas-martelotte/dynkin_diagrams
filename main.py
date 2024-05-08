from root_systems.root_system import RootSystem
from root_systems.e8 import E8RootSystem
import numpy as np

a = RootSystem(
    np.matrix([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]])
)
a.dynkin_diagram()

a = RootSystem(np.matrix([[1, 0], [-1, 0], [0, 1], [0, -1]]))
a.dynkin_diagram()


e8 = E8RootSystem()
e8.dynkin_diagram()
