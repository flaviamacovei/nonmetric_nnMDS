import colour
import torch
import numpy as np
from torch.utils.data import TensorDataset

munsell_data = colour.notation.munsell.MUNSELL_COLOURS_ALL

lab_colours = torch.from_numpy(np.array([colour.convert(data[1], 'CIE XYZ', 'CIE Lab') for data in munsell_data])).float()

# data = torch.cdist(lab_colours, lab_colours)

points_3d = torch.from_numpy(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 3], [0, 3, 8], [10, 3, 15]])).float()
dataset = TensorDataset(points_3d)
# dataset = TensorDataset(lab_colours[:10])

"""
10 3 10
0 3 8
0 0 3
"""

