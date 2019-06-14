""""
Auxiliary functions for image processing
"""
import numpy as np


def rgb2gray(rgb):
    """
    RGB image transformation to Luma component assuming BT.601 color.
    https://en.wikipedia.org/wiki/Luma_(video)
    """
    return np.expand_dims(np.dot(rgb[..., :3], [0.299, 0.587, 0.114]), axis=2)
