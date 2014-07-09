__author__ = 'kiruba'
# stage area relationship curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spread import sp
# copy the code from http://code.activestate.com/recipes/577878-generate-equally-spaced-floats/ #
import itertools
from matplotlib import rc
from linecache import getline


def floodfill(c, r, mask):
    """
    Crawls a mask array containing
    only 1 and 0 values from the starting
    point (c=column, r=row-a.k.a.x,y) and returns
    an array with all 1 values connected to the starting
     cell. This algorithm performs a 4-way check non-recursively
    :param c: column
    :param r: row -a.k.a
    :param mask:
    :return:
    """
    # cells already filled
    filled = set()
    # cells to fill
    fill = set()
    fill.add((c,r))
    width = mask.shape[1]-1
    height = mask.shape[0]-1
    # Our output inundation array
    flood = np.zeros_like(mask,dtype=np.int8)
    # Loop through and modify the cells which need to be checked
    while fill:
        #grab a cell
        x,y = fill.pop()
        if y == height or x ==width or x < 0 or y < 0:
            #Dont fill
            continue







