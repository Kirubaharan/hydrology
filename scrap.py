__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import operator

plate = [{u'matches_template': 0, u'plate': u'EA7THE', u'confidence': 92.405113}, {u'matches_template': 0, u'plate': u'EA7TBE', u'confidence': 83.526604}, {u'matches_template': 0, u'plate': u'EA7TRE', u'confidence': 82.455276}, {u'matches_template': 0, u'plate': u'EA7T8E', u'confidence': 82.432587}, {u'matches_template': 0, u'plate': u'6A7THE', u'confidence': 80.914902}, {u'matches_template': 0, u'plate': u'BA7THE', u'confidence': 80.913948}, {u'matches_template': 0, u'plate': u'EA7THB', u'confidence': 78.234192}, {u'matches_template': 0, u'plate': u'GA7THE', u'confidence': 78.146278}, {u'matches_template': 0, u'plate': u'EA7TE', u'confidence': 77.999985}, {u'matches_template': 0, u'plate': u'A7THE', u'confidence': 77.449036}, {u'matches_template': 0, u'plate': u'EA7TH6', u'confidence': 76.716492}, {u'matches_template': 0, u'plate': u'6A7TBE', u'confidence': 72.036392}, {u'matches_template': 0, u'plate': u'BA7TBE', u'confidence': 72.035439}, {u'matches_template': 0, u'plate': u'6A7TRE', u'confidence': 70.965065}, {u'matches_template': 0, u'plate': u'BA7TRE', u'confidence': 70.964104}, {u'matches_template': 0, u'plate': u'6A7T8E', u'confidence': 70.942375}, {u'matches_template': 0, u'plate': u'BA7T8E', u'confidence': 70.941422}, {u'matches_template': 0, u'plate': u'EA7TBB', u'confidence': 69.355682}, {u'matches_template': 0, u'plate': u'GA7TBE', u'confidence': 69.267769}, {u'matches_template': 0, u'plate': u'A7TBE', u'confidence': 68.570526}]

for i in plate:
    print i