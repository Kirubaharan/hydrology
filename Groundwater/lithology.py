__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from striplog import striplog, Legend, Lexicon

legend = Legend.default()
lexicon = Lexicon.default()



csv_string = """  200.000,  230.329,  Anhydrite
  230.329,  233.269,  Grey vf-f sandstone
  233.269,  234.700,  Anhydrite
  234.700,  236.596,  Dolomite
  236.596,  237.911,  Red siltstone
  237.911,  238.723,  Anhydrite
  238.723,  239.807,  Grey vf-f sandstone
  239.807,  240.774,  Red siltstone
  240.774,  241.122,  Dolomite
  241.122,  241.702,  Grey siltstone
  241.702,  243.095,  Dolomite
  243.095,  246.654,  Grey vf-f sandstone
  246.654,  247.234,  Dolomite
  247.234,  255.435,  Grey vf-f sandstone
  255.435,  258.723,  Grey siltstone
  258.723,  259.729,  Dolomite
  259.729,  260.967,  Grey siltstone
  260.967,  261.354,  Dolomite
  261.354,  267.041,  Grey siltstone
  267.041,  267.350,  Dolomite
  267.350,  274.004,  Grey siltstone
  274.004,  274.313,  Dolomite
  274.313,  294.816,  Grey siltstone
  294.816,  295.397,  Dolomite
  295.397,  296.286,  Limestone
  296.286,  300.000,  Volcanic
"""


striplog = striplog.Striplog.from_csv(csv_string, lexicon=lexicon)
striplog.plot(legend, aspect=10, interval=5)
plt.show()