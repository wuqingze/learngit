from __future__ import division
import os, sys

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


import matplotlib.pyplot as plt

np.random.seed(12345)
np.set_printoptions(precision = 4)
plt.rc('figure', figsize=(10,6))
br = '\n'

print('hello world')


s2 = Series([0,9,8,7],['a','b','c','d'])

print(type(s2[['c','b','d']]))


