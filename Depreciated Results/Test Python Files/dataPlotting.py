# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:33:45 2024

@author: shash
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
import seedAnalysis as sa
import tsmoothie.smoother as sm
import scipy as sc
from scipy.signal import find_peaks

idText = '003'
date = '20230427'
trajectorySeed3 = pandas.read_csv(f'{date}_Data/{idText}  Trajectory.csv')

