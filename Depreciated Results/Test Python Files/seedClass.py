'''
id,     mass,   area,               loading,            span,               chord,              aspect
3,      0.4,    1168.7016962908854, 342.2806703109598,  61.121899469049126, 24.83221218349697,  47.0639380678127
6,      0.7,    1522.2761139517897, 459.86532507739935, 74.06287225946649,  27.060777530154667, 56.25396802643494
7,      0.5,    1050.1453453453453, 476.1531365313653,  61.968094064523505, 21.986914804819467, 47.762287463594326
47,     0.6,    941.1195844493143,  637.5767861117292,  59.12029666385135,  20.89870341532939,  45.0324388908737
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas
import seedAnalysis as sa
import tsmoothie.smoother as sm
import scipy as sc
from scipy.signal import find_peaks

class seed:
    def __init__ (self, propertyArray)
        properties = []
        for property in propertyArray:

        self.id = propertyArray[0]
        self.mass = propertyArray[1]

