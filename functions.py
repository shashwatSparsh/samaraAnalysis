# -*- coding: utf-8 -*-
"""
Created on Sun May 11 12:08:38 2025

@author: shash

Response Analysis Functions

"""

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import pandas
import seedAnalysis as sa
import tsmoothie.smoother as sm
import scipy as sc
from scipy.signal import find_peaks
import heapq


def getDominantFrequencies(dataSignal, timeSignal):
    timeStepSize = timeSignal[1]-timeSignal[0]
    numSamples = dataSignal.size
    signalMagnitudes = abs(fft.rfft(dataSignal))
    signalFrequencies = fft.rfftfreq(numSamples, d=timeStepSize)
    
    firstDominantFreqMag = np.max(signalMagnitudes)
    secondDominantFreqMag = heapq.nlargest(2, signalMagnitudes)[1]
    
    firstDominantFreq = signalFrequencies[np.where(signalMagnitudes == firstDominantFreqMag)]
    secondDominantFreq = signalFrequencies[np.where(signalMagnitudes == secondDominantFreqMag)]
    
    solutions = np.array([firstDominantFreq, secondDominantFreq])
    return solutions

def getDominantRotationRates(frequencies, scalingFactor):
    
    
    
    return frequencies