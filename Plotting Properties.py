# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:01:19 2025

@author: shash
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from scipy.signal import argrelmax,argrelmin,find_peaks
from scipy.interpolate import interp1d,PchipInterpolator
import csv
import os
import glob
import math
import tsmoothie.smoother as sm
import pandas as pd
import seaborn as sns

#%%
csfont = {'fontname':'Times New Roman'}

fastSeeds = pd.read_csv('Statistical Analysis\Spread sheets for Plots\Fast Seeds.csv')
avgSeeds = pd.read_csv('Statistical Analysis\Spread sheets for Plots\Average Seeds.csv')
slowSeeds = pd.read_csv('Statistical Analysis\Spread sheets for Plots\Slow Seeds.csv')
#seeds = pd.read_csv('')
# fastMass = fastSeeds.loc[:,"Mass [g]"]

#plt.scatter(fastSeeds.loc[:,"m"], fastSeeds.loc[:,"t"])

fig, ((ax1, ax5), (ax2, ax4), (ax3, ax6)) = plt.subplots(3, 2, figsize=(12,9), dpi=800)
#ax1.scatter(fastSeeds.loc[:,"m"], fastSeeds.loc[:,"t"])
#ax1.scatter(avgSeeds.loc[:,"m"], avgSeeds.loc[:,"t"])
#ax1.scatter(slowSeeds.loc[:,"m"], slowSeeds.loc[:,"t"])
fig.suptitle("Morphological Property Distributions", fontsize="16", y="0.92")

#fig, ax = plt.subplots(figsize = (8, 6))
ax1.grid()
ax1.set_axisbelow(True)
ax1.scatter(avgSeeds.loc[:,"m"], avgSeeds.loc[:,"t"], label="Average")    
ax1.scatter(fastSeeds.loc[:,"m"], fastSeeds.loc[:,"t"], label="Fast")
ax1.scatter(slowSeeds.loc[:,"m"], slowSeeds.loc[:,"t"], label="Slow")
ax1.set_xlabel("Mass [$g$]")

#fig, ax2 = plt.subplots(figsize = (8,6))
ax2.grid()
ax2.set_axisbelow(True)
ax2.scatter(avgSeeds.loc[:,"A"], avgSeeds.loc[:,"t"])    
ax2.scatter(fastSeeds.loc[:,"A"], fastSeeds.loc[:,"t"])
ax2.scatter(slowSeeds.loc[:,"A"], slowSeeds.loc[:,"t"])
ax2.set_xlabel("Area [$mm^2$]")
ax2.set_ylabel("Transition Time [$s$]")


#fig, ax3 = plt.subplots(figsize = (8, 6))
ax3.grid()
ax3.set_axisbelow(True)
ax3.scatter(avgSeeds.loc[:,"L"], avgSeeds.loc[:,"t"])    
ax3.scatter(fastSeeds.loc[:,"L"], fastSeeds.loc[:,"t"])
ax3.scatter(slowSeeds.loc[:,"L"], slowSeeds.loc[:,"t"])
ax3.set_xlabel("Loading [$N/m^2$]")

#fig, ax4 = plt.subplots(figsize = (8, 6))
ax4.grid()
ax4.set_axisbelow(True)
ax4.scatter(avgSeeds.loc[:,"b"], avgSeeds.loc[:,"t"])    
ax4.scatter(fastSeeds.loc[:,"b"], fastSeeds.loc[:,"t"])
ax4.scatter(slowSeeds.loc[:,"b"], slowSeeds.loc[:,"t"])
ax4.set_xlabel("Span [$mm$]")

#fig, ax5 = plt.subplots(figsize = (8, 6))
ax5.grid()
ax5.set_axisbelow(True)
ax5.scatter(avgSeeds.loc[:,"c"], avgSeeds.loc[:,"t"])    
ax5.scatter(fastSeeds.loc[:,"c"], fastSeeds.loc[:,"t"])
ax5.scatter(slowSeeds.loc[:,"c"], slowSeeds.loc[:,"t"])
ax5.set_xlabel("Chord [$mm$]")

#fig, ax6 = plt.subplots(figsize = (8, 6))
ax6.grid()
ax6.set_axisbelow(True)
ax6.scatter(avgSeeds.loc[:,"AR"], avgSeeds.loc[:,"t"])    
ax6.scatter(fastSeeds.loc[:,"AR"], fastSeeds.loc[:,"t"])
ax6.scatter(slowSeeds.loc[:,"AR"], slowSeeds.loc[:,"t"])
ax6.set_xlabel("Aspect Ratio")

labels=["Average", "Fast", "Slow"]
fig.legend(labels, loc="center", bbox_to_anchor=(0.95, 0.15))
plt.subplots_adjust(hspace=0.3)

#%%

#seedTrajectory = pd.read_csv('Positional Data\Positions_03.csv')
seedTrajectory = pd.read_csv('Positional Data\Positions_03.csv')

fig, ax = plt.subplots(figsize=(12,5), dpi=800)
ax.grid()
ax.plot(seedTrajectory.loc[:,"tNorm"],seedTrajectory.loc[:,"xNorm"], label="Normalized X Position")
ax.plot(seedTrajectory.loc[:,"tNorm"],seedTrajectory.loc[:,"yNorm"], label="Normalized Y position")
ax.set_xlabel("Time [s]", fontsize="14")
ax.set_ylabel("Normalized Position", fontsize="14")
posLabels = ["Normalized X Position", "Normalized Y Position"]
plt.legend(posLabels, loc="best")
ax.set_axisbelow(True)
plt.title("Normalized Position of Seed 3 vs Time", fontsize="18")
#ax.plot(seed3Trajectory.loc[:,"tNorm"],seed3Trajectory.loc[:,"zNorm"])

