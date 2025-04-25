#%% Imports and other informational Data

# import os
# os.chdir(path='samaraAnalysis')

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import pandas
import seedAnalysis as sa
import tsmoothie.smoother as sm
import scipy as sc
from scipy.signal import find_peaks
import heapq

# script for analyzing csv files output from createCSV.py
# lots of refinement could be done and there are a lot of unnecessary lines 
# but this will serve as an explanation of my steps

# Air Density
rho = 1.225 #kg/m^3

# Seed Data
'''
id,     mass,   area,               loading,            span,               chord,              aspect
3,      0.4,    1168.7016962908854, 342.2806703109598,  61.121899469049126, 24.83221218349697,  47.0639380678127
6,      0.7,    1522.2761139517897, 459.86532507739935, 74.06287225946649,  27.060777530154667, 56.25396802643494
7,      0.5,    1050.1453453453453, 476.1531365313653,  61.968094064523505, 21.986914804819467, 47.762287463594326
47,     0.6,    941.1195844493143,  637.5767861117292,  59.12029666385135,  20.89870341532939,  45.0324388908737
'''

## The following portion should be automated in the future to reference the seed properties csv file and index based on the id set.
# Currently the values are hard-coded for simplicity.
## Seed Specific Information
# Span in radius [mm] ID
spanmm3 = 61.12  # [mm]
spanmm6 = 74.06   
spanmm7 = 61.96
spanmm47 = 59.12
radii = .001*(np.array([spanmm3, spanmm7, spanmm47])-15)
# Masses [g]
massGrams3 = 0.4                        # [g]
massGrams6 = 0.7                        # [g]
massGrams7 = 0.5                        # [g]
massGrams47 = 0.6                       # [g] temporarily hard coded for seed three test case
massesKg = .001*np.array([massGrams3, massGrams7, massGrams47])

bladeAreas = (10**-6)*np.array([1168, 1050, 941])

colors = ['tab:orange', 'tab:blue', 'tab:green']
seedVal = [0, 1, 2]
coningAngleTCS = [np.arctan(8/15), np.arctan(9/17), np.arctan(8/12)]
coningAngleSSS = [np.arctan(5/17), np.arctan(4/19), np.arctan(4/15)]

ids = [3, 7, 47]

#%% Setting ID to analyze specific Seed
# Set ID to analyze a particular seed

seedIndex = 0
currentID = ids[seedIndex]

id = currentID
ssIndex = 800



currentSpan = radii[seedIndex]
currentMass = massesKg[seedIndex]
currentSeedVal = seedVal[seedIndex]


date = '20230427'

idText = 'text'
# get id text
if id < 10:
    idText = f'00{int(id)}'
elif id >= 10 and id < 100:
    idText = f'0{int(id)}'
elif id >= 100:
    idText = f'{int(id)}'
    

print(idText)

# read in files
dims = pandas.read_csv('sampleProperties.csv')

#%% Front View Analysis 
# Process the data to get the positional vectors
# Note, the "x" column in this actually refers to the Z position of the seed.
# It is named x because the video frame is horizontal: in other words the z axis is horizontal
# This is the same reaosn the vertical axis is called y
data = pandas.read_csv(f'{date}_Data/{idText} Drop_front.csv', header = None)
data.rename(columns = {0:'time',1:'x',2:'y',3:'angle',4:'major axis',5:'minor axis'},inplace=True)

# bottom
# Read the Bottom View DataFrame
dataB = pandas.read_csv(f'{date}_Data/{idText} Drop_bottom.csv', header = None)
# Rename the Columns of the DataFrame
dataB.rename(columns = {0:'time',1:'x',2:'y',3:'angle',4:'major axis',5:'minor axis'},inplace=True)
# Extract the Time column from the bottom view
timeB = dataB.iloc[:, 0];

# angle output is defined clockwise, make counterclockwise
data['angle'] = np.absolute(-data['angle']+180)
dataB['angle'] = dataB['angle']

# get span of seed with id = id
len = dims[dims['id'] == id]['span'].values[0]

# Getting Angles to compute orientation
yVec,zVec,thetaFront,quad = sa.vectorize_front(data['angle'])
xVec,yVec2,alpha = sa.vectorize_bottom(dataB['angle'])


# Smothing Variables
# Define the smoother parameters
smoother = sm.KalmanSmoother(component='level',component_noise={'level':0.1})
# Smooth the data one vector at a time, in this case starting with the YVector
smoother.smooth(yVec)
# Set the new Y positional data from the smoother object.
smoothY = smoother.smooth_data[0]
# Rinse and Repeat. Everytime the smoother is called, the old data is wiped
# In other words, once smoother is used on the Z Vector, smoother.smooth_data[0]
# no longer refers to yVector but now has been overwritten with zVector data.
smoother.smooth(zVec)
smoothZ = smoother.smooth_data[0]
smoother.smooth(xVec)
smoothX = smoother.smooth_data[0]

# The Beta Angle is the frontTheta Angle Extracted
beta = thetaFront

smoother.smooth(data['x'])
smoothYpos = smoother.smooth_data[0]

# # # Generate Data Frame of positional vectors post-smoothing
# positionalVectorsData = np.array([timeB, smoothX, smoothY, smoothZ])
# positionalColumnNames = ['Time',
#                           'smooth X Position',
#                           'smooth Y Position',
#                           'smooth Z Position']
# positionalDataFrame = pandas.DataFrame(positionalVectorsData, positionalColumnNames)

# positionalDataFrame.to_csv('rawPositionalData.csv')

#%% Data Normalization using eccentricities and Angle normalization
# Normalization to longer time series since data and dataB have different lengths
tNorm,xNorm,yNorm,zNorm = sa.lengthMatch(data['time'],dataB['time'],smoothX,smoothY,smoothZ)
_,xPos,yPos,zPos = sa.lengthMatch(data['time'],dataB['time'],dataB['x'],data['y'],data['x'])

tNorm, alphaNorm, betaNorm, quadNorm = sa.lengthMatch(data['time'],dataB['time'],alpha,beta,quad)
_,majorAxisNorm,_,_ = sa.lengthMatch(data['time'],dataB['time'],dataB['major axis'],data['major axis'],data['minor axis'])
# alpha,beta = sa.angleCorr(data['angle'],dataB['angle'])

# Smooth out the alpha and beta norm angles using the smoother defined previously
smoother.smooth(betaNorm)
betaNormS = smoother.smooth_data[0]
smoother.smooth(alphaNorm)
alphaNormS = smoother.smooth_data[0]

#%% Bottom View Analysis
# create scale factor for bottom view using length of samara from dims
mean = np.mean(majorAxisNorm[650:])
conf95 = 2*np.std(majorAxisNorm[650:])
maxLen = 2*conf95 + mean

scaleFac = len/maxLen

majorAxisNorm = [l*scaleFac for l in majorAxisNorm]

#scaleFac = np.linspace(0.132*25.4,0.095*25.4,np.size(majorAxisNorm))
#majorAxisNorm = [l*s for l,s in zip(majorAxisNorm,scaleFac)]

smoother2 = sm.KalmanSmoother(component='level',component_noise={'level':0.009})

# create x,y,z vector outputs for X_1
XX,YY,ZZ,alphaTrue,betaTrue = sa.polar2cartesian(alphaNorm,betaNorm,majorAxisNorm,len)
smoother2.smooth(alphaTrue)
alphaTrueS = smoother2.smooth_data[0]

# smooth them outa
smoother.smooth(XX)
smoothXX = smoother.smooth_data[0]
smoother.smooth(YY)
smoothYY = smoother.smooth_data[0]
smoother.smooth(ZZ)
smoothZZ = smoother.smooth_data[0]

# eccentricity
eccYZ = sa.ecc(data['major axis'],data['minor axis'])
eccXY = sa.ecc(dataB['major axis'],dataB['minor axis'])

# normalization to time
_, eccXY, eccYZ, _ = sa.lengthMatch(data['time'],dataB['time'],eccXY,eccYZ,data['x'])

smoother2.smooth(eccYZ)
smoothEccYZ = smoother2.smooth_data[0]
smoother.smooth(eccXY)
smoothEccXY = smoother.smooth_data[0]

'''
This Find Peaks Code did not appear to work correctly and has been depreciated. Fixing this code may offer some
insight into possible future eccentricity tracking and optimization.

# peaks
# Error:   File "/Users/shashwatsparsh/Documents/GitHub/samaraAnalysis/seedAnalysis.py", line 826, in normalVector
#    fX = PchipInterpolator(tOrient,xOrient, extrapolate=True)
# Future Fix
# The following two lines are the broken pieces of code: 
# minima,maxima,tOrient,xOrient,yOrient,zOrient,tCont,xCont,yCont,zCont = sa.normalVector(tNorm,smoothEccYZ,smoothXX,smoothYY,smoothZZ)
# ecc = np.sqrt(1-data['minor axis']**2/data['major axis']**2)
'''
# With these variables, plots can be created. This whole setup can also be put into a for loop in order to generate plots of many things

#%% Kinematic Response Analysis

# Calculate descent speed, remember the "x" variable here is the z position during descent
descentPosition = data['x']
descentPositionTiming = data['time']
dPosSmootherK = sm.KalmanSmoother(component='level_trend',
                                  component_noise={'level':0.1,'trend':0.1},
                                  observation_noise=1)
dPosSmootherK.smooth(descentPosition)
descentPositionSmoothedK = dPosSmootherK.smooth_data[0]
dPosSmootherL = sm.LowessSmoother(0.3, 1)
dPosSmootherL.smooth(descentPosition)
descentPositionSmoothedL = dPosSmootherL.smooth_data[0]

descentSpeed2 = sa.descentSpeed(descentPositionSmoothedL, descentPositionTiming)

descentVelocityRaw = sa.descentSpeed(data['x'],data['time'])

# Smooth out the descent velocity
# Note: There should be a better smoothing function created here that is better
# suited to remove noise

# Kalman Filtering
dVKsmoother = sm.KalmanSmoother(component='level',component_noise={'level':0.5})
dVKsmoother.smooth(descentVelocityRaw)
descentVelocityK = dVKsmoother.smooth_data[0]

# Lowess Filtering
Lsmoother = sm.LowessSmoother(0.9, 1)
Lsmoother.smooth(descentVelocityRaw)
descentVelocityL = Lsmoother.smooth_data[0]


## Computing Velocity in m/s
# Thesis Page 17 figure 2.5 for calibration and unit conversion
# Multiply by conversion factor of 0.045 in/pixel * .0254m/in
# This conversion factor is based on Kai's original dataset
# Shashwat Sparsh Data set uses the following conversion factor: 
# Make sure you remember to put a TSQUARE in the camera view frame and take a snapshot to get this value

# Kais Conversion Factor
# Do not delete this conversion factor from the code, it is necessary to analyze legacy data.
Kbot = 0.1 # in/pixel
Kfront = 0.045 # in/pixel

# Set to appropriate Conversion Factor
conversionFactorFront = Kbot
conversionFactorBot = Kfront
# Convert from inches to Meters
inchesToMeters = 0.0254
# Convert to m/s
descentVelocity = descentVelocityL * inchesToMeters
# Pull Time from Data Frame and remove last value as there are n-1 Velocities
vTime = data['time'].to_numpy()[:-1]
# Two arrays of descent velocity and descent time have been generated

#axs[2].plot(vTime, vYsmooth * conversionFactorFront * inchesToMeters,
#         linestyle='--', label='Lowess Filter Smoothing')
#plt.legend(velocitySmoothingLabels, loc="best")


#%% Dynamic Response Computation

## Computing Acceleration in m/s^2
# use numpy backward differencing function on on vYsmoothMPS and vTime
# a = delta V / delta T
descentAcceleration = np.diff(descentVelocity) / np.diff(vTime)
Lsmoother.smooth(descentAcceleration)
descentAccelerationS = Lsmoother.smooth_data[0]
# Remove last value as there are n-2 Accelerations compared to positions
aTime = vTime[:-1]

## Computing Average Thrust Force
# Fnet = m*a = m*g - Thrust <=> Thrust = m*g - m*a <=> Thrust = m*(g-a)
# note: a = a_net
# ThrustAccelerationIPS2 = -1 *(aYsmoothIPS2 - gIPS2)
g = 9.81                                    # Gravitational Acceleration [m/s^2]
ThrustForce = massesKg[0] * (g - descentAcceleration)    # [kg*m/s^2] = [N]

# Take the Last n Values from the Thrust Computation because they are steady state
numEvals = 4
thrustValS = numEvals                                                 # Seed 3
startThrust = ThrustForce.size - thrustValS
stopThrust = ThrustForce.size
stepThrust = 1
slicedThrust = ThrustForce[startThrust:stopThrust:stepThrust]
evaluationDuration = tNorm[tNorm.size-1] - tNorm[tNorm.size-(numEvals+1)]

#%% Finding Transition

# Transition Completion is the instant the steady-state rotation begins
# It is abberiviated in the code as: TC
# Find when Transition Occurs
# transitionTimeIndex is the index that can be used to parse for various values @ the index

#transitionTimeMarker, transitionTimeIndex = sa.findTransition(tNorm,alpha=np.array(alpha),startSearch=0.35)

# The alternate and maybe more appropriate method of computing transition time may be to find the minima of the normalized
# position after some period of time based on each seed, (eg fast transitioning seeds, 0.35s, average, 0.4, etc.)

tNormStart = np.where(tNorm >= 0.3)[0][0]
tNormEnd = np.argmax(tNorm>=0.36)

transitionTimeIndex = tNormStart + \
                      np.argmin(xNorm[tNormStart:tNormEnd])

# Find Transition Time
transitionTime = tNorm[transitionTimeIndex]
print(transitionTime)



#%% Steady State FFT

# The Steady-State Rotation speed can be computed by slicing the xNorm Array for AFTER Transition
xNormSS = xNorm[transitionTimeIndex+1:]
tNormSS = tNorm[transitionTimeIndex+1:]

## FFT Analysis
# Follow this Video for more details: https://www.youtube.com/watch?v=O0Y8FChBaFU
# Get Time Step Size in Seconds
tNormStepSize = tNorm[1]-tNorm[0] # Sample Time Interval
# Get total number of samples to compute relevant frequencies
numSamplesSS = xNormSS.size

# Compute the Frequency Magnitudes for the real input
steadyStateSpectrum = abs(fft.rfft(xNormSS))
# Compute the corresponding Frequencies using the total number of samples and the sample step size
steadyStateFrequencies = fft.rfftfreq(numSamplesSS, d=tNormStepSize)
# plt.plot(rotationFrequencies, rotationSpectrum)

# Compute the two Dominant Frequencies for the corresponding modes
dominantFreqMagSS1 = np.max(steadyStateSpectrum)
dominantFreqMagSS2 = heapq.nlargest(2, steadyStateSpectrum)[1]

dominantFreqSS1 = steadyStateFrequencies[np.where(steadyStateSpectrum == dominantFreqMagSS1)] # Most Dominant
dominantFreqSS2 = steadyStateFrequencies[np.where(steadyStateSpectrum == dominantFreqMagSS2)] # Second Most Dominant

# Get Rotation Rate from Frequency
omegaSS = dominantFreqSS2 * (2*np.pi)
omegaSS1 = dominantFreqSS1 * (2*np.pi)

#%% Transition Analysis

# The Steady-State Rotation speed can be computed by slicing the xNorm Array for AFTER Transition
xNormTr = xNorm[:transitionTimeIndex]
tNormTr = tNorm[:transitionTimeIndex]

## FFT Analysis
# Follow this Video for more details: https://www.youtube.com/watch?v=O0Y8FChBaFU
# Get Time Step Size in Seconds
tNormStepSize = tNorm[1]-tNorm[0] # Sample Time Interval
# Get total number of samples to compute relevant frequencies
numSamplesTr = xNormTr.size

# Compute the Frequency Magnitudes for the real input
transitionSpectrum = abs(fft.rfft(xNormTr))
# Compute the corresponding Frequencies using the total number of samples and the sample step size
transitionFrequencies = fft.rfftfreq(numSamplesTr, d=tNormStepSize)
# plt.plot(transitionFrequencies, transitionSpectrum)

# Compute the two Dominant Frequencies for the corresponding modes
dominantFreqMagTr1 = np.max(transitionSpectrum)
dominantFreqMagTr2 = heapq.nlargest(2, transitionSpectrum)[1]

dominantFreqTr1 = transitionFrequencies[np.where(transitionSpectrum == dominantFreqMagTr1)] # Most Dominant
dominantFreqTr2 = transitionFrequencies[np.where(transitionSpectrum == dominantFreqMagTr2)] # Second Most Dominant


#%% Computing Disk Loading and Dynamic Pressure to estimate the CL at the end of transition

# Find the Coning Angle at end of Transition
# coningAngleTC = betaTrue[transitionTimeIndex]
# The coning Angle Computation is incorrect in the software and was computed manually by reviewing the footage

# Seed 3: SS np.arctan(8/18) TC np.arctan
#coningAngleTC = np.arctan(8/15)
#coningAngleSS = np.arctan(5/17)
#coningAngleTC = np.arctan(9/17) # in Radians
#coningAngleSS = np.arctan(4/19)

coningAngleTC = coningAngleTCS[seedIndex]
coningAngleSS = coningAngleSSS[seedIndex]

# Find the descent velocity at the end of Transition
descentVelocityTC = descentVelocity[transitionTimeIndex]
descentVelocitySS = descentVelocity[ssIndex]
# Acceleration at completion of transition
accelerationTC = descentAcceleration[transitionTimeIndex]


## Find the incomming flow over the blade during rotation
# V_incomming = Rotational Speed * Span;
# Because the incomming flow over the entire blade area varies based on where along the span
# you are measuring, you can average out the lowest and highest speeds instead
# Oncomming flow speed at the root of the blade is effectively zero because it 
# lines up with the axis of rotation
rootSpeed = 0;
# The highest oncomming flow speed as at the tip because this maximizes the radius of rotation
# Thus the oncomming flow speed can be calculated as:
tipSpeed = (omegaSS1*currentSpan)
oncommingFlowAvg = 0.5*(tipSpeed+rootSpeed)

# Because the vectors of the flow experienced by the blade are different, the sum can
# be calculated by taking the vecotr sum of the descent Velocity and the oncommingFlowAvg
totalBladeFlowVelocity = np.sqrt(np.square(oncommingFlowAvg)+np.square(descentVelocityTC))
#totalBladeFlowVelocity = oncommingFlowAvg + (descentVelocityTC*np.cos(coningAngleTC))

# Based on the Niu Atkins Intrinsic Equilibrium of Samara Auto-rotation, the CL can be approximated
# by balancing the disk loading and the dynamic pressure
# EQN: 1/CL ~ 0.5rhoVd^2/delta where delta is disk-loading and Vd is descent velocity


# Dynamic Pressure for a propeller is evaluated using the following:
# https://www.grc.nasa.gov/WWW/k-12/VirtualAero/BottleRocket/airplane/propth.html
# https://www.grc.nasa.gov/www/k-12/airplane/propth.html
# As the free-stream velocity is zero in this test, the room has still air, the V0 = 0;
# The flow experienced over the samara blade was calculated prior

dynamicPressureTC = 0.5*rho*np.square(descentVelocityTC)
dynamicPressureSS = 0.5*rho*np.square(descentVelocitySS)

# The Diskloading is a function of the coning angle of the seed as greater coning angles reduce the area of disk generating during the spin
#      *
#  ___*   Steeper angle of the blade indicated by  * with respect to the horizontal indicated ___
#      **
# ___**   Shallower angle of the blade indicated by * with respect to the horizontal indicated ___
# Thus the radius of the disk is equal to span * cos(coningAngle) == span*cos(betaNormS)

# Disk Area: A = pi * (span*cos(betaNormS))^2
diskRadiusTC = currentSpan*np.cos(coningAngleTC)
diskRadiusSS = currentSpan*np.cos(coningAngleSS)
diskAreaTC = np.pi * (diskRadiusTC**2)
diskAreaSS = np.pi * (diskRadiusSS**2)

# Disk Loading = mg/A
diskLoadingTC = (currentMass*g)/diskAreaTC

diskLoadingSS = (currentMass*g)/diskAreaSS

# Disk Solidity = Ablade/Adisk
# diskSolidity = 0.0011687/diskArea
#print(diskSolidity)

# CL at completion of transition
CL_TC = diskLoadingTC/dynamicPressureTC
CL_SS = diskLoadingSS/dynamicPressureSS

# Recall Thrust is given by T = m * (g-a)
ThrustTC = currentMass * (g-accelerationTC) # [N]
# Because this value is super small, it becomes useful to convert to mN
ThrustTC_mN = ThrustTC*1000
# To get the specific thrust and normalize by mass, divide the mN thrust by the mass
# The mass should be converted back to grams from kg in order for this parameter to make sense
specificThrustTC = ThrustTC_mN/(currentMass*1000) # mN/g Transition

ThrustSS = currentMass * (g-descentAcceleration[ssIndex])
specificThrustSS = ThrustSS/currentMass

#%% Computing Drag Coefficient
# The equation is as follows: mg ~ C_d * 0.5rhoVd^2 * A costheta
# C_d = mg / (0.5rhoVd^2 * A costheta)

CdTC = (currentMass*g) / (0.5*rho*(descentVelocityTC**2)*bladeAreas[seedIndex])
CdSS = (currentMass*g) / (0.5*rho*(descentVelocity[ssIndex]**2)*bladeAreas[seedIndex])


#%% Plot FFT Results

axsYFFT = [xNormTr, xNormSS, transitionSpectrum, steadyStateSpectrum]
axsXFFT = [tNormTr, tNormSS, transitionFrequencies, steadyStateFrequencies]
axsYLabels = ["Normalized X Position", "Frequency Magnitude", "Normalized X Position", "Frequency Magnitude"]
axsXLabels = ["Time [s]", "Frequency [Hz]", "Time [s]", "Frequency [Hz]"]

fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2, 2, figsize=(20,10), dpi=800)

ax1.plot(tNormTr, xNormTr, colors[currentSeedVal])
ax2.plot(tNormSS, xNormSS, colors[currentSeedVal])
ax3.plot(transitionFrequencies, transitionSpectrum, colors[currentSeedVal])
ax4.plot(steadyStateFrequencies, steadyStateSpectrum, colors[currentSeedVal])

ax1.set_xlabel("Time [s]", fontsize=18)
ax1.set_ylabel("Normalized X Position \n Transition", fontsize=18)
ax2.set_xlabel("Time [s]", fontsize=18)
ax2.set_ylabel("Normalized X Position \n Steady-State", fontsize=18)

ax3.set_xlabel(axsXLabels[1], fontsize=18)
ax3.set_ylabel(axsYLabels[1], fontsize=18)
ax4.set_xlabel(axsXLabels[1], fontsize=18)
ax4.set_ylabel(axsYLabels[1], fontsize=18)

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

fig.suptitle("FFT Analysis", fontsize=48)

for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(axis='both', which='major', labelsize=18)

fig.align_labels()
plt.tight_layout()

#%% Normalized Trajectory
fig, ax = plt.subplots(figsize = (12, 6), dpi = 800)
normalizedTrajectories = [xNorm, yNorm]
normColors = ['r', 'b']
normLabels = ["Normalized X Position", "Normalized Y Position"]
for i in range(2):
    ax.plot(tNorm, normalizedTrajectories[i], normColors[i], label=normLabels[i])

ax.set_xlabel("Time [s]", fontsize=18)
ax.set_ylabel("Normalized Position", fontsize = 18)
fig.suptitle("Normalized Trajectory Seed 3", fontsize=28)

ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=12)
fig.align_labels()
ax.grid()
ax.legend()
plt.tight_layout()

#%% Descent Plot Generation

numPlots = 4
fig, axs = plt.subplots(numPlots, figsize=(17,12), dpi=800)
axs[0].plot(descentPositionTiming, descentPosition*inchesToMeters, label="Descent Position", color=colors[currentSeedVal])
axs[0].set_ylabel("Descent Position Raw [m]", fontsize=14)
axs[1].plot(tNorm, xNorm, label="xNorm", color=colors[currentSeedVal])
axs[1].set_ylabel("Normalized X Position", fontsize=14)
axs[2].plot(vTime, descentVelocity, label="Descent Velocity", color=colors[currentSeedVal])
axs[2].set_ylabel("Descent Velocity [m/s]", fontsize=14)
axs[3].plot(aTime, descentAccelerationS, label='Descent Acceleration', color=colors[currentSeedVal])
axs[3].set_ylabel("Descent Acceleration [$m/s^2$]", fontsize=14)
axs[3].set_xlabel("Time [s]", fontsize=14)

transitionTimeLabel = "Transition Time", round(transitionTime, 3), "s"

for i in range(numPlots):
    axs[i].grid()
    axs[i].axvline(x = transitionTime, color='black',
                    linestyle=':', label=transitionTimeLabel)
    # axs[i].legend()
axs[3].legend()
fig.align_labels()
# plt.text(transitionTime-.03, -320, 'Transition time')
fig.suptitle("Seed 3 Response: Fast Transition", fontsize=22)
plt.tight_layout()

#%% Filtering Comparison
# numPlots = 3

# fig, ax = plt.subplots(numPlots, figsize=(10, 8), dpi=800)

# yLabels = ["Raw Data [m/s]", "Kalman Filtering [m/s]", "Lowess Filtering [m/s]"]
# descentVelocitiesPlot = [descentVelocityRaw, descentVelocityK, descentVelocityL]
# for i in range(numPlots):
#     ax[i].grid()
#     ax[i].set_ylabel(yLabels[i])
#     ax[i].plot(vTime, descentVelocitiesPlot[i]*inchesToMeters)
    
# ax[2].set_xlabel("Time [s]")
# fig.suptitle("Filtering Comparison")
# plt.tight_layout()

# ax[0].plot(vTime, descentVelocityK, label='Kalman Filtering')
# ax[1].plot(vTime, descentVelocityL, label='Lowess Filtering')
#%% Lab Trajectory
# fig, ax = plt.subplots(figsize = (12, 6), dpi = 800)
# Trajectories = [smoothX, yNorm]
# normColors = ['r', 'g', 'b']
# normLabels = ["X Position", "Y Position", "Z Position"]
# for i in range(2):
#     ax.plot(tNorm, Trajectories[i], normColors[i], label=normLabels[i])

# ax.set_xlabel("Time [s]", fontsize=18)
# ax.set_ylabel("Normalized Position", fontsize = 18)
# fig.suptitle("Normalized Trajectory Seed 3", fontsize=28)

# ax.minorticks_on()
# ax.tick_params(axis='both', which='major', labelsize=12)
# ax.grid()
# ax.legend()
# plt.tight_layout()