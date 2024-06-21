import numpy as np
import matplotlib.pyplot as plt
import pandas
import seedAnalysis as sa
import tsmoothie.smoother as sm
import scipy as sc
from scipy.signal import find_peaks

# script for analyzing csv files output from createCSV.py
# lots of refinement could be done and there are a lot of unnecessary lines 
# but this will serve as an explanation of my steps

# Seed Data
'''
id,     mass,   area,               loading,            span,               chord,              aspect
3,      0.4,    1168.7016962908854, 342.2806703109598,  61.121899469049126, 24.83221218349697,  47.0639380678127
6,      0.7,    1522.2761139517897, 459.86532507739935, 74.06287225946649,  27.060777530154667, 56.25396802643494
7,      0.5,    1050.1453453453453, 476.1531365313653,  61.968094064523505, 21.986914804819467, 47.762287463594326
47,     0.6,    941.1195844493143,  637.5767861117292,  59.12029666385135,  20.89870341532939,  45.0324388908737
'''


# Set ID to analyze a particular seed
id = 6
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

# front
data = pandas.read_csv(f'{date}_Data/{idText} Drop_front.csv', header = None)
data.rename(columns = {0:'time',1:'x',2:'y',3:'angle',4:'major axis',5:'minor axis'},inplace=True)

# bottom
dataB = pandas.read_csv(f'{date}_Data/{idText} Drop_bottom.csv', header = None)
dataB.rename(columns = {0:'time',1:'x',2:'y',3:'angle',4:'major axis',5:'minor axis'},inplace=True)

# angle output is defined clockwise, make counterclockwise
data['angle'] = np.absolute(-data['angle']+180)
dataB['angle'] = dataB['angle']

# get span of seed with id = id
len = dims[dims['id'] == id]['span'].values[0]

# create angle outputs
yVec,zVec,thetaFront,quad = sa.vectorize_front(data['angle'])
xVec,yVec2,alpha = sa.vectorize_bottom(dataB['angle'])

# smooth variables
smoother = sm.KalmanSmoother(component='level',component_noise={'level':0.1})
smoother.smooth(yVec)
smoothY = smoother.smooth_data[0]
smoother.smooth(zVec)
smoothZ = smoother.smooth_data[0]
smoother.smooth(xVec)
smoothX = smoother.smooth_data[0]

beta = thetaFront

smoother.smooth(data['x'])
smoothYpos = smoother.smooth_data[0]

smoother2 = sm.KalmanSmoother(component='level',component_noise={'level':0.009})

# normalization to longer time series since data and dataB have different lengths
tNorm,xNorm,yNorm,zNorm = sa.lengthMatch(data['time'],dataB['time'],smoothX,smoothY,smoothZ)
_,xPos,yPos,zPos = sa.lengthMatch(data['time'],dataB['time'],dataB['x'],data['y'],data['x'])

tNorm, alphaNorm, betaNorm, quadNorm = sa.lengthMatch(data['time'],dataB['time'],alpha,beta,quad)
_,majorAxisNorm,_,_ = sa.lengthMatch(data['time'],dataB['time'],dataB['major axis'],data['major axis'],data['minor axis'])
#alpha,beta = sa.angleCorr(data['angle'],dataB['angle'])

# more smoothing
smoother.smooth(betaNorm)
betaNormS = smoother.smooth_data[0]
smoother.smooth(alphaNorm)
alphaNormS = smoother.smooth_data[0]

# create scale factor for bottom view using length of samara from dims
mean = np.mean(majorAxisNorm[650:])
conf95 = 2*np.std(majorAxisNorm[650:])
maxLen = 2*conf95 + mean

scaleFac = len/maxLen

majorAxisNorm = [l*scaleFac for l in majorAxisNorm]

#scaleFac = np.linspace(0.132*25.4,0.095*25.4,np.size(majorAxisNorm))
#majorAxisNorm = [l*s for l,s in zip(majorAxisNorm,scaleFac)]

# create x,y,z vector outputs for X_1
XX,YY,ZZ,alphaTrue,betaTrue = sa.polar2cartesian(alphaNorm,betaNorm,majorAxisNorm,len)
smoother2.smooth(alphaTrue)
alphaTrueS = smoother2.smooth_data[0]

# smooth them out
smoother.smooth(XX)
smoothXX = smoother.smooth_data[0]
smoother.smooth(YY)
smoothYY = smoother.smooth_data[0]
smoother.smooth(ZZ)
smoothZZ = smoother.smooth_data[0]

# calculate descent speed
vY = sa.descentSpeed(data['x'],data['time'])

# smooth out the descent velocity
smoother2.smooth(vY)
vYsmooth = smoother2.smooth_data[0]

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
# peaks
# Error:   File "/Users/shashwatsparsh/Documents/GitHub/samaraAnalysis/seedAnalysis.py", line 826, in normalVector
#    fX = PchipInterpolator(tOrient,xOrient, extrapolate=True)
# Future Fix
# minima,maxima,tOrient,xOrient,yOrient,zOrient,tCont,xCont,yCont,zCont = sa.normalVector(tNorm,smoothEccYZ,smoothXX,smoothYY,smoothZZ)
#ecc = np.sqrt(1-data['minor axis']**2/data['major axis']**2)
'''
# With these variables, plots can be created. This whole setup can also be put into a for loop in order to generate plots of many things


## Seed Specific Information
# Span in radius [mm] ID
radiusmm3 = 61.121899469049126  # [mm]
radiusmm6 = 74.06287225946649   
radiusmm7 = 61.968094064523505
radiusmm47 = 59.12029666385135
# Masses [g]
massGrams3 = 0.4                        # [g]
massGrams6 = 0.7                        # [g]
massGrams7 = 0.5                        # [g]
massGrams47 = 0.6                       # [g] temporarily hard coded for seed three test case


## Computing Velocity in m/s
# Thesis Page 17 figure 2.5 for calibration and unit conversion
# Multiply by conversion factor of 0.1 in/pixel * .0254m/in
conversionFactorFront = 0.1 # in/pixel
conversionFactorBot = 0.045 # in/pixel
inchesToMeters = 0.0254

vYsmoothMPS = vYsmooth * conversionFactorFront * inchesToMeters
# Pull Time from Data Frame and remove last value as there are n-1 Velocities
vTime = data['time'].to_numpy()[:-1]


## Computing Acceleration in m/s^2
# a = delta V / delta T
# use numpy differencing function on on vYsmoothMPS and vTime
aYMPS2 = np.diff(vYsmoothMPS) / np.diff(vTime)
# Remove last value as there are n-2 Accelerations
aTime = vTime[:-1]


## Computing Thrust Force
# Fnet = m*a = m*g - Thrust <=> Thrust = m*g - m*a <=> Thrust = m*(g-a)
# note: a = a_net
# ThrustAccelerationIPS2 = -1 *(aYsmoothIPS2 - gIPS2)

g = 9.81                                # m/s^2
massKg = massGrams6 / (1000)           # [kg] 
ThrustForce = massKg * (g - aYMPS2)     # [kg*m/s^2] = [N]

# Take the Last n Values from the Thrust Computation because they are steady state
numEvals = 15
thrustValS = numEvals                                                 # Seed 3
startThrust = ThrustForce.size - thrustValS
stopThrust = ThrustForce.size
stepThrust = 1
slicedThrust = ThrustForce[startThrust:stopThrust:stepThrust]
evaluationDuration = tNorm[tNorm.size-1] - tNorm[tNorm.size-(numEvals+1)]


''' Debugging
#print(g-aYMPS2[start:stop:step])
#print("Accelerations are", aYMPS2[start:stop:step])
#print("SlicedThrust Values are", slicedThrust)
#print(ThrustForce)
'''

## Computing Rotational Velocity: Omega
# Peaks 2 is the indexes of the peaks -- Use with xNorm to find Peak value and tNorm to find time value
peaks2, _ = find_peaks(xNorm, prominence = 1)

# Linear Interpolation to find the time when xNorm is last Zero
LXNGZ = .0102342            # Last XNorm Value Greater than Zero
NXNLZ = -0.013871           # Next XNorm Value Less than Zero
LXNGZTime = 0.349397        # Corresponding Time Stamp
NXNLZTime = 0.349974        # Next TimeStamp

# Online Linear Interpolator
initialTimeStamp = 0.34969951702537216

# Compute Time Stamp of Negative Peak
negNorm = xNorm * -1
peaks3, _ = find_peaks(negNorm, prominence = 0.01)
maxNegXNormEnd = xNorm[peaks3[peaks3.size-2]]
lowIndex = np.where(xNorm == maxNegXNormEnd)
finalTimeStamp = tNorm[lowIndex]
period = (finalTimeStamp - initialTimeStamp) * 4



'''
timeDifference = np.zeros(peaks2.size - 1)
#print(timeDifference)
# Actual Values @ peaks
xNormPeaks = xNorm[peaks2]
tNormPeaks = tNorm[peaks2]
# Compute Time Difference between each tNorm Peaks Value -- dektaT [s]
timeDifference = np.diff(tNormPeaks)    # [s]
negNorm = xNorm * -1
peaks3, _ = find_peaks(negNorm, prominence = 0.01)
xNormPeaksNegative = xNorm[peaks3]
tNormPeaksNegative = tNorm[peaks3]
timeDifference = tNormPeaksNegative[tNormPeaksNegative.size-2] - tNormPeaks
periodForSeed3 = timeDifference
'''

# Slicing Peaks Data for only steady state conditions
numSteadyState = 1  # of steady state rotations

'''
# Steady State Periods
startPeriod = timeDifference.size - numSteadyState
stopPeriod = timeDifference.size
stepPeriod = 1
periods = timeDifference[startPeriod:stopPeriod:stepPeriod] # [s]
# Steady State Time Stamps
stepTime = 1
steadyStateTimeStamps = tNormPeaks[tNormPeaks.size-numSteadyState:
                                   tNormPeaks.size:
                                       stepTime]
'''
#periods = periodForSeed3 * 2    
    
# Compute Angular Speed in Rad/s -> 1 Rotation 2pi
thetaDot = (1/period) * 2 * np.pi
# Average Angular Speed for vtip computation
#averageThetaDot = np.average(thetaDot)
#omega = averageThetaDot
omega = thetaDot

## Computing Thrust Coefficient
# Source: https://scienceworld.wolfram.com/physics/ThrustCoefficient.html
# T = CT * 0.5 * rho * (omega*r)^2 * A
# CT = T * 1/(0.5 * rho * (omega*r)^2 * A)
# T:        Thrust
# CT:       Coefficient of Thrust
# rho:      Density
# omega:    Rotational Velocity
# r:        Radius of blade
# A:        Disk Area
# A = pi * r^2

rM = radiusmm6 * (0.001)       # [m]
vtip = omega * rM               # [m/s]
rho = 1.225                     # kg/m3

DiskArea = np.pi*rM*rM          # [m^2]
# Tip Velocity
omegaRM = omega*rM              # [m/s]

thrustCoeffList = slicedThrust * (1/(rho*(omegaRM)*(omegaRM)*(DiskArea)))
avgThrustCoeff = np.average(thrustCoeffList)
avgThrust = np.average(slicedThrust)
print("Evaluation Duration: ", evaluationDuration)
print("Average Thrust: ", avgThrust)
print("Average Omega: ", omega)

#print("Thrust Coefficient List: ", thrustCoeffList)
#print("Average Thrust Coeff: ", avgThrustCoeff)


'''
## Generate DataFrames
resultsPosDf = pandas.DataFrame({'tNorm' : tNorm,
                                 'xNorm' : xNorm,
                                 'yNorm' : yNorm,
                                 'zNorm' : zNorm})

vYSmoothDf = pandas.DataFrame({'Time [s]' : vTime,
                               'Descent Velocity [m/s]' : vYsmoothMPS})


thrustDf = pandas.DataFrame({   'Time [s]' : aTime,
                                'Net Acceleration [m/s^2]' : aYMPS2,
                                'Thrust Force [N]' : ThrustForce })
'''

## Exporting to CSV
# resultsPosDf.to_csv('Positions_03.csv')
# vYSmoothDf.to_csv('Velocities_03IPS.csv')
# thrustDf.to_csv('AcclerationsAndThrust__003_01.csv')
# Accelerations.to_csv('Accelerations with filters.csv')

zeroX = 0
zeroT = initialTimeStamp

lowestX = maxNegXNormEnd
lowestT = finalTimeStamp 

plt.plot(tNorm, xNorm)
plt.plot(zeroT, zeroX, 'ro')
plt.text(zeroT + .01, zeroX + .03, s = [initialTimeStamp])
plt.plot(lowestT, lowestX, 'bo')
plt.text(0.4, lowestX - .08, s = finalTimeStamp)
plt.plot(tNorm[peaks2], xNorm[peaks2], 'go')
plt.text(tNorm[peaks2] + .01, xNorm[peaks2], s = tNorm[peaks2])
plt.grid()
plt.title(label=' Seed 6 xNorm vs Time')
plt.xlabel("Time [s]")
plt.ylabel("Normalized X Position")
plt.show()