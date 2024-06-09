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
,id,mass,area,loading,span,chord,aspect
3,0.4,1168.7016962908854,342.2806703109598,61.121899469049126,24.83221218349697,47.0639380678127
'''


id = 3
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

# peaks
# Error:   File "/Users/shashwatsparsh/Documents/GitHub/samaraAnalysis/seedAnalysis.py", line 826, in normalVector
#    fX = PchipInterpolator(tOrient,xOrient, extrapolate=True)
# Future Fix
# minima,maxima,tOrient,xOrient,yOrient,zOrient,tCont,xCont,yCont,zCont = sa.normalVector(tNorm,smoothEccYZ,smoothXX,smoothYY,smoothZZ)
#ecc = np.sqrt(1-data['minor axis']**2/data['major axis']**2)

# With these variables, plots can be created. This whole setup can also be put into a for loop in order to generate plots of many things


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
#ThrustAccelerationIPS2 = -1 *(aYsmoothIPS2 - gIPS2)
g = 9.81 # m/s^2
massGrams = 0.4                     # [g] temporarily hard coded for seed three test case
massKg = massGrams / (1000)         # [kg] 
ThrustForce = massKg * (g - aYMPS2) # [kg*m/s^2] = [N]


# Take the Last 25 Values from the Thrust Computation because they are steady state
start = ThrustForce.size - 25
stop = ThrustForce.size
step = 1
slicedThrust = ThrustForce[start:stop:step]

#print(g-aYMPS2[start:stop:step])
#print("Accelerations are", aYMPS2[start:stop:step])
#print("SlicedThrust Values are", slicedThrust)

#print(ThrustForce)

## Omega Computation (rotational velocity)
peaks2, _ = find_peaks(xNorm, prominence=1)     
timeDifference = np.zeros(peaks2.size - 1)

for i in range(timeDifference.size):
    currTimeIndex = peaks2[i]
    nextTimeIndex = peaks2[i+1]
    timeDifference[i] = tNorm[nextTimeIndex]-tNorm[currTimeIndex]

xNormPeaks = xNorm[peaks2]

periodT = np.average([timeDifference[timeDifference.size-1],timeDifference[timeDifference.size-2]]) # [s]
#rps = 1/periodT
omega = (1/periodT) * 2 * np.pi

## Computing Thrust Coefficient
# Source: https://commons.erau.edu/cgi/viewcontent.cgi?article=1427&context=ijaaa
# Equation: T = ((0.25*vinf^2)+((1/6)*vtip^2))*rho*CL*Sb
# vinf = flow velocity = descent speed ----> zero no free stream v
# vtip = blade tip velocity = omega*R
# rho = density
# CL = Coefficient of Lift -> CT = Coefficient of Thrust
# Sb = Total Blade Area = Elipse Area?

Sb_mm2 = 1168.7016962908854 # [mm^2] Wetted Surface Area [mm^2]
SbM2 = Sb_mm2 * (1.e-06)    # [m^2]
rmm = 61.121899469049126    # [mm] seed blade length: span
rM = rmm * (0.001)          # [m]
vtip = omega * rM           # [m/s]
rho = 1.225                 # kg/m3
vinf = 0

# Based on the fact that Thrust has already been computed, Equation needs to be reformatted
# CL = CT = T * ((rho*Sb)^-1) * (((0.25*vinf^2)+((1/6)*vtip^2))^-1)
# CT = slicedThrust * (1/(rho*SbM2)) * (1/((1/6)*vtip^2))
thrustCoeffList = slicedThrust * (1/(rho*SbM2)) * (1/((1/6)*vtip*vtip))
avgThrustCoeff = np.average(thrustCoeffList)

## Computing Torque Coefficient
mjrAxisLength = rM  # [m]
ellipseArea = SbM2  # [m^2]
# Area of Ellipse Equation: A = pi*a*b == b = A / (pi*a)
# b = minorAxisLength/2, a = majorAxisLength/2
a = mjrAxisLength * 0.5
b = ellipseArea / (np.pi * a)
#Icm = massKg * ( ())

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


## Exporting to CSV
# resultsPosDf.to_csv('Positions_03.csv')
# vYSmoothDf.to_csv('Velocities_03IPS.csv')
# thrustDf.to_csv('AcclerationsAndThrust__003_01.csv')
# Accelerations.to_csv('Accelerations with filters.csv')



'''
plt.plot(aTime, aYMPS2, color='silver', label='Acceleration')
plt.plot(aTime, ksmoothAcceleration2, color='blue', linestyle='--', label='Kalman 2 Smoothing')
plt.plot(aTime, smoothAcceleration, color='green', linestyle='dotted', label='Kalman Smoothing')
plt.plot(aTime, PsmoothAcceleration, color='red', linestyle='-.', label='Polynomial Smoothing')
plt.show()
'''

'''
## Generate Plots
# Increase Plot Resolution (if necessary)
plt.rcParams['figure.dpi']=100
plt.grid(axis='y')

# testTime = tNorm

# Create plots
plt.plot(aTime, aYMPS2, color='silver', label='Acceleration')
plt.plot(aTime, PsmoothAcceleration, color='blue', linestyle='--', label='Polynomial Smoothing')
plt.plot(aTime, smoothAcceleration, color='green', linestyle='dotted', label='Kalman Smoothing')
plt.xlabel("Time [s]")
plt.ylabel('Acceleration [m/s^2]')
plt.title('Acceleration Smoothing')

plt.show()
'''