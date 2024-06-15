# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:57:52 2024

@author: shash
"""

''' Old thrust Computation

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
rmm47 = 59.12029666385135     # [mm]
#rmm6 = 74.06287225946649
rM = rmm47 * (0.001)          # [m]
vtip = omega * rM           # [m/s]
rho = 1.225                 # kg/m3
vinf = 0

DiskArea = np.pi*rM*rM          # [m^2]
omegaRM = omega*rM

# Based on the fact that Thrust has already been computed, Equation needs to be reformatted
# CL = CT = T * ((rho*Sb)^-1) * (((0.25*vinf^2)+((1/6)*vtip^2))^-1)
# CT = slicedThrust * (1/(rho*SbM2)) * (1/((1/6)*vtip^2))
#thrustCoeffList = slicedThrust * (1/(rho*SbM2)) * (1/((1/6)*vtip*vtip))
'''



'''
# Compute Angular Accleration in Rad/s^2 by finite differencing: delta omega / delta time
#thetaDotDiff = np.diff(thetaDot)
#timeDiff = np.diff(steadyStateTimeStamps)
thetaDoubleDot = np.diff(thetaDot)/np.diff(steadyStateTimeStamps)
#print(thetaDot)
#print(averageThetaDot)
#print(thetaDoubleDot)
#print(np.average(thetaDoubleDot))


## Computing Torque Coefficient
# Source: http://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node86.html#SECTION06374200000000000000
# Equation Old: Q = (0.5*rho*(v_tip^2)*Sb*D)
# Equation New: Q = Cq * rho * vtip^2 * D^5
# D:= Diameter of prop = 2*rM
# Identify Geometric Values
# Major Axis Length b is the span of the blade
mjrAxisLength = rM  # [m]
# Ellipse Area is the Wetted Area
ellipseArea = SbM2  # [m^2]
# Area of Ellipse Equation: A = pi*a*b == b = A / (pi*a)
# b = minorAxisLength/2, a = majorAxisLength/2
majorRadius = mjrAxisLength * 0.5
minorRadius = ellipseArea / (np.pi * majorRadius)
minorAxisLength = minorRadius * 2

# Moment of Inertia along major Axis and Center of Mass
Icm = (massKg/4) * (np.square(majorRadius) + np.square(minorRadius))
# Parallel Axis Theorem
# Irotation reflects the moment of inertia of the seed as it is an ellipse rotating about the tip of the major axis
Irotation = Icm + (np.square(majorRadius)*massKg)
# Torque = Q = I * alpha => Irotation * thetaDoubleDot
Torque = Irotation * thetaDoubleDot

# Reformatting Torque Equation
# Cq = Q/ (rho * vtip^2 * D^5)
torqueCoeffList = Torque / (rho*np.square(vtip)*np.power(rM,5))

#print("Theta Dot = ", thetaDot)
#print("Average Theta Dot = ", averageThetaDot)
#print("Theta Double Dot = ", thetaDoubleDot)
#print("Average Theta Double Dot = ", np.average(thetaDoubleDot))
#print("Torque Coefficient List = ", torqueCoeffList)

'''

'''
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
