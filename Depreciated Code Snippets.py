# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 19:24:54 2025

@author: shashwat
"""

#%% Computing Thrust Coefficient
# # Source: https://scienceworld.wolfram.com/physics/ThrustCoefficient.html
# # T = CT * 0.5 * rho * (omega*r)^2 * A
# # CT = T * 1/(0.5 * rho * (omega*r)^2 * A)
# # T:        Thrust
# # CT:       Coefficient of Thrust
# # rho:      Density
# # omega:    Rotational Velocity
# # r:        Radius of blade
# # A:        Disk Area
# # A = pi * r^2

# rM = radii[0] * (0.001)       # [m]
# vtip = omega * rM               # [m/s]
# rho = 1.225                     # kg/m3

# DiskArea = np.pi*rM*rM          # [m^2]
# # Tip Velocity
# omegaRM = omega*rM              # [m/s]

# thrustCoeffList = slicedThrust * (1/(rho*(omegaRM)*(omegaRM)*(DiskArea)))
# avgThrustCoeff = np.average(thrustCoeffList)

# print("Evaluation Duration: ", evaluationDuration)
# #print("Thrust Coefficient List: ", thrustCoeffList)
# #print("Average Thrust Coeff: ", avgThrustCoeff)

#%% Plotting the Filtering Results
# # Comment this section out when un-needed
# fig, axs = plt.subplots(6, figsize=(12,15), sharex=True, dpi=800)
# #velocitySmoothingLabels=['Kalman Filter Smoothing','Lowess Filter Smoothing']
# for i in range(6):
#     axs[i].grid()
# # axs[0].grid()
# # axs[1].grid()
# # axs[2].grid()
# #ax.plot(vTime, vY, color='gray', linestyle=':' ,label='Raw Data')
# axs[0].plot(vTime, data['x'][:1514],
#             label="Raw Z Position (pixels)")
# axs[1].plot(vTime, xNorm[:1514],
#         linestyle='-', label='xNorm')
# axs[2].plot(vTime, vYsmooth * conversionFactorFront * inchesToMeters,
#         linestyle='-', label='Lowess Filter Smoothing')
# axs[3].plot(vTime, descentSpeed2 * conversionFactorFront * inchesToMeters,
#           linestyle='-', label='Lowess Filter Smoothing Original')
# axs[4].plot(aTime, np.diff(vYsmooth * conversionFactorFront * inchesToMeters)/np.diff(vTime),
#             label="Descent Acceleration")
# axs[5].plot(aTime, np.diff(descentSpeed2 * conversionFactorFront * inchesToMeters)/np.diff(vTime),
#             label="Descent Acceleration")
# # axs[5].plot(aTime, descentAcceleration)

# axs[0].set_ylabel("Descent Z Position")
# axs[1].set_ylabel("XNorm Position")
# axs[2].set_ylabel("Descent Velocity (K Filter)")
# axs[3].set_ylabel("Descent Velocity (L Filter)")
# axs[4].set_ylabel("Descent Acceleration (K Filter)")
# axs[5].set_ylabel("Descent Acceleration (L Filter)")
# axs[4].set_xlabel("Time [s]")

# plt.tight_layout()

#%% Descent Velocity Log Plot
# numPlots = 2
# fig, ax = plt.subplots(numPlots, figsize=(10,8), dpi=800)

# ax[0].set_yscale('log')
# ax[0].plot(vTime, descentVelocity)
# ax[0].grid()
# ax[0].set_ylabel("Descent Velocity [m/s]")
# ax[1].plot(vTime, descentVelocity)
# ax[1].set_ylabel("Descent Velocity [m/s]")
# ax[1].set_xlabel("Time [s]")
# ax[1].grid()
# ax[0].set_title('Log Scale Descent Velocity')
# ax[1].set_title("Descent Velocity")
# ax[0].axvline(x = transitionTime, color='red',
#               linestyle=':', label='Transition-Time')
# ax[1].axvline(x = transitionTime, color='red',
#               linestyle=':', label='Transition-Time')

# fig.align_labels()
# # for i in range(numPlots):
# #     ax[i].grid()
# #     ax[i].axvline(x = transitionTime, color='red',
# #                   linestyle=':', label='Transition-Time')
# #     plt.legend()
# plt.tight_layout()

#%%
# # Loop through each subplot and set tick label size
# for ax in [ax1, ax2]:
#     ax.tick_params(axis='both', which='major', labelsize=18)

# plt.tight_layout()


# fig, [ax1, ax2] = plt.subplots(2, figsize=(16,12), dpi=800)
# ax1.plot(tNormTr, xNormTr, '-')
# ax1.set_xlabel("Time [s]", fontsize=18)
# ax1.set_ylabel("Normalized X Position", fontsize=18)
# ax1.grid()
# ax2.plot(transitionFrequencies, transitionSpectrum, '-')
# ax2.set_xlabel("Frequency [Hz]", fontsize=18)
# ax2.set_ylabel("Magnitude", fontsize=18)
# ax2.grid()
# fig.suptitle("FFT Analysis of Transition", fontsize=48)

# # Loop through each subplot and set tick label size


# plt.tight_layout()

#%% FFT Validation using Find Peaks

# # Determine the index of the rotation peaks
# rotationPeaks, _ = find_peaks(xNorm, prominence = 1)     
# timeDifference = np.zeros(rotationPeaks.size - 1)

# # Actual Values @ peaks
# xNormPeaks = xNorm[rotationPeaks]
# tNormPeaks = tNorm[rotationPeaks]
# # Compute Time Difference between each tNorm Peaks Value -- dektaT [s]
# timeDifference = np.diff(tNormPeaks)    # [s]

# # Slicing Peaks Data for only steady state conditions
# numSteadyState = 3 # of steady state rotations to consider

# # Steady State Periods
# startPeriod = timeDifference.size - numSteadyState
# stopPeriod = timeDifference.size
# stepPeriod = 1
# periods = timeDifference[startPeriod:stopPeriod:stepPeriod] # [s]
# # Steady State Time Stamps
# stepTime = 1
# steadyStateTimeStamps = tNormPeaks[tNormPeaks.size-numSteadyState:
#                                     tNormPeaks.size:
#                                         stepTime]

# # Compute Angular Speed in Rad/s -> 1 Rotation 2pi
# thetaDot = (1/periods) * 2 * np.pi
# # Average Angular Speed for vtip computation
# averageThetaDot = np.average(thetaDot)
# omega = averageThetaDot