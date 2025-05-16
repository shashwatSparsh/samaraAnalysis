# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 01:08:39 2024

@author: shash
"""
'''
## Smoothing
# Rolling Average Smoother
# kernel_size = 20
# kernel = np.ones(kernel_size) / kernel_size
# smoothedAcceleration = np.convolve(aYMPS2, kernel, mode='same')

# Polynomial Smoother
polySmooth = sm.PolynomialSmoother(degree=5)
polyData = aYMPS2
polySmooth.smooth(polyData)
PsmoothAcceleration = polySmooth.smooth_data[0]

# Kalman Smoother
kSmoother = sm.KalmanSmoother(  component='level_trend',
                                component_noise={'level':0.1, 'trend':0.1})
kSmoother2 = sm.KalmanSmoother( component='level',component_noise={'level':0.009})
kSmoother.smooth(aYMPS2)
kSmoother2.smooth(aYMPS2)
smoothAcceleration = kSmoother.smooth_data[0]
ksmoothAcceleration2 = kSmoother2.smooth_data[0]

Accelerations = pandas.DataFrame({'Time [s]' : aTime,
                                  'aYMPS2 [m/s^2]' : aYMPS2,
                                  'Polynomial [m/s^2]' : PsmoothAcceleration,
                                  'Kalman1 [m/s^2]' : smoothAcceleration,
                                  'Kalman2 [m/s^2]' : ksmoothAcceleration2 })

kSmootherAlpha = sm.KalmanSmoother( component='level',component_noise={'level':0.009})
kSmootherAlpha.smooth(vY)
kSmootherBeta = sm.KalmanSmoother( component='level',component_noise={'level':0.0009})
kSmootherBeta.smooth(vY)
specSmoother = sm.SpectralSmoother(smooth_fraction=0.5, pad_len=1)
specSmoother.smooth(vY)
polySmoother = sm.PolynomialSmoother(degree=5)
polySmoother.smooth(vY)
gSmoother = sm.GaussianSmoother(n_knots=100,sigma=0.01)
gSmoother.smooth(vY)

smoothResults = [kSmootherAlpha.smooth_data[0],
                 kSmootherBeta.smooth_data[0],
                 specSmoother.smooth_data[0],
                 polySmoother.smooth_data[0],
                 gSmoother.smooth_data[0]]

print(smoothResults)

velocitiesDframe = pandas.DataFrame({'vTime' : vTime,
                                     'vY' : vY,
                                     'Kalman Smoother' : smoothResults[0],
                                     'Spectral Smoother' : smoothResults[2],
                                     'Polynomial Smoother' : smoothResults[3],
                                     'Gaussian Smoother' : smoothResults[4]})

velocitiesDframe.to_csv('Smoothed Velocities.csv')


#fig, axs = plt.subplots(5)
#axs[0] = plt.plot(vTime, vY, color = 'silver')
#axs[0] = plt.plot(vTime, smoothResults[0])
#axs[1] = plt.plot(vTime, vY, color = 'silver')
#axs[1] = plt.plot(vTime, smoothResults[1])
#axs[2] = plt.plot(vTime, vY, color = 'silver')
#axs[2] = plt.plot(vTime, smoothResults[2])
#axs[3] = plt.plot(vTime, vY, color = 'silver')
#axs[3] = plt.plot(vTime, vY, smoothResults[3])
#axs[4] = plt.plot(vTime, vY, color = 'silver')
#axs[4] = plt.plot(vTime, smoothResults[4])

#plt.plot(vTime, vY, color = 'silver')
#plt.plot(vTime, smoothResults[0])
# plt.plot(vTime, smooth)
#plt.show()
'''