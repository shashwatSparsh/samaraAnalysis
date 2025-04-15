#%% Evaluating Rotational Speed

# Transition Completion is the instant the steady-state rotation begins
# It is abbreviated in the code as: TC

transitionTimeIndex = 518

# The Steady-State Rotation speed can be computed by slicing the xNorm Array for AFTER Transition
xNormSS = xNorm[transitionTimeIndex+1:]
tNormSS = tNorm[transitionTimeIndex+1:]

## FFT Analysis
# Follow this Video for more details: https://www.youtube.com/watch?v=O0Y8FChBaFU
# Get Time Step Size in Seconds
tNormStepSize = tNorm[1]-tNorm[0] # Sample Time Interval
# Get total number of samples to compute relevant frequencies
numSamples = xNormSS.size

# Compute the Frequency Magnitudes for the real input
rotationSpectrum = abs(fft.rfft(xNormSS))
# Compute the corresponding Frequencies using the total number of samples and the sample step size
rotationFrequencies = fft.rfftfreq(numSamples, d=tNormStepSize)
# plt.plot(rotationFrequencies, rotationSpectrum)

# Compute the two Dominant Frequencies for the corresponding modes
dominantFreq = rotationFrequencies[np.where(rotationSpectrum == np.max(rotationSpectrum))] # Most Dominant
secondDominantFreq = rotationFrequencies[np.where(rotationSpectrum == heapq.nlargest(2, rotationSpectrum)[1])] # Second Most Dominant