import numpy as np
import matplotlib.pyplot as plt
import pandas
import seedAnalysis as sa
import tsmoothie.smoother as sm

# script for analyzing csv files output from createCSV.py
# lots of refinement could be done and there are a lot of unnecessary lines 
# but this will serve as an explanation of my steps

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

#print(tCont)
#print(len(tCont) + " " + len(xCont) + " " + len(yCont) + " " + len(zCont))
#print(len(vY))

posDic = {'tCont' : tNorm,
          'xCont' : xNorm,
          'yCont' : yNorm,
          'zCont' : zNorm}

resultsPos = pandas.DataFrame(posDic)
resultsPos.to_csv('Positions_03.csv')

#newVySmooth = vYsmooth.append(0)
newVySmooth = np.append(vYsmooth, 0)

#print(vYsmooth.size)
#print(tCont.size)
vYSmoothDf = pandas.DataFrame({'tCont' : data['time'],
                               'vYSmooth' : newVySmooth})
vYSmoothDf.to_csv('Velocities_03.csv')

# Note, double check figure 2.5 for the calibration to identify the appropriate units (in/s)
# Thesis Page 17
# Resolution: 1/2 Res Front View
# 
# 0.045 in/pixel front view