# Library for functions analyzing seeds
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

# object detector
# obj_detector = cv2.createBackgroundSubtractorMOG2(history = 1000, varThreshold=10, detectShadows=False)

# analysis class
class vidAnalysis:

    # init function
    def __init__(self, filename):

        self.name = filename
        self.reducedName = filename[15:-4]

        #print(self.name)
        

        # filename = path location of file
        self.cap = cv2.VideoCapture(filename)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # original video filmed at 1500 fps
        self.frameTime = 1/1500 # sec/frame

        self.resFront = 0.041 # in/px
        self.resBottom = 0.02 # in/px

    # reopen capture for future analysis
    def reopen(self):
        self.cap = cv2.VideoCapture(self.name)

    # analysis functions, amking use of the other functions in this file
    def analyze(self, lower, upper, lower2, upper2, type, date):
        
        # type = string

        cents = []
        vxs = []
        vys = []
        xs = []
        ys = []
        angles = []
        frames = []
        timeStamps = []
        majorLens = []
        minorLens = []
        y_axis = np.array([0,1])


        """ plt.ion()
        fig = plt.figure(figsize=(16,4))
        ax = fig.add_subplot(111)
        plt.grid() """

        # line1, = ax.plot(timeStamps, angles,'b-')
        

        # analysis for front view
        if type == 'front':

            # starting csv writer
            f = open(f"{date}_Data/{self.reducedName}_front.csv",'w')
            writer = csv.writer(f)

            # setting labels
            # plt.xlabel("Time (sec)")
            # plt.ylabel("Ellipse Angle (degrees)")
            # plt.title(f"Ellipse Angle vs. Time {self.name}")
            # plt.xlim([0,0.9])
            # plt.ylim([0,180])
            # plt.rc('axes', labelsize = 20)
            # fig.show()
            i = 0

            # performing frame iteration:
            while self.cap.isOpened():

                ret, frame = self.cap.read()


                #frame = imutils.resize(frame,width = 1200)

                if ret == True:

                    # iterate frame count
                    i += 1

                    # isolate frames of reference
                    frontView = self.frontView(frame)

                    # bottomView = bottomView(frame)

                    # get contours from each frame
                    cnt, mask_front = self.outlineColor(frontView,lower,upper,lower2,upper2)

                    # conts_bottom, mask_bottom = sa.outlineColor(bottomView, lower,upper)
                    # check cnt, replace with very small rectangle
                    if len(cnt) < 5:
                        
                        cnt = np.array([[1,1],[1,0],[0,0],[1,0]])
                    
                    # calc area
                    area = cv2.contourArea(cnt)

                    # check area and draw contour
                    if area > 100:
                        #cv2.drawContours(frontView, [cnt], -1, (0,255,0),2)
                        
                        ((x,y), (majorAxis,minorAxis), angle) = cv2.fitEllipse(cnt)
                        
                        ## COMMENT THIS LINE TO TURN ON/OFF contours
                        cv2.ellipse(frontView,((x,y), (majorAxis,minorAxis), angle), (0,255,0),2)

                        if i != 0:
                            # append frame count
                            frames.append(i)

                            # elapsed time
                            timeStamps.append(i*self.frameTime)
                            
                            # x position
                            xs.append(x*self.resFront)
                            ys.append(y*self.resFront)
                            #myLine = np.array([vxs[-1],vys[-1]])
                            #dotProd = np.dot(y_axis,myLine)
                            angles.append(angle)
                            #angles.append(np.sin(vys[-1]/vxs[-1])*180/np.pi)
                            #line1.set_data(timeStamps,angles)

                            majorLens.append(majorAxis)
                            minorLens.append(minorAxis)
                            

                            #fig.canvas.draw()
                            
                            #fig.canvas.flush_events()
                            #time.sleep(1e-10)

                    

                    # show views for testing
                    cv2.imshow('Front View',frontView)
                    cv2.imshow('Front Mask',mask_front) 

                    #cv2.imshow('Bottom View',bottomView)
                    #cv2.imshow('Bottom Mask',mask_bottom)

                    # quit statement
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
    
                else:
                    cv2.destroyAllWindows()

                    # write csv files
                    for i in range(len(timeStamps)):
                        writer.writerow([timeStamps[i],xs[i],ys[i],angles[i],majorLens[i],minorLens[i]])

                    #f.close()
                    break
        

        ## BOTTOM ##
        elif type == 'bottom':
            obj_detector = cv2.createBackgroundSubtractorKNN(5000,700,detectShadows=False)
            # starting csv writer
            f = open(f"{date}_Data/{self.reducedName}_bottom.csv",'w')
            writer = csv.writer(f)

            # setting labels
            # plt.xlabel("Time (Sec)")
            # plt.ylabel("Ellipse Angle (degrees)")
            # plt.title(f"Ellipse Angle vs. Time {self.name}")
            # plt.xlim([0,0.9])
            # plt.ylim([0,180])

            # fig.show()
            i = 0

            # performing frame iteration:
            while self.cap.isOpened():

                ret, frame = self.cap.read()


                #frame = imutils.resize(frame,width = 1200)

                if ret == True:

                    # iterate frame count
                    i += 1

                    # isolate frames of reference
                    # frontView = frontView(frame)

                    bottomView = self.bottomView(frame)

                    # get contours from each frame
                    #conts_front, mask_front = outlineColor(frontView,lower,upper)

                    #cnt, mask_bottom = self.outlineColor(bottomView, lower,upper,lower2,upper2)
                    cnt,mask_bottom = self.outline(bottomView,obj_detector)

                    if len(cnt) < 5:
                        
                        cnt = np.array([[1,1],[1,0],[0,0],[1,0]])
                    
                    # calc area
                    area = cv2.contourArea(cnt)

                    # check area and draw contour
                    if area > 90:
                        #cv2.drawContours(bottomView, [cnt], -1, (0,255,0),2)

                        ((x,y), (majorAxis,minorAxis), angle) = cv2.fitEllipse(cnt)
                        
                        ## COMMENT THIS LINE TO SHOW ELLIPSE ##
                        cv2.ellipse(bottomView,((x,y), (majorAxis,minorAxis), angle), (0,255,0),2)

                        if i != 0:
                            # append frame count
                            frames.append(i)

                            # elapsed time
                            timeStamps.append(i*self.frameTime)
                            
                            # x position
                            xs.append(x*self.resFront)
                            ys.append(y*self.resFront)
                            #myLine = np.array([vxs[-1],vys[-1]])
                            #dotProd = np.dot(y_axis,myLine)
                            angles.append(angle)
                            #angles.append(np.sin(vys[-1]/vxs[-1])*180/np.pi)
                            #line1.set_data(timeStamps,angles)

                            majorLens.append(majorAxis)
                            minorLens.append(minorAxis)
                            
                            

                            #fig.canvas.draw()
                            
                            #fig.canvas.flush_events()
                            #time.sleep(1e-10)


                    # show views for testing
                    #cv2.imshow('Front View',frontView)
                    #cv2.imshow('Front Mask',mask_front) 

                    cv2.imshow('Bottom View',bottomView)
                    cv2.imshow('Bottom Mask',mask_bottom)

                    # quit statement
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
    
                else:
                    cv2.destroyAllWindows()

                    for i in range(len(timeStamps)):
                        writer.writerow([timeStamps[i],xs[i],ys[i],angles[i],majorLens[i],minorLens[i]])
                    
                    #f.close()
                    break

    def outline(self,frame, obj_detector):
        # applies outline contour to input frame using background subtractor
            # see cv2.createBackGroundSubtractorMOG2 docs
        
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        blurred = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        
        # apply mask
        mask = obj_detector.apply(blurred)
        mask = cv2.dilate(mask,None,iterations = 4)
        mask = cv2.erode(mask,None,iterations = 2)
        _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
        
        
        conts, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #conts = max(conts, key = cv2.contourArea)

        if len(conts) != 0:
            bigCont = max(conts,key = cv2.contourArea)
        else:
            bigCont = conts
        return bigCont, mask

    def outlineColor(self,frame, lower1, upper1, lower2, upper2):
        # USES HSV mask for outline extraction
        
        # adj 20230525 to incl. reds better
        frame = cv2.GaussianBlur(frame, (3, 3), cv2.BORDER_DEFAULT)

        # convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # perform masking operation w/ first range
        mask1 = cv2.inRange(hsv, lower1, upper1)
        #mask = cv2.erode(mask,None,iterations = 2)
        
        mask1 = cv2.dilate(mask1,None,iterations = 4)
        mask1 = cv2.erode(mask1,None,iterations = 2)

        # again with second range
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        mask2 = cv2.dilate(mask2,None,iterations = 4)
        mask2 = cv2.erode(mask2,None,iterations = 2)

        mask = mask1 | mask2

        # find contours
        conts, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(conts) != 0:
            bigCont = max(conts,key = cv2.contourArea)
        else:
            bigCont = conts
        return bigCont, mask

    def outlineBW(self,frame,lower,upper):

        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # create BW image
        bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('bw', bw)

        # mask with range of bw intensities
        mask = cv2.inRange(bw, lower, upper)
        #cv2.imshow('mask',mask)
        #mask = cv2.erode(mask,None,iterations = 2)
        mask = cv2.dilate(mask,None,iterations = 1)

        conts, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return conts, mask


    ## VIEWS
    def bottomView(self,frame):
        # applies rotation and transformation data to bottom view
            # additionally sharpens data
            # frame must be ad

        # 20230427 Frame
        frame = frame[25:197,238:402]
        rotAngle = 51

        # 20230525 Frame
        #frame = frame[13:183,273:432]
        #rotAngle = 54.9

        # rotate image
        
        h,w = frame.shape[:2]

        

        center = (w/2,h/2)
        rotationMatrix = cv2.getRotationMatrix2D(center,rotAngle,1)
        rotatedImage = cv2.warpAffine(frame,rotationMatrix,(h,w))

        #cv2.imshow('original',frame)
        #cv2.imshow('rotated',rotatedImage)

        h,w = rotatedImage.shape[:2]

        # crop image
        # 20230427
        crop = rotatedImage[38:h-35,15:w-22]

        # 20230525
        #crop = rotatedImage[36:h-30,15:w-22]
        
        h,w = crop.shape[:2]

        #cv2.imshow('cropped',crop)

        resized = cv2.resize(crop,[w*2,h*2],interpolation=cv2.INTER_CUBIC)
        #cv2.imshow('resized',resized)

        #kernel = np.array([[0, -1, 0],
        #                [-1, 5,-1],
        #                [0, -1, 0]])
        #image_sharp = cv2.filter2D(src=resized, ddepth=-1, kernel=kernel)
        #cv2.imshow('sharpened',image_sharp)

        image_sharp = resized

        return image_sharp

    def frontView(self,frame):
    

        # crop image
        frontView = frame[300:,:]

        # Rectangle Params
        startRow= 265
        endCol = 75
        h = 25
        color = (255,255,255)
        thickness = -1

        start_pt = (0,startRow)

        end_pt = (endCol,startRow + h)
        
        centerPt = (20,275)
        radius = 50

        # blank out dropper w/ circle and rectangle
        frontView = cv2.circle(frontView,centerPt,radius,color,thickness)
        frontView = cv2.rectangle(frontView,start_pt,end_pt,color,thickness)

        return frontView

def clearfiles():
    files = glob.glob('/20230427_Data/*')
    for f in files:
        os.remove(f)

# functions for calculating orientation information
# front view (gives y and z)
def vectorize_front(angle_data):

    #loop to calculate orientation vector for yz
    y = []
    z = []
    beta = []
    quad = []

    # perform logic for flip point
    flip = 0
    i = 0
    diff = 0
    ind_flip = len(angle_data)-1

    while flip == 0 and i < len(angle_data):

        # get angle in radians
        theta = angle_data[i]*np.pi/180

        # perform assignment for vector pre-flip
        y.append(-np.cos(theta))
        z.append(-np.sin(theta))
        beta.append(theta + np.pi)

        if beta[i] > 3*np.pi/2 and beta[i] <= 2*np.pi:
            quad.append(4)
        elif beta[i] >= np.pi and beta[i] <= 3*np.pi/2:
            quad.append(3)

        # get difference number for data after first iter
        if i >= 1:
            diff = np.abs(angle_data[i] - angle_data[i-1])

        # perform check for loop exit
        if diff > 170:
            flip = 1
            ind_flip = i
        
        # increase i as if nothing happened (in most cases nothing happens)
        i += 1

    # reset i to ind_flip
    i = ind_flip

    # continue loop with post-flip logic
    for angle in angle_data[ind_flip:]:

        # get angle in radians
        theta = angle*np.pi/180

        # reset ind_flip index of y and z, then continue as normal
        if i == ind_flip:
            y[i] = np.cos(theta)
            z[i] = np.sin(theta)
            beta[i] = theta

            if beta[i] > np.pi/2 and beta[i] <= np.pi:
                quad[i] = 2
            elif beta[i] >= 0 and beta[i] <= np.pi/2:
                quad[i] = 1
        else:
            y.append(np.cos(theta))
            z.append(np.sin(theta))
            beta.append(theta)

            if beta[i] > np.pi/2 and beta[i] <= np.pi:
                quad.append(2)
            elif beta[i] >= 0 and beta[i] <= np.pi/2:
                quad.append(1)
        
        
        

        i += 1

    yVec = y
    zVec = z

    return yVec,zVec,beta,quad

# bottom view (gives x)
def vectorize_bottom(angle_data):

    # initialize x and diff
    x = []
    y = []
    alpha = []
    diff = 0

    # go through logic for data
    flip = 0
    i = 0

    # calibration angle transformation
    pt1 = (-6.8163,0.3014)
    pt2 = (-3.0861, -3.8993)

    ratio = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    adj = np.arctan(ratio)

    # angle adjustment
    #adj = 51 + adj*180/np.pi

    #adj = 153.9
    adj = 61.1
    j = 0
    # create angle adjustment
    for angle in angle_data:
        # adjusted angle
        adjAngle = angle + adj

        if adjAngle > 180:
            angle_data[j] = adjAngle - 180
        else:
            angle_data[j] = adjAngle
        
        j += 1

    for angle in angle_data:
        theta = angle*np.pi/180

        # start diff loop
        if i != 0:
            diff = np.abs(angle-angle_data[i-1])
        
        # check if flip occurred on prev iteration
        '''
        if diff > 150:
            if flip == 0:
                flip == 1
                print('flip (zero to 1)!')
            else:
                flip == 0
                print('flip (1 to zero!)')
        '''
        if diff > 160 and flip == 0:
            flip = 1

        elif diff > 160 and flip == 1:
            flip = 0
            

            
        # perform assignment based on current flip value
        if flip == 0:
            y.append(-np.cos(theta))
            x.append(-np.sin(theta))
            alpha.append(theta + np.pi)
        if flip == 1:
            y.append(np.cos(theta))
            x.append(np.sin(theta))
            alpha.append(theta)

            

        i += 1
    
    return x,y,alpha

# alpha and theta_bottom coords to l=1 cartesian vector
def polar2cartesian(alpha,thetaFront,majAxis,len):
    # use normalized vectors (eq length)
    '''
    smoother = sm.KalmanSmoother(component='level',component_noise={'level':0.1})
    smoother.smooth(alpha)
    alpha = smoother.smooth_data[0]
    smoother.smooth(thetaFront)
    thetaFront=smoother.smooth_data[0]
    '''
    x = []
    y = []
    z = []
    beta = []

    

    # iterate and assign variable names
    i = 0
    bNew = 0
    flip = 0
    #flip_inds = []
    #j = 0
    count = 0
    for tF,tB,maj in zip(thetaFront,alpha,majAxis):
        
        #beta.append(np.arctan(np.tan(tF)*np.cos(tB)))
        if maj/len <= 1:
            if tF > np.pi:
                beta.append(-np.arccos(maj/len))
            if tF < np.pi:
                beta.append(np.arccos(maj/len))
        elif i > 0:
            beta.append(beta[i-1])
        elif i == 0:
            beta.append(0)
        '''
        if i < 1:
            bNew = np.arctan2(np.tan(tF)*np.sin(tB),1)
        if bNew > 0:
            beta.append(-1*np.arctan2(np.tan(tF)*np.sin(tB),1))
        elif bNew < 0:
            beta.append(1*np.arctan2(np.tan(tF)*np.sin(tB),1))
        
        diff = np.abs(beta[i]-beta[i-1])

        if diff > 100*np.pi/180 and flip == 0 and count == 0:
            flip = 1
            #flip_inds.append(i)
            count += 1
        
        elif diff > 100*np.pi/180 and flip == 1:
            flip = 0
            flip_inds.append(i)
            j += 1

        if len(flip_inds) > 1 and flip_inds[j-1] - flip_inds[j-2] < 10:
            if flip == 1:
                flip == 0
            elif flip == 0:
                flip == 1
        
        if flip == 1:
            beta[i] = -beta[i]
            
        elif flip == 0:
            beta[i] = beta[i]

        #beta.append(1*np.arctan2(np.tan(tF)*np.sin(tB),1))
        i += 1
        '''
        i += 1

    for a,b in zip(alpha,beta):
        x.append(np.cos(b)*np.cos(a))
        y.append(np.cos(b)*np.sin(a))
        z.append(np.sin(b))
    
    #'tis just that simple
    return x,y,z,alpha,beta

# use z position to get vertical speed
def descentSpeed(pos_data,time_data):

    vY = np.diff(pos_data)/np.diff(time_data)
    j = 0
    for speed in vY:
        if math.isinf(speed):
            vY[j] = (vY[j+1]+vY[j-1])/2  

        j += 1
    
    return vY

def ecc(majAxis,minAxis):
    # gets eccentricity for given csv file
    ecc = []
    for (maj,min) in zip(majAxis,minAxis):
        ecc.append(np.sqrt(1-((maj/2)**2)/((min/2)**2)))

    return ecc

# make unit vector
def makeUnit(x,y,z):

    x1 = []
    y1 = []
    z1 = []
    for xVal, yVal, zVal in zip(x,y,z):
        mag = np.sqrt(xVal**2 + yVal**2 + zVal**2)
        x1.append(xVal/mag)
        y1.append(yVal/mag)
        z1.append(zVal/mag)

    return x1,y1,z1

def normalVector(time,ectry,x,y,z):
    # will output vector normal to surface of leaf
    # x, y, z, ecc should be of normalized length

    # make vector unit length 
    '''
    k = 0
    for xVal, yVal, zVal in zip(x,y,z):
        mag = np.sqrt(xVal**2 + yVal**2 + zVal**2)
        x[k] = xVal/mag
        y[k] = yVal/mag
        x[k] = zVal/mag
        k += 1
    '''
    #xNorm, yNorm, zNorm = []
    
    ## notes:
    # min value: vector in yz plane, orthogonal to yz of existing vector (x=0)
    # max value: vector in xz plane, orthogonal to xz of existing vector (y=0)
    # some sort of interpolation between those vector values (sinusoid? cubic?)
    # z should probably never really be zero based on how samaras fall

    # find minima (indices)
    ectryMin = -ectry
    minima,_ = find_peaks(ectryMin,width=20,distance = 30,prominence = 0.0035)
    
    # find maxima (indices)
    maxima,_ = find_peaks(ectry,width = 20,distance=30,prominence=0.0035)
    '''
    for i in range(len(maxima)):
        if i >= 1:
            if np.abs(ectry[maxima[i]] - ectry[minima[i-1]]) < 0.006:
                 ectry2 = np.delete(ectry,[minima[i-1],maxima[i]])
                 ectry = ectry2
    '''
    # find x,z for minima vector
    tMin = time[minima]
    xMin = x[minima]
    yMin = y[minima]
    zMin = z[minima]
    
    xOrientMin = []
    yOrientMin = []
    zOrientMin = []
    # create points of known vector output
    for xInd,yInd,zInd in zip(xMin,yMin,zMin):
        xOrientMin.append(-zInd)
        yOrientMin.append(0)
        zOrientMin.append(xInd)

    # find y,z for maxima vector
    tMax = time[maxima]
    xMax = x[maxima]
    yMax = y[maxima]
    zMax = z[maxima]

    xOrientMax = []
    yOrientMax = []
    zOrientMax = []
    # create points of known vector output
    for xInd,yInd,zInd in zip(xMax,yMax,zMax):
        xOrientMax.append(0)
        yOrientMax.append(-zInd)
        zOrientMax.append(yInd)
    
    
    # join datasets (assumes there is always a max first and that 
    # maxes and mins always alternate)
    xOrient = []
    yOrient = []
    zOrient = []
    tOrient = []
    for j in range(len(minima)):
        tOrient.append(tMax[j])
        tOrient.append(tMin[j])

        xOrient.append(xOrientMax[j])
        xOrient.append(xOrientMin[j])

        yOrient.append(yOrientMax[j])
        yOrient.append(yOrientMin[j])

        zOrient.append(zOrientMax[j])
        zOrient.append(zOrientMin[j])

        if j+1 < len(minima) or j+1 < len(maxima):
            # stop loop once alternation is no longer possible (two repeating minima/maxima)
            for j in range(len(minima) - 1):  # Subtract 1 to prevent going out of bounds
                if minima[j] < maxima[j+1] and minima[j+1] < maxima[j+1]:
                    break
                # for j in range(len(maxima) - 2):  # Subtract 2 to prevent going out of bounds
                elif maxima[j+1] < minima[j+1] and maxima[j+2] < minima[j+1]:
                    break

        # if j+1 < len(minima) and j+1 < len(maxima):
        #    # stop loop once alternation is no longer possible (two repeating minima/maxima)
        #    if minima[j] < maxima[j+1] and minima[j+1] < maxima[j+1]:
        #        break
        #    elif maxima[j+1] < minima[j+1] and maxima[j+2] < minima[j+1]:
        #        break

    
        
    # draw smooth curve between points (using PchipInterpolator):
    tNew = time[0:np.where(time == tOrient[-1])[0][0]+1]
    '''
    fX = interp1d(tOrient,xOrient, kind = 'nearest', fill_value='extrapolate', bounds_error=False)
    fY = interp1d(tOrient,yOrient, kind = 'nearest', fill_value='extrapolate', bounds_error=False)
    fZ = interp1d(tOrient,zOrient, kind = 'nearest', fill_value='extrapolate', bounds_error=False)
    '''
    fX = PchipInterpolator(tOrient,xOrient, extrapolate=True)
    fY = PchipInterpolator(tOrient,yOrient, extrapolate=True)
    fZ = PchipInterpolator(tOrient,zOrient, extrapolate=True)

    xCont = fX(tNew)
    yCont = fY(tNew)
    zCont = fZ(tNew)

    return minima,maxima,tOrient,xOrient,yOrient,zOrient,tNew,xCont,yCont,zCont

# resample vectors to match data together
# use smoothed data, should output 1 new time vector and x,y,z with matching dimensions
def lengthMatch(time1,time2,x,y,z):
    # time1 = front view
    # time2 = bottom view


    
    totalSize1 = len(time1)
    totalSize2 = len(time2)

    # check which time series has data latest, resample to 1200 datapts
    if time1[totalSize1-1] > time2[totalSize2-1]:
        n = totalSize1
        tNew3 = np.linspace(time1[0],time1[totalSize1-1],n)
    else:
        n = totalSize2
        tNew3 = np.linspace(time2[0],time2[totalSize2-1],n)

    f_xNew = interp1d(time2,x,'linear',bounds_error=False,fill_value='extrapolate')
    xNew3 = f_xNew(tNew3)

    f_yNew = interp1d(time1,y,bounds_error=False,fill_value='extrapolate')
    yNew3 = f_yNew(tNew3)

    f_zNew = interp1d(time1,z,bounds_error=False,fill_value='extrapolate')
    zNew3 = f_zNew(tNew3)


    

    return tNew3,xNew3,yNew3,zNew3

def findTransition(time,alpha):
    tROI = np.where(time>0.35)[0]
    searchTime = time[tROI]
    searchAlpha = alpha[tROI]

    indBegin = tROI[0]
    # pick moment that samara begins second full rotation (2nd flip of alpha)
    count = 0
    tTransition = 0
    searchInd = 0
    i = 0
    for t,a in zip(searchTime,searchAlpha):
        if i > 1:
            if np.abs(a - searchAlpha[i-2]) > 6:
                count += 1
            if count == 2:
                tTransition = t
                searchInd = i
                break
        i += 1

    # marker to evaluate transition being reached
    if count == 0:
        tTransition = -1

    ind = searchInd+indBegin

    return tTransition, ind