import numpy as np
import cv2
import pandas

#%% Reading Data

# for reading csv of masses and scan of samaras
# takes grid layout of scans such as in seedLayoutRaw.png and outputs dimensions for each seed
    # seedLayoutRaw has 100 seeds, in 5 rows of 20 seeds
# outputs several images with id value, mass, and area

# read csv of masses
    # cols = ids,masses
# The CSV name in question should be changed based on what CSV is being read
kaiCSV = 'Mass.csv'
saucyCSV = 'fastMass.csv'
dims = pandas.read_csv(saucyCSV, header = None)
dims.rename(columns = {0:'id',1:'mass'},inplace=True)

# read in raw scan file
# Set Which ImageLayout to read from
kaiLayout = 'seedLayourRaw.png'
saucyLayout = 'newSeedLayoutRaw.jpg'

samples = cv2.imread(saucyLayout)

# Change distance/pixel based on which image you are reading
# input distance/pixel value of file
kaiRes = .02/1.11 #in/px
saucyRes = 1.72/510 #in/px
res = saucyRes

#%% Variable Initialization
#cv2.imshow('raw image',samples)

# convert image to black and white for masking
samplesBW = cv2.cvtColor(samples, cv2.COLOR_BGR2GRAY)

# take all nonwhite pixels as 'true'
mask = cv2.inRange(samplesBW,0,254)

# width and height of rectangle encompassing ONE seed
w = 87
h = 228

# code for testing w and h values
im1 = mask[0:h,0:w]
im2 = mask[0:h,w+1:w+1+w]
im3 = mask[h:h+1+h,w+1:w+1+w]
#cv2.imshow('crop',im1)
#cv2.imshow('crop2',im2)
#cv2.imshow('im3',im3)

# initialize variables
area = []
span = []
chord = []
# shift variable accounts for some error
shift = 5
idText = 'text'
ImList = []

# initialize fonts
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
textColor = (0,0,0)

#%% Setting IDS
# Set the linspace based on your sample size
# E.G. for 13 seeds: np.linespace(1,13,13) and so on soforth
# set ids
id = np.linspace(1,100,100)

# erode and dilate mask to eliminate tails of samaras
mask = cv2.erode(mask,None,iterations = 2)
mask = cv2.dilate(mask,None,iterations = 2)


# This loop should be adjusted based on the number of rows and columns
# Kai's Original Dataset had 100 seeds with 5 rows and 20 columns.
# Fast Data set includes 13 seeds with 1 row and 13 columns.
kaiRows = 5;
kaiCols = 20;
saucyRows = 1;
saucyCols = 13;

numRows = saucyRows
numColumns = saucyCols
k = 0
# iterate over number of rows
for i in range(numRows):
    # draw vertical line to divide cols
    cv2.line(samples,(0,i*h),(samples.shape[1],i*h),(0,0,0),2)

    # iterate over row
    for j in range(numColumns):

        # get current seed in image
        im = mask[i*(h):(i+1)*h - 1 , j*(w) + shift:(j+1)*w - 1 + shift]

        # get color version of same frame
        color = samples[i*(h):(i+1)*h - 1 , j*(w) + shift:(j+1)*w - 1 + shift]
        
        ImList.append(color)

        # find contours
        conts, _ = cv2.findContours(im,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ensure that only largest contour (seed outline) is in contours
        if len(conts) != 0:
            bigCont = max(conts,key = cv2.contourArea)
            area.append(cv2.contourArea(bigCont)*res*res)
        
        # fit ellipse for span and chord analysis
        ((x,y),(c,b),a) = cv2.fitEllipse(bigCont)
        ellip = (x,y),(c,b),a

        # show ellipse if desired
        #cv2.ellipse(color,ellip,color=(255,0,0),thickness=2)

        # use major and minor axis lengths for span and chord (converted to mm)
        span.append(b*res*25.4)
        chord.append(c*res*25.4)

        # get id text
        if id[k] < 10:
            idText = f'00{int(id[k])}'
        elif id[k] >= 10 and id[k] < 100:
            idText = f'0{int(id[k])}'
        elif id[k] >= 100:
            idText = f'{int(id[k])}'

        # reset some font variables
        fontSize = 0.7
        fontThickness = 2
        # line 1 of text = id

        # determine textsize and gap variables for coordinate
        textsize = cv2.getTextSize(idText, cv2.FONT_HERSHEY_COMPLEX, fontSize, fontThickness)[0]
        gap = textsize[1] + 5

        # coord of text (lower left corner)
        y = (i)*(h) + gap
        x = j*w + 5

        # write text
        cv2.putText(samples,idText,(x,y),cv2.FONT_HERSHEY_COMPLEX,
                    fontSize,
                    textColor,
                    fontThickness)
        
        fontsize = 1
        
        # perform a similar operation for writing mass and area values on image
            # uncomment cv2.putText lines to write these values
        mass = dims['mass'][k]
        textsize = cv2.getTextSize(idText, font, fontSize, fontThickness)[0]
        gap = textsize[1] + 5
        # line 2, mass
        y = (i*h) + 2*gap + 175

        
        """ cv2.putText(samples,f'{mass}g',(x,y),font,
                    fontSize,
                    textColor,
                    fontThickness) """
        
        # line 3, area
        a = area[k]
        y = (i*h) + 3*gap + 175
        """ cv2.putText(samples,f'{a:.2f}sqin',(x,y),font,
                    fontSize,
                    textColor,
                    fontThickness) """

        # writing div. lines for rows
        cv2.line(samples,(j*w,0),(j*w,samples.shape[0]),(0,0,0),2)
        
        # some debugging image shows below
        #cv2.drawContours(color,[bigCont],contourIdx=0,color = (0,255,0), thickness=3)
        #cv2.imshow(f'im{i},{j}',color)

        # print row and iterate total id count (k)
        print(f'row {i+1}, col {j+1}')
        k += 1
    
# create edge lines:
maxH = samples.shape[0]
maxW = samples.shape[1]
cv2.line(samples,(maxW,0),(maxW,maxH),(0,0,0),2)
cv2.line(samples,(0,maxH),(maxW,maxH),(0,0,0),2)

#%% Data Frame Construction
# Note, the Aspect Ratio and Loading Computations are incorrect and are changed manually
# concatenate pandas dataframe with new data (in mm and mm^2)

dims['area'] = area
dims['loading'] = dims['mass']/dims['area']*1550
dims['area'] = dims['area']*645.2
dims['span'] = span
dims['chord'] = chord
dims['aspect'] = dims['area']/dims['chord']

cv2.imshow('w/text',samples)

#%% Change these File paths EVERYTIME working with a new data set.
# write files
# Change name of property generation file
kaiName = 'sampleProperties.csv'
saucyName = 'fastSampleProperties.csv'

filepath = saucyName
dims.to_csv(filepath)

# change filename here as desired
kaiImageName = 'seedLayoutMarkupNoMass.png'
saucyImageName = 'fastSeedLayoutMarkupNoMass.png'
cv2.imwrite(saucyImageName,samples)
cv2.waitKey(0)
cv2.destroyAllWindows()