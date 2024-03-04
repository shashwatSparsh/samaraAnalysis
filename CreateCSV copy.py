import seedAnalysis as sa
import numpy as np

## SCRIPT for creating CSV from raw video data
# ensure destination folder for files is empty before running
# change parameters in sa.bottomView to ensure bottom view is in frame
# destination folder = f'{date}_Data'
    # subfolders can be added to this and output filenames will be the same
    # e.g. f'{date}_Data/FastTransition/{output filename}'
# output is two CSV files: f'{date}_Tests/00{id} Drop_bottom.csv' and f'{date}_Tests/00{id} Drop_front.csv'
    # handled by sa.vidAnalysis.analyze()

# set ID list
ids = (3,7,47)

# set date video was taken
date = 20230427

filename = f'003 Drop.mp4'

vid = sa.vidAnalysis(filename)
print("This is the reduced Name " + vid.reducedName)
# print(vid.name)
#print(vid.cap)

lower = (0, 0, 0)
upper = (75, 95*255/100, 95*255/100)

lower2= (210/2,0,0)
upper2 = (180, 95*255/100, 95*255/100)


# run function to split and analyze front view, then restart video
vid.analyze(lower,upper,lower2,upper2,'front',date)

# vid.reopen()

'''
# set HSV range for bottom view
lower = (195/2,0,50*255/100)
upper = (360/2,55*255/100,80*255/100)

lower2 = (0,0,50*255/100)
upper2 = (150/2,55*255/100,80*255/100)

# reanalyze video, but for bottom view
vid.analyze(lower,upper,lower2,upper2,'bottom',date)
'''

'''
# start looping through IDs
for id in ids:
    id = int(id)

    # create 3-digit text of id and make filename
    if id < 10:
        filename = f'{date}_Tests/00{id} Drop.mp4'
    elif id < 100:
        filename = f'{date}_Tests/0{id} Drop.mp4'
    elif id >= 100:
        filename = f'{date}_Tests/{id} Drop.mp4'

    print(filename)

    # initialize vidAnalysis object
    vid = sa.vidAnalysis(filename)
    


    # set HSV range values for front view
    lower = (0, 0, 0)
    upper = (75, 95*255/100, 95*255/100)

    lower2= (210/2,0,0)
    upper2 = (180, 95*255/100, 95*255/100)

    # run function to split and analyze front view, then restart video
    vid.analyze(lower,upper,lower2,upper2,'front',date)
    vid.reopen()
    
    # set HSV range for bottom view
    lower = (195/2,0,50*255/100)
    upper = (360/2,55*255/100,80*255/100)

    lower2 = (0,0,50*255/100)
    upper2 = (150/2,55*255/100,80*255/100)

    # reanalyze video, but for bottom view
    vid.analyze(lower,upper,lower2,upper2,'bottom',date)





'''