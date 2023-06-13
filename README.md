README for Samara Analysis

All dates in YYYYMMDD format

Libraries Req'd:
    OpenCV (cv2)
    matplotlib.pyplot
    numpy
    tsmoothie
    scipy.interpolate
    scipy.signal
    pandas
    csv
    os
    glob
    math


Video Filenaming:
    {id} Drop.mp4
if multiple drops:
    {id} Drop {drop number}.mp4

Tests are organized by folders with the following structure:
    {date of test}_Tests/{intermediate category}/{id} Drop {drop number}.mp4

{intermediate category} = Fast Transition, Slow Transition, or some other feature of interest

General Workflow:
1. Generate full sampleProperties.csv file using propertyAnalysis.py
2. Use createCSV.py to iterate through IDs and analyze videos
3. Use CSVanalysis.py to read CSV files and generate plots
    The bulk of analysis is done with CSVAnalysis.py
    Also note that transition time analysis was done by hand by looking at the alpha output graph in conjunction with theta_front and seeing where the graph became periodic

Notes on seedAnalysis.py library:
-Use vidAnalysis.outline for videos without clear color definition
-Use vidAnalysis.outlineColor for videos with clear color definition
-Frame info in vidAnalysis.bottomView needs to be adjusted for each individial test day
-sa.clearfiles() and sa.findTransition are nonfunctional but have good baseline information
-For sa.vectorize_front and sa.vectorize_bottom, the only outputs that matter are angle outputs, all others   are legacy and don't matter for final analysis (they create inaccurate vectors)
-sa.polar2cartesian is the only vector output that is accurate, uses input from sa.vectorize_front and sa.vectorize_bottom
-sa.normalVector has a lot of legacy outputs used for debugging, tCont, xCont, yCont, zCont are actual vector output. All others are some iteration of the maxima and minima

This compilation was meant to generate code for 3 videos. It will require significant adaptation to get it to work with different structures and different test setups