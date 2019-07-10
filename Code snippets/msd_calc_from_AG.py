# python script to calculate the second and fourth moments of the displacements
import csv
import time
import math
import numpy as np

# for timing test to compare with Mathematica
start = time.time()


# read in three column trajectory (t,x,y) in csv format
def Read_Three_Column_File(file_name):
    with open(file_name, 'r') as f_input:
        csv_input = csv.reader(f_input, delimiter=' ', skipinitialspace=True)
        t = []
        x = []
        y = []
        for cols in csv_input:
            t.append(float(cols[0]))
            x.append(float(cols[1]))
            y.append(float(cols[2]))

    return t, x, y


t, x, y = Read_Three_Column_File('c2_COM_xy_coord.dat')


def compute_msd(timeStep, trajLength, xCoords, yCoords):
    sum2 = 0
    sum4 = 0
    count = 0

    # generate list of each timestep to be used for moving time origin
    timeOrigins = list(range(int(trajLength-timeStep)))

    for i in timeOrigins:
        xInit = xCoords[i]
        yInit = yCoords[i]
        xDt = xCoords[i+timeStep]
        yDt = yCoords[i+timeStep]
        disp = math.sqrt((xDt-xInit)**2 + (yDt-yInit)**2)
        sum2 += disp**2
        sum4 += disp**4
        count += 1

    return sum2/count, sum4/count


# set the max lag time to half of trajectory length
maxTime = max(t)
maxLag = int(np.floor(maxTime / 2))

# initialize arrays
msd = np.zeros(maxLag)
fourMoment = np.zeros(maxLag)

# outputs three column data {lag time, msd, fourth moment}
# right now it prints to screen or forwarded to output file
for tau in range(0, maxLag):
    msd[tau], fourMoment[tau] = compute_msd(tau, maxTime, x, y)
    print("%.6f %.6f %.6f" % (tau, msd[tau], fourMoment[tau]))


end = time.time()
print(end-start)
