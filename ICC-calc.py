import numpy as np
import csv
import os
import pandas
import math, string, copy
np.set_printoptions(suppress=True)
from scipy import stats
ratingFiles = os.listdir('/home/fred/BFD/python/MusicSOM/Ratings-by-participant-reliable')

ratingFiles.sort()

allScores = np.zeros([21,80])
# tested, works
for i in range(21):
    with open(('/home/fred/BFD/python/MusicSOM/Ratings-by-participant-reliable/' + ratingFiles[i])) as csvfile:
        reader = csv.reader(csvfile, delimiter=",") #so far 21 reliable participants
        j = 0
        for row in reader:
            allScores[i,j] = row[0]
            j+=1

def icc(allScores, icc_type='icc2'):
    ''' Calculate intraclass correlation coefficient for data within
        Brain_Data class
    Args:
        icc_type: type of icc to calculate (icc: voxel random effect,
                icc2: voxel and column random effect, icc3: voxel and
                column fixed effect)

    Returns:
        ICC: (np.array) intraclass correlation coefficient

    '''
    Y = allScores.T
    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k-1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
                                X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc / n

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == 'icc1':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
        NotImplementedError("This method isn't implemented yet.")

    elif icc_type == 'icc2':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        ICC = (MSR - MSE) / (MSR + (k-1) * MSE + k * (MSC - MSE) / n)

    elif icc_type == 'icc3':
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        ICC = (MSR - MSE) / (MSR + (k-1) * MSE)

    return ICC

print(icc(allScores,icc_type='icc3'))
k2, p = stats.normaltest(allScores)