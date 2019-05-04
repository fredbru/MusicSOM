import numpy as np
from matplotlib import pyplot as plt
import math
import csv

allGrooves = np.load('Eval-matricies.npy')
allNames = np.load('Eval-names.npy')

def getEuclideanDistance(grooveA, grooveB):
    """Returns norm-2 of a 1-D numpy array for CPU computation.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    x = grooveA.flatten()-grooveB.flatten()
    return math.sqrt(np.dot(x, x.T))

with open("/home/fred/BFD/python/Similarity-Eval/eval-pairings.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        grooveAName = row[0]
        grooveBName = row[1]
        for i in range(len(allNames)):
            if allNames[i] == grooveAName:
                grooveA = allGrooves[i]
            if allNames[i] == grooveBName:
                grooveB = allGrooves[i]
        euclideanRhythmDistance = 1/(1+getEuclideanDistance(grooveA,grooveB))
        print(grooveAName + "  " + grooveBName + '    Euclidean distance = ' + str(euclideanRhythmDistance))