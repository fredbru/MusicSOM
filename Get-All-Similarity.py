import numpy as np
from matplotlib import pyplot as plt
import math
import csv
import editdistance as edcalc

allGrooves = np.load('Eval-matricies.npy')
allNames = np.load('Eval-names.npy')

def getEuclideanDistance(a, b):
    """Returns norm-2 of a 1-D numpy array for CPU computation.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    x = (a.flatten()-b.flatten())
    return math.sqrt(np.dot(x, x.T))

def getHammingDistance(a, b):
    # Same as euclidean, without velocity
    binaryA = np.ceil(a).flatten()
    binaryB = np.ceil(b).flatten()
    return np.count_nonzero(binaryA != binaryB)

def getDirectedSwapDistance(a, b):
    # Directed swap: swap but for rhythms with variable #of onsets. Uses binary rhythms.
    binaryA = np.ceil(a).flatten()
    binaryB = np.ceil(b).flatten()
    pass

def getEditDistance(a,b):
    # get edit distance no velocity
    binaryA = np.ceil(a)
    binaryB = np.ceil(b)
    aKick, aSnare, aClosed, aOpen, aTom = splitKitParts(binaryA)
    bKick, bSnare, bClosed, bOpen, bTom = splitKitParts(binaryB)
    combinedEditDistance = edcalc.eval(aKick,bKick) + edcalc.eval(aSnare,bSnare) + \
                           edcalc.eval(aClosed,bClosed) + edcalc.eval(aOpen,bOpen) + \
                           edcalc.eval(aTom,bTom)
    return combinedEditDistance

def splitKitParts(groove):
    kick = groove[:,0]
    snare = groove[:,1]
    closed = groove[:,2]
    open = groove[:,3]
    tom = groove[:,4]
    return kick, snare, closed, open, tom

allEuclideanDistances = []
allHammingDistances = []
allEditDistances = []
j=0
with open("/home/fred/BFD/python/Similarity-Eval/eval-pairings.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        aName = row[0]
        bName = row[1]
        for i in range(len(allNames)):
            if allNames[i] == aName:
                a = allGrooves[i]
            if allNames[i] == bName:
                b = allGrooves[i]
        euclideanRhythmDistance = getEuclideanDistance(a,b)
        hammingDistance = getHammingDistance(a,b)
        editdistance = getEditDistance(a,b)

        allEuclideanDistances.append(euclideanRhythmDistance)
        allHammingDistances.append(hammingDistance)
        allEditDistances.append(editdistance)

        print(aName + "  " + bName + '    Euclidean = ' + str(euclideanRhythmDistance) + "Hamming = "
              + str(hammingDistance))
        j=j+1

plt.figure()
plt.hold(True)
plt.bar(np.arange(100),allEuclideanDistances)
plt.title("Euclidean Distances")

plt.figure()
plt.bar(np.arange(100),allHammingDistances)
plt.title("Hamming Distances")

plt.figure()
plt.bar(np.arange(100),allEditDistances)
plt.title("Edit Distances")
plt.show()
