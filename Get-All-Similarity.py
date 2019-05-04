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
    x = (grooveA.flatten()-grooveB.flatten())
    return math.sqrt(np.dot(x, x.T))

def getHammingDistance(grooveA, grooveB):
    # Same as euclidean, without velocity
    binaryA = np.ceil(grooveA).flatten()
    binaryB = np.ceil(grooveB).flatten()
    return np.count_nonzero(binaryA != binaryB)

def getDirectedSwapDistance(grooveA, grooveB):
    # Directed swap: swap but for rhythms with variable #of onsets. Uses binary rhythms.
    binaryA = np.ceil(grooveA).flatten()
    binaryB = np.ceil(grooveB).flatten()

    pass


allEuclideanDistances = []
allHammingDistances = []
j=0
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
        euclideanRhythmDistance = getEuclideanDistance(grooveA,grooveB)
        hammingDistance = getHammingDistance(grooveA,grooveB)

        allEuclideanDistances.append(euclideanRhythmDistance)
        allHammingDistances.append(hammingDistance)

        print(grooveAName + "  " + grooveBName + '    Euclidean = ' + str(euclideanRhythmDistance) + "Hamming = "
              + str(hammingDistance))
        j=j+1
plt.figure()
plt.hold(True)
plt.bar(np.arange(100),allEuclideanDistances)
plt.title("Euclidean Distances")

plt.figure()
plt.bar(np.arange(100),allHammingDistances)
plt.title("Hamming Distances")
plt.show()