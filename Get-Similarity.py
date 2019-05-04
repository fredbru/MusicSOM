import numpy as np
from matplotlib import pyplot as plt
import math
import editdistance


def getEuclideanDistance(A, B):
    """Returns norm-2 of a 1-D numpy array for CPU computation.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    x = (A.flatten()-B.flatten())
    return math.sqrt(np.dot(x, x.T))


def getDirectedSwapDistance(A, B):
    # Directed swap: swap but for rhythms with variable #of onsets. Uses binary rhythms.
    binaryA = np.ceil(A)
    binaryB = np.ceil(B)
    # aKick, aSnare, aClosed, aOpen, aTom = splitKitParts(binaryA)
    # bKick, bSnare, bClosed, bOpen, bTom = splitKitParts(binaryB)

def getEditDistance(A,B):
    # get edit distance no velocity
    binaryA = np.ceil(A)
    binaryB = np.ceil(B)
    aKick, aSnare, aClosed, aOpen, aTom = splitKitParts(binaryA)
    bKick, bSnare, bClosed, bOpen, bTom = splitKitParts(binaryB)
    combinedEditDistance = editdistance.eval(aKick,bKick) + editdistance.eval(aSnare+bSnare) + \
                           editdistance.eval(aClosed+bClosed) + editdistance.eval(aOpen,bOpen) + \
                           editdistance.eval(aTom+bTom)
    print(combinedEditDistance)

def splitKitParts(groove):
    kick = groove[:,0]
    snare = groove[:,1]
    closed = groove[:,2]
    open = groove[:,3]
    tom = groove[:,4]
    return kick, snare, closed, open, tom

def calculateMonoSwapDistance(partA, partB):
    if partA.sum() > partB.sum():
        pass

def testMonoEditDistance():
    A = np.array([1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    B = np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
    print(editdistance.eval(A, B)) #should = 12

    son = np.array([1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0])
    shiko = np.array([1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0])
    print(editdistance.eval(son, shiko)) #should = 2


evalGrooves = np.load('Eval-matricies.npy')
evalNames = np.load('Eval-names.npy')

for i in range(evalNames.shape[0]):
    if 'Blues CH1 a' in evalNames[i]:
        A = evalGrooves[i]
    if 'Blues CH1 b' in evalNames[i]:
        B = evalGrooves[i]

print(getEditDistance())