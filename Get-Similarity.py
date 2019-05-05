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

def getGomezFeatureDistance(A,B):
    # binary for now. low syncopation, mid density, high density, hiness, hisyness
    # low = kick, mid = snare and tom, hi = cymbals. summed simply - not trying to model loudness - two onsets at one
    # time (eg snare and a tom hit) just count as one

    binaryA = np.ceil(A)
    binaryB = np.ceil(B)

    lowA, midA, highA = splitKitParts3Ways(binaryA)
    lowB, midB, highB = splitKitParts3Ways(binaryB)

    losync_A = getSyncopation(lowA)
    losync_B = getSyncopation(lowB)

    midD_A = getDensity(midA)
    midD_B = getDensity(midB)

    hiD_A = getDensity(highA)
    hiD_B = getDensity(highB)

    totalD_A = getDensity(binaryA)
    totalD_B = getDensity(binaryB)

    hiness_A = float(hiD_A)/float(totalD_A)
    hiness_B = float(hiD_B)/float(totalD_B)

    hisynessA = float(getSyncopation(highA))/float(np.count_nonzero(highA == 1))
    hisynessB = float(getSyncopation(highB))/float(np.count_nonzero(highB == 1))
    print(hisynessA,hiness_A,hiD_A,midD_A,losync_A)
    print(lowA)
    print(highA)


def getSyncopation(part):
    # From Longuet-Higgins  and  Lee 1984 metric profile. i+1 = R, i=N
    salienceProfile = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1,
                       4, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5] # extra value for comparing forwards
    syncopation = 0

    for i in range(len(part)):
        if part[i] == 1.0:
            if part[(i+1)%32] == 0: #only syncopation when not followed immediately by another onset
                # ???if salienceProfile[i] < salienceProfile[i+1]: #weight of note before must be smaller than weight of note after
                syncopation = syncopation + (salienceProfile[i]-5) #syncopation = difference in profile weights
                print(syncopation)
    return syncopation

def getWitekSyncopationDistance(A,B):
    # Get syncopation distance for loop based on Witek 2014
    # salience profile different to gomez/longuet higgins + lee
    # Combines insrument weighting for cross-instrument syncopation. For now - just considering witek's 3 syncopation
    # types with 3 kit parts. Later look at implementing 4 parts: add open hi hat, and syncopation with hi hat off pulse
    # then look at adding velocity somehow (use velocity of syncopating part?)

    salienceProfile = [0,-3,-2,-3,-1,-3,-2,-3,-1,-3,-2,-3,-1,-3,-2,-3,
                       0,-3,-2,-3,-1,-3,-2,-3,-1,-3,-2,-3,-1,-3,-2,-3,0]
    binaryA = np.ceil(A)
    binaryB = np.ceil(B)

    lowA, midA, highA = splitKitParts3Ways(binaryA)
    lowB, midB, highB = splitKitParts3Ways(binaryB)
    totalSyncopation = 0

    for i in range(len(lowA)):
        kickSync = findKickSync(lowA, midA, highA, i, salienceProfile)
        snareSync = findKickSync(lowA, midA, highA, i, salienceProfile)
        totalSyncopation += kickSync
        totalSyncopation += snareSync

    for i in range(len(lowB)):
        kickSync = findKickSync(lowB, midB, highB, i, salienceProfile)
        snareSync = findKickSync(lowB, midB, highB, i, salienceProfile)
        totalSyncopation += kickSync
        totalSyncopation += snareSync
    return totalSyncopation

def findKickSync(low, mid, high, i, salienceProfile):
    # find instances  when kick syncopates against hi hat/snare on the beat
    kickSync = 0
    if low[i] == 1.0:
        if high[(i+1)%32] == 1.0 and mid[(i+1)%32] == 1.0:
            if salienceProfile[i+1] > salienceProfile[i]: #if hi hat is on a stronger beat - syncopation
                kickSync = 2
        elif mid[(i+1)%32] == 1.0:
            if salienceProfile[i + 1] > salienceProfile[i]: #my own estimate - more syncopated that hi hat on pulse too (?)
                kickSync = 3
        elif high[(i+1)%32] == 1.0:
            if salienceProfile[i + 1] > salienceProfile[i]:
                kickSync = 5
    return kickSync

def findSnareSync(low, mid, high, i, salienceProfile):
    # find instances  when snare syncopates against hi hat/kick on the beat
    snareSync = 0
    if mid[i] == 1.0:
        if high[(i+1)%32] == 1.0 and low[(i+1)%32] == 1.0:
            if salienceProfile[i + 1] > salienceProfile[i]:
                snareSync = 1
        elif high[(i+1)%32] == 1.0:
            if salienceProfile[i+1] > salienceProfile[i]: #if hi hat is on a stronger beat - syncopation
                snareSync = 5
        elif low[(i+1)%32] == 1.0:
            if salienceProfile[i + 1] > salienceProfile[i]: # my best guess - kick without hi hat
                snareSync = 1
    return snareSync

def findHiHatSync(low, mid, high, i, salienceProfile):
    # find instances  when hiaht syncopates against snare/kick on the beat. this is my own adaptation of Witek 2014
    # may or may not work. currently doesn't consider velocity or open hi hats
    hihatSync = 0
    if high[i] == 1.0:
        if low[(i+1)%32] == 1.0:
            if salienceProfile[i+1] > salienceProfile[i]:
                hihatSync = 1 ### bit of a guess - maybe should be 0.5?
        elif mid[(i+1)%32] == 1.0:
            if salienceProfile[i + 1] > salienceProfile[i]:
                hihatSync =1 ### another guess
    return hihatSync




def getDensity(part):
    # get density for one or more kit parts
    numSteps = part.size
    numOnsets = np.count_nonzero(part == 1)
    density = float(numOnsets)/float(numSteps)
    return density

def splitKitParts3Ways(groove):
    kick = groove[:,0]
    snare = groove[:,1]
    closed = groove[:,2]
    open = groove[:,3]
    tom = groove[:,4]

    low = kick
    mid = np.clip(snare + tom, 0, 1)
    high = np.clip(closed + open, 0, 1)

    return low, mid, high


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
    if 'Funk CH1 a' in evalNames[i]:
        A = evalGrooves[i]
    if 'Blues CH1 b' in evalNames[i]:
        B = evalGrooves[i]

print(getWitekSyncopationDistance(A,B))