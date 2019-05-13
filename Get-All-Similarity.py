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

def getBinaryEditDistance(A,B):
    # get edit distance no velocity
    binaryA = np.ceil(A)
    binaryB = np.ceil(B)
    aKick, aSnare, aClosed, aOpen, aTom = splitKitParts(binaryA)
    bKick, bSnare, bClosed, bOpen, bTom = splitKitParts(binaryB)
    combinedEditDistance = edcalc.eval(aKick,bKick) + edcalc.eval(aSnare,bSnare) + \
                           edcalc.eval(aClosed,bClosed) + edcalc.eval(aOpen,bOpen) + \
                           edcalc.eval(aTom,bTom)
    return combinedEditDistance

def getVelocityEditDistance(A,B):
    # get edit distance no velocity
    aKick, aSnare, aClosed, aOpen, aTom = splitKitParts(A)
    bKick, bSnare, bClosed, bOpen, bTom = splitKitParts(B)
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

def getGomezFeatureDistance(A,B):
    # binary for now. low syncopation, mid density, high density, hiness, hisyness
    # low = kick, mid = snare and tom, hi = cymbals. summed simply - not trying to model loudness - two onsets at one
    # time (eg snare and a tom hit) just count as one
    # at the moment hisyness parameter dominates - much larger values than the rest - so divided it by 20 (the max value
    # in the dataset


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

    hiness_A = (float(hiD_A)/float(totalD_A))/10.0
    hiness_B = (float(hiD_B)/float(totalD_B))/10.0

    if hiD_A != 0:
        hisynessA = float(getSyncopation(highA))/float(np.count_nonzero(highA == 1))
    else:
        hisynessA = 0

    if hiD_B != 0:
        hisynessB = float(getSyncopation(highB)) / float(np.count_nonzero(highB == 1))
    else:
        hisynessB = 0

    featureWeighting = np.array([-0.66, -0.86,-0.068,-0.266,+0.118]) #might need to switch signs on this (-/+)
    vectorA = np.hstack([midD_A, hiD_A, hiness_A, losync_A, hisynessA]*featureWeighting)
    vectorB = np.hstack([midD_B, hiD_B, hiness_B, losync_B, hisynessB]*featureWeighting)
    print(vectorA)
    print(vectorB)
    return getEuclideanDistance(vectorA, vectorB)

def getSyncopation(part):
    # From Longuet-Higgins  and  Lee 1984 metric profile. theres a -5 in gomez's code for who knows why.
    # for now, just normalise the syncopation by dividing by the largest number in dataset
    #The level  of  the  topmost  metrical  unit is arbitrarily set equal to 0, and the level of any other unit is
    # assigned thevalue n-1, where n is the level of its parent unit in the rhythm - is this why you need a 5?
    salienceProfile = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1,
                       4, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5] # extra value for comparing forwards

    syncopation = 0
    for i in range(len(part)):
        if part[i] == 1:
            if part[(i+1)%32] == 0: #only syncopation when not followed immediately by another onset
                syncopation = syncopation + abs(salienceProfile[i] - 5) #syncopation = difference in profile weights
    return syncopation

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
    print(totalSyncopation)
    return totalSyncopation

def findKickSync(low, mid, high, i, salienceProfile):
    # find instances  when kick syncopates against hi hat/snare on the beat. looking for kick proceeded by another hit
    # on a weaker metrical position
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


allEuclideanDistances = []
allHammingDistances = []
allEditDistances = []
allVelocityEditDistances = []
allGomezDistances = []
allWitekDistances = []
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
        bineditdistance = getBinaryEditDistance(a,b)
        velocityEditdistance = getVelocityEditDistance(a,b)
        gomezDistance = getGomezFeatureDistance(a,b)
        witekDistance = getWitekSyncopationDistance(a,b)

        allEuclideanDistances.append(euclideanRhythmDistance)
        allHammingDistances.append(hammingDistance)
        allEditDistances.append(bineditdistance)
        allVelocityEditDistances.append(velocityEditdistance)
        allGomezDistances.append(gomezDistance)
        allWitekDistances.append(witekDistance)

        # print(aName + "  " + bName + '    Euclidean = ' + str(euclideanRhythmDistance) + "Hamming = "
        #       + str(hammingDistance))
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

plt.figure()
plt.bar(np.arange(100), allVelocityEditDistances)
plt.title("Velocity Edit distances")

plt.figure()
plt.bar(np.arange(100),allGomezDistances)
plt.title("Gomez Feature Distances")

plt.figure()
plt.bar(np.arange(100),allWitekDistances)
plt.title("Witek Feature Distances")
plt.show()
