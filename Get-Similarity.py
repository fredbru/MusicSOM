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

def getBinaryEditDistance(A,B):
    # get edit distance no velocity
    binaryA = np.ceil(A)
    binaryB = np.ceil(B)
    aKick, aSnare, aClosed, aOpen, aTom = splitKitParts(binaryA)
    bKick, bSnare, bClosed, bOpen, bTom = splitKitParts(binaryB)
    combinedEditDistance = editdistance.eval(aKick,bKick) + editdistance.eval(aSnare,bSnare) + \
                           editdistance.eval(aClosed,bClosed) + editdistance.eval(aOpen,bOpen) + \
                           editdistance.eval(aTom,bTom)
    print(combinedEditDistance)

def getVelocityEditDistance(A,B):
    # get edit distance no velocity
    aKick, aSnare, aClosed, aOpen, aTom = splitKitParts(A)
    bKick, bSnare, bClosed, bOpen, bTom = splitKitParts(B)
    combinedEditDistance = editdistance.eval(aKick,bKick) + editdistance.eval(aSnare,bSnare) + \
                           editdistance.eval(aClosed,bClosed) + editdistance.eval(aOpen,bOpen) + \
                           editdistance.eval(aTom,bTom)
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
    print(np.count_nonzero(highA == 1))

    featureWeighting = np.array([0.66, 0.86,0.068,0.266,-0.118]) #might need to switch signs on this (-/+)
    vectorA = np.hstack([losync_A, midD_A, hiD_A, hiness_A, hisynessA]*featureWeighting)
    vectorB = np.hstack([losync_B, midD_B, hiD_B, hiness_B, hisynessB]*featureWeighting)
    print(vectorA)
    print(vectorB)
    return getEuclideanDistance(vectorA, vectorB)



def getSyncopation(part):
    # From Longuet-Higgins  and  Lee 1984 metric profile.
    # for now, just normalise the syncopation by dividing by the largest number in dataset
    #The level  of  the  topmost  metrical  unit isa rbitrarily set equal to 0, and the level of any other unit is
    # assigned thevalue n-1, where n is the level of its parent unit in the rhythm - is this why you need a 5?
    # as of 13/5 - added velocity - just multiply by velocity at part. will be 1 anyway for binary
    salienceProfile = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1,
                       4, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5] # extra value for comparing forwards

    syncopation = 0
    for i in range(len(part)):
        if part[i] != 0:
            if part[(i+1)%32] == 0: #only syncopation when not followed immediately by another onset
                syncopation = syncopation + (abs(salienceProfile[i] - 5)*part[i]) #syncopation = difference in profile weights
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
    totalSyncopation = 0

    for i in range(len(lowA)):
        kickSync = findKickSync(lowA, midA, highA, i, salienceProfile)
        snareSync = findKickSync(lowA, midA, highA, i, salienceProfile)
        totalSyncopation += kickSync
        totalSyncopation += snareSync
        print(totalSyncopation)

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

def getPanteliFeatures(A):
    # 6 features:
    # Autocorellation: skewness, max amplitude, centroid, harmonicity of peaks
    # Theirs were 4 bar patterns
    # weighted syncopation (same as Gomez but with velocity)
    # symmetry of metrical profile
    # Denotes the ratio of the number of onsets in the second half of the pattern that appear in
    # exactly the same position in the first half of the pattern

    #symmetry: number of onsets in the same position for bars 1 and 2
    low,mid,high = splitKitParts3Ways(A)
    lowSym = getSymmetry1Part(low)
    midSym = getSymmetry1Part(mid)
    highSym = getSymmetry1Part(high)
    averageSymmetry = (lowSym + midSym + highSym)/3.0 #this works

    #syncopation - same as Gomez, using velocity though
    lowSync = getSyncopation(low)
    midSync = getSyncopation(mid)
    highSync = getSyncopation(high)
    averageSync = (lowSync + midSync + highSync)/3.0

    # autocorrelation features - sum across stream then calculate features
    lowC = getAutocorrelation(low)
    midC = getAutocorrelation(mid)
    highC = getAutocorrelation(high)

    autoCorrelationSum = (lowC+midC+highC)/3.0
    autoCorrelationMaxAmplitude = autoCorrelationSum.max() #max, summed and scaled between 0 and 1
    #print("Max Amplitude" + str(autoCorrelationMaxAmplitude))

    from scipy import stats
    autoCorrelationSkew = stats.skew(autoCorrelationSum)

    # import pandas as pd
    # print(pd.DataFrame(lowC).skew(axis=0))
    # print(pd.DataFrame(midC).skew(axis=0))
    # print(pd.DataFrame(highC).skew(axis=0))

    #centroid

    print(getCentroid(lowC))
    print(getCentroid(midC))
    print(getCentroid(highC))


    #harmonicity
    # find peaks.




def getCentroid(part):
    # weighted mean of all frequencies in the signal. add all periodicities then divide by total weight
    # weight = level of periodicity.
    # remove negative periodicities - not relevant and give a - answer which is weird and unhelpful.
    centroidSum = 0
    totalWeights = 0
    for i in range(len(part)):
        addition = part[i]*i
        if addition >= 0:
            totalWeights += part[i]
            centroidSum +=addition
    centroid = centroidSum / totalWeights
    return centroid



def getAutocorrelation(part):
    # the plotting looks right. now need to make sure I can get right features
    #result = np.correlate(part,part,mode='full')
    print(part)
    from pandas import Series
    from pandas.plotting import autocorrelation_plot


    import matplotlib.pyplot as plt
    plt.figure()
    ax = autocorrelation_plot(part)
    autocorrelation = ax.lines[5].get_data()[1]
    plt.plot(range(1,33),autocorrelation) #plots from 1 to 32 inclusive - autocorrelation starts from 1 not 0 - 1-32
    plt.show()
    return autocorrelation

def getSymmetry1Part(part):
    sym = 0
    part1 = part[0:16]
    part2 = part[16:32]
    for i in range(16):
        if part1[i] != 0 and part2[i] != 0:
            sym +=1
    print(sym)
    return sym

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
    if 'Heavy Metal OH3 FBar3' in evalNames[i]:
        A = evalGrooves[i]
    if 'Funk Ride a' in evalNames[i]:
        B = evalGrooves[i]
    C = evalGrooves[12]

# getBinaryEditDistance(A,B)
# getVelocityEditDistance(A,B)

getPanteliFeatures(B)