import numpy as np
from matplotlib import pyplot as plt
import math
import csv
import editdistance as edcalc
from scipy import stats
from scipy.signal import find_peaks
import os

allGrooves = np.load('Eval-matricies.npy')
allNames = np.load('Eval-names.npy')

def getEuclideanDistance(a, b):
    """Returns norm-2 of a 1-D numpy array for CPU computation.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    x = (a.flatten()-b.flatten())
    return math.sqrt(np.dot(x, x.T))

def getEuclideanRhythmDistance(a, b):
    """Returns norm-2 of a 1-D numpy array for CPU computation.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    lA,mA,hA = splitKitParts3Ways(a)
    a = np.hstack([lA,mA,hA])
    lB,mB,hB = splitKitParts3Ways(b)
    b = np.hstack([lB,mB,hB])

    x = (np.power(a,1.0/5.0).flatten()-np.power(b,1.0/5.0).flatten())
    return math.sqrt(np.dot(x, x.T))

def getHammingDistance(a, b):
    # Same as euclidean, without velocity
    binaryA = np.ceil(a)
    binaryB = np.ceil(b)
    lA,mA,hA = splitKitParts3Ways(a)
    a = np.hstack([lA,mA,hA])
    lB,mB,hB = splitKitParts3Ways(b)
    b = np.hstack([lB,mB,hB])
    return np.count_nonzero(a != b)

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
    # -0.66, -0.86,-0.068,-0.266,+0.118


    binaryA = np.ceil(A)
    binaryB = np.ceil(B)

    lowA, midA, highA = splitKitParts3Ways(binaryA)
    lowB, midB, highB = splitKitParts3Ways(binaryB)

    lowAV, midAV, highAV = splitKitParts3Ways(A)
    lowBV, midBV, highBV = splitKitParts3Ways(B)


    losync_A = getSyncopation(lowAV)
    losync_B = getSyncopation(lowBV)


    midD_A = getDensity(midAV)
    midD_B = getDensity(midBV)

    hiD_A = getDensity(highAV)
    hiD_B = getDensity(highBV)

    totalD_A = getDensity(A)
    totalD_B = getDensity(B)

    hiness_A = (float(hiD_A)/float(totalD_A))/10.0
    hiness_B = (float(hiD_B)/float(totalD_B))/10.0

    if hiD_A != 0.:
        hisynessA = float(getSyncopation(highAV))/float(np.count_nonzero(highA == 1))
    else:
        hisynessA = 0

    if hiD_B != 0.:
        hisynessB = float(getSyncopation(highBV)) / float(np.count_nonzero(highB == 1))
    else:
        hisynessB = 0

    featureWeighting = np.array([-0.58, -0.146,-0.88,-0.0194,+0.110]) #might need to switch signs on this (-/+)
    vectorA = np.hstack([midD_A, hiD_A, hiness_A, losync_A, hisynessA]*featureWeighting)
    vectorB = np.hstack([midD_B, hiD_B, hiness_B, losync_B, hisynessB]*featureWeighting)
    return getEuclideanDistance(vectorA, vectorB)

def getSyncopation(part):
    # From Longuet-Higgins  and  Lee 1984 metric profile.
    # for now, just normalise the syncopation by dividing by the largest number in dataset
    #The level  of  the  topmost  metrical  unit isa rbitrarily set equal to 0, and the level of any other unit is
    # assigned thevalue n-1, where n is the level of its parent unit in the rhythm - is this why you need a 5?
    # as of 13/5 - added velocity - just multiply by velocity at part. will be 1 anyway for binary
    salienceProfile = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1,
                       4, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5] # extra value for comparing forwards

    syncopation = 0.0
    for i in range(len(part)):
        if part[i] != 0:
            if part[(i+1)%32] == 0.0: #only syncopation when not followed immediately by another onset
                syncopation = float(syncopation + (abs(salienceProfile[i] - 5)*pow(part[i],0.2))) #syncopation = difference in profile weights
    return syncopation

def getDensity(part):
    # get density for one or more kit parts
    numSteps = part.size
    numOnsets = np.count_nonzero(np.ceil(part) == 1)
    averageVelocity = np.mean(np.nonzero(part))
    if np.isnan(averageVelocity):
        averageVelocity = 0.0
    density = pow(averageVelocity,0.2) * float(numOnsets)/float(numSteps)
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

    lowA, midA, highA = splitKitParts3Ways(A)
    lowB, midB, highB = splitKitParts3Ways(B)
    totalSyncopation = 0
    totalSyncopation = 0

    for i in range(len(lowA)):
        kickSync = findKickSync(lowA, midA, highA, i, salienceProfile)
        snareSync = findSnareSync(lowA, midA, highA, i, salienceProfile)
        totalSyncopation += kickSync
        totalSyncopation += snareSync

    for i in range(len(lowB)):
        kickSync = findKickSync(lowB, midB, highB, i, salienceProfile)
        snareSync = findSnareSync(lowB, midB, highB, i, salienceProfile)
        totalSyncopation += kickSync
        totalSyncopation += snareSync
    return totalSyncopation

def findKickSync(low, mid, high, i, salienceProfile):
    # find instances  when kick syncopates against hi hat/snare on the beat. looking for kick proceeded by another hit
    # on a weaker metrical position
    kickSync = 0
    if low[i] != 0.0:
        if high[(i+1)%32] != 0.0 and mid[(i+1)%32] != 0.0:
            if salienceProfile[i+1] > salienceProfile[i]: #if hi hat is on a stronger beat - syncopation
                kickSync = 2.0 * pow(low[i],0.2)
        elif mid[(i+1)%32] != 0.0:
            if salienceProfile[i + 1] > salienceProfile[i]: #my own estimate - more syncopated that hi hat on pulse too (?)
                kickSync = 3.0 * pow(low[i],0.2)
        elif high[(i+1)%32] != 0.0:
            if salienceProfile[i + 1] > salienceProfile[i]:
                kickSync = 5.0 * pow(low[i],0.2)
    return kickSync

def findSnareSync(low, mid, high, i, salienceProfile):
    # find instances  when snare syncopates against hi hat/kick on the beat
    snareSync = 0
    if mid[i] != 0.0:
        if high[(i+1)%32] != 0.0 and low[(i+1)%32] != 0.0:
            if salienceProfile[i + 1] > salienceProfile[i]:
                snareSync = 1.0 * pow(mid[i],0.2)
        elif high[(i+1)%32] != 0.0:
            if salienceProfile[i+1] > salienceProfile[i]: #if hi hat is on a stronger beat - syncopation
                snareSync = 5.0 * pow(mid[i],0.2)
        elif low[(i+1)%32] != 0.0:
            if salienceProfile[i + 1] > salienceProfile[i]: # my best guess - kick without hi hat
                snareSync = 1.0 * pow(mid[i],0.2)
    print(snareSync)
    return snareSync

def findHiHatSync(low, mid, high, i, salienceProfile):
    # find instances  when hiaht syncopates against snare/kick on the beat. this is my own adaptation of Witek 2014
    # may or may not work. currently doesn't consider velocity or open hi hats
    hihatSync = 0
    if high[i] != 0.0:
        if low[(i+1)%32] != 0.0:
            if salienceProfile[i+1] > salienceProfile[i]:
                hihatSync = 1 ### bit of a guess - maybe should be 0.5?
        elif mid[(i+1)%32] != 0.0:
            if salienceProfile[i + 1] > salienceProfile[i]:
                hihatSync =1 ### another guess
    return hihatSync

def getPanteliFeatureDistance(A,B):
    return getEuclideanDistance(getPanteliFeatures(A),getPanteliFeatures(B))

def getPanteliFeatures(A):
    # 6 features:
    # Autocorellation: skewness, max amplitude, centroid, harmonicity of peaks
    # Theirs were 4 bar patterns
    # weighted syncopation (same as Gomez but with velocity)
    # symmetry of metrical profile
    # Denotes the ratio of the number of onsets in the second half of the pattern that appear in
    # exactly the same position in the first half of the pattern

    #symmetry: number of onsets in the same position for bars 1 and 2
    binaryA = np.ceil(A)
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
    autoCorrelationSkew = stats.skew(autoCorrelationSum)
    autoCorrelationCentroid = getCentroid(autoCorrelationSum)
    harmonicity = getHarmonicity(autoCorrelationSum)


    weighting = np.array([0.5,0.48,0.31,0.15,0.7,6.0])
    panteliFeatureSet = np.multiply(weighting,np.array([autoCorrelationSkew,autoCorrelationMaxAmplitude,autoCorrelationCentroid,
                                                averageSync,harmonicity,averageSymmetry]))
    #print(panteliFeatureSet)
    return np.nan_to_num(panteliFeatureSet)

def getHarmonicity(part):
    # need to 0 out negative periodicities.
    # this appears to be working - close to 1 for perfect harmonicity, close to 0 for zero harmonicity
    alpha = 0.15
    for i in range(len(part)):
        if part[i] < 0:
            part[i]=0
    peaks = np.asarray(find_peaks(part)) #weird syntax due to 2.x/3.x compatibility issues here
    peaks = peaks[0] + 1#peaks = lags
    inharmonicSum = 0.0
    inharmonicPeaks = []
    for i in range(len(peaks)):
        remainder1 = 16%peaks[i]
        #remainder2 = 16%peaks[i]
        if remainder1 > 16*0.15 and remainder1 < 16*0.85:
            inharmonicSum += pow(part[peaks[i]-1],0.2) #add magnitude of inharmonic peaks
            inharmonicPeaks.append(part[i])

    harmonicity = math.exp((-0.25*len(peaks)*inharmonicSum/float(part.max())))
    return harmonicity

def getCentroid(part):
    # weighted mean of all frequencies in the signal. add all periodicities then divide by total weight
    # weight = level of periodicity.
    # remove negative periodicities - not relevant and give a - answer which is weird and unhelpful.
    centroidSum = 0
    totalWeights = 0
    part = np.power(part, 0.2)
    for i in range(len(part)):
        addition = part[i]*i
        if addition >= 0:
            totalWeights += part[i]
            centroidSum +=addition
    if totalWeights != 0:
        centroid = centroidSum / totalWeights
    else:
        centroid = 16
    return centroid

def getAutocorrelation(part):
    # the plotting looks right. now need to make sure I can get right features
    #result = np.correlate(part,part,mode='full')
    from pandas import Series
    from pandas.plotting import autocorrelation_plot


    import matplotlib.pyplot as plt
    plt.figure()
    ax = autocorrelation_plot(part)
    autocorrelation = ax.lines[5].get_data()[1]
    plt.plot(range(1,33),autocorrelation) #plots from 1 to 32 inclusive - autocorrelation starts from 1 not 0 - 1-32
    #plt.show()
    plt.cla()
    plt.clf()
    plt.close()
    return autocorrelation

def getSymmetry1Part(part):
    sym = 0.0
    part1 = part[0:16]
    part2 = part[16:32]
    for i in range(16):
        if part1[i] != 0 and part2[i] != 0:
            sym += abs(pow(part1[i],0.2)-pow(part2[i],0.2))
    return sym

allEuclideanDistances = []
allHammingDistances = []
allEditDistances = []
allVelocityEditDistances = []
allGomezDistances = []
allWitekDistances = []
allPanteliDistances = []
j=0
with open("/home/fred/BFD/python/Similarity-Eval/eval-pairings-reduced.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        aName = row[0]
        bName = row[1]
        for i in range(len(allNames)):
            if allNames[i] == aName:
                a = allGrooves[i]
            if allNames[i] == bName:
                b = allGrooves[i]
        euclideanRhythmDistance = getEuclideanRhythmDistance(a,b)
        hammingDistance = getHammingDistance(a,b)
        #bineditdistance = getBinaryEditDistance(a,b)
        #velocityEditdistance = getVelocityEditDistance(a,b)
        gomezDistance = getGomezFeatureDistance(a,b)
        witekDistance = getWitekSyncopationDistance(a,b)
        #panteliDistance = getPanteliFeatureDistance(a,b)

        allEuclideanDistances.append(euclideanRhythmDistance)
        allHammingDistances.append(hammingDistance)
        # allEditDistances.append(bineditdistance)
        # allVelocityEditDistances.append(velocityEditdistance)
        allGomezDistances.append(gomezDistance)
        allWitekDistances.append(witekDistance)
        #allPanteliDistances.append(panteliDistance)

        # print(aName + "  " + bName + '    Euclidean = ' + str(euclideanRhythmDistance) + "Hamming = "
        #       + str(hammingDistance))
        j=j+1

ratingFiles = os.listdir('/home/fred/BFD/python/MusicSOM/ratings')

ratingFiles.sort() #alphabetize
ratingFiles = ratingFiles[10:90]
scoresByPair = np.zeros([80,21])
print(scoresByPair.shape)
def getResultsPerPair(scoresByPair):
    # tested, works
    for i in range(80): #just non-repeated ones
        with open(('/home/fred/BFD/python/MusicSOM/ratings/' + ratingFiles[i])) as csvfile:
            reader = csv.reader(csvfile, delimiter=",") #so far 21 participants
            j = 0
            for row in reader:
                if row[0] in (None, ""):
                    pass
                elif row[0] == "file_keys":
                    pass
                elif j < 21:
                    scoresByPair[i,j] = row[1]
                    j+=1
    return scoresByPair

def getRepeatedPairs(subject):
    first10 = np.round(subject[0:10] *10.0)+1
    last10 = np.round(subject[80:90] *10.0)+1
    return np.vstack([first10,last10])

scoresBySubject = getResultsPerPair(scoresByPair).T
# plt.figure()
# plt.bar(np.arange(80),np.mean(scoresBySubject, axis=0))
#
# plt.figure()
# plt.hold(True)
# plt.bar(np.arange(80),allEuclideanDistances)
# plt.title("Euclidean Distances")
#
# plt.figure()
# plt.bar(np.arange(80),allHammingDistances)
# plt.title("Hamming Distances")

meanScores = 1.0 - np.mean(scoresBySubject, axis=0)
print(np.array(allHammingDistances).shape)
print(np.array(allPanteliDistances).shape)
coeff1, p1 = stats.spearmanr(meanScores,np.array(allHammingDistances))
coeff2, p2 = stats.spearmanr(meanScores,np.array(allEuclideanDistances))
#coeff3, p3 = stats.spearmanr(meanScores,np.array(allPanteliDistances))
coeff4, p4 = stats.spearmanr(meanScores,np.array(allWitekDistances))
coeff5, p5 = stats.spearmanr(meanScores,np.array(allGomezDistances))

print("Hamming correlation: ", coeff1,p1)
print("Euclidean correlation: ", coeff2,p2)
#print("Panteli correlation: ", coeff3,p3)
print("Witek correlation: ", coeff4,p4)
print("Gomez correlation: ",coeff5,p5)


# plt.figure()
# plt.bar(np.arange(80),allGomezDistances)
# plt.title("Gomez Feature Distances")
#
# plt.figure()
# plt.bar(np.arange(80),allWitekDistances)
# plt.title("Witek Feature Distances")
#
# plt.figure()
# plt.bar(np.arange(80),allPanteliDistances)
# plt.title("Panteli Feature Distances")

plt.show()
