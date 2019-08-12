"""
    Parse groove bundle XMLs into numpy array format. Saves two numpy files - one an array of groove names, the other
    an matrix of stacked feature vectors for each groove, in same order as names file.
    Contains unused functions for plotting individual grooves and printing information about hits within a groove.
"""

import os
from xml.dom.minidom import parse
import numpy as np
from matplotlib import pyplot as plt
import csv

def addRelativePathToSystemPath(relPath):
    if __name__ == '__main__' and __package__ is None:
        from os import sys, path
        sys.path.append(path.join(path.dirname(path.abspath(__file__)), relPath))

addRelativePathToSystemPath("../shared")

from fx.bfd.groove import *
from fx.common.filesystem import *

def getHitInfo(hit):
    """
    Print info for each hit in a groove
    :param hit:
    :return:
    """
    # Slot index = kpslot
    # beats = position as a fraction of beat = pos /1000000.0
    print("Slot index = ", hit.slotIndex)
    print("Articulation ID = ", hit.articIndex)
    print("Beats = ", hit.beats)
    print("Velocity = ", hit.velocity)
    print("Muted = ", hit.muted)

def make2Bar(groove):
    """
    Convert length of an individual groove to 2 bars (makes all feature lengths equal)
    :param groove:
    :return:
    """
    if  groove[-1,1] < 6:
        x = groove
        groove[:,1] = groove[:,1] + 4
        newGroove = np.vstack([x, groove])
    return groove


def getCollapsedGrooveArray(hits, grooveLength):
    """ Create groove array as combination of hits. Combine open/splash/crash cymbals together, closed/ride together
    and toms together.
    :param hits:
    :param grooveLength:
    :return: grooveArray
    """
    grooveArray = np.zeros([len(hits), 4])
    openHiHatArtics = {0,1,2,3,6,7,19}
    closedHiHatGroupIndex = {6,7,8}
    crashIndex = 6
    kickIndex = 0
    snareIndex = 1
    hihatIndex = 2
    tomGroupIndex = {3,4,5}
    closedHiHatArtics = {10,11,14,15,16,17,18,20,8000}
    for i in range(len(hits)):

        grooveArray[i,0] = grooveLength
        grooveArray[i,1] = hits[i].beats
        grooveArray[i,2] = hits[i].velocity
        kitPieceSlot = int(hits[i].slotIndex)
        if kitPieceSlot == kickIndex:
            grooveArray[i,3] == 0
        if kitPieceSlot == snareIndex:
            grooveArray[i,3] =1
        # Split open and closed articulations into different categories.
        if kitPieceSlot == hihatIndex:
            if int(hits[i].articIndex) in openHiHatArtics:
                grooveArray[i,3] = 3
            else:
                grooveArray[i,3] = 2
        if kitPieceSlot in closedHiHatGroupIndex:
            grooveArray[i,3] = 2
        if kitPieceSlot == crashIndex:
            grooveArray[i,3] = 3
        if kitPieceSlot in tomGroupIndex:
            grooveArray[i,3] = 4
    return grooveArray

def getGrooveArray(hits, grooveLength):
    """ Get groove array as combination of hits without combining kit piece parts.
    Not recommended for generating features for SOM - collapsed features perform better in terms of mapping quality and
    computation speed
    :param hits:
    :param grooveLength:
    :return: grooveArray
    """
    grooveArray = np.empty([len(hits), 4])
    for i in range(len(hits)):
        grooveArray[i,0] = grooveLength
        grooveArray[i,1] = hits[i].beats
        grooveArray[i,2] = hits[i].velocity
        grooveArray[i,3] = hits[i].slotIndex
    return grooveArray

def plotGroove(grooveArray):
    """
    Plot an individual groove as kit piece against time
    :param grooveArray:
    :return:
    """
    beats, velocity, kpiece = np.hsplit(grooveArray, 3)
    plt.scatter(beats, kpiece)
    plt.grid(which="both")
    plt.show()

def getGrooveFileNames(path):
    """
    Get names of groove bundle files in a given directory
    :param path:
    :return:
    """
    grooveFileNames = os.listdir(path)
    return grooveFileNames

def getGroovesFromBundle(grooveFileName, grooveDom, evalGrooveNames, multiPalGrooveList,multiPalGrooveMatchingPalettes):
    """ For each palette file, get a grooves from within a bundle. Extract hits and
    groove names into arrays, then put grooves and names into two separate lists
    :param grooveDom: groove object from .xml
    :return:
    """
    bundleNode = getGrooveBundleNode(grooveDom)
    grooveBundle = getGrooveNodes(bundleNode)
    grooveList = []
    grooveNames=[]
    paletteNames=[]
    for i in range(len(grooveBundle)):
        newGroove = getGrooveFromNode(grooveBundle[i])

        if newGroove.name in evalGrooveNames:
            for k in range(len(multiPalGrooveList)):
                if newGroove.name == multiPalGrooveList[k]:
                    if multiPalGrooveMatchingPalettes[k] == grooveFileName:

                # print(grooveFileName) #if groove palette list index matches to the groove name list index
                # if grooveFileName in multiPalGrooveMatchingPalettes:
                #     for j in range(len(multiPalGrooveMatchingPalettes):
                #         if multiPalGrooveMatchingPalettes[j] = grooveFileName:

                        #print(grooveFileName)
                        grooveLength = newGroove.lengthInBeats
                        hits = newGroove.getAudibleHits()
                        grooveArray = getCollapsedGrooveArray(hits, grooveLength)

                        # round to semiquavers (0.125)
                        grooveArray[:,1] = grooveArray[:,1]*16
                        grooveArray[:,1] = grooveArray[:,1].round(decimals=0)/16

                        roundedGroove = grooveArray
                        grooveList.append(roundedGroove)
                        grooveNames.append(newGroove.name)
                        paletteNames.append(grooveFileName)
            if newGroove.name not in multiPalGrooveList:
                #print(newGroove.name)
                grooveLength = newGroove.lengthInBeats
                hits = newGroove.getAudibleHits()
                grooveArray = getCollapsedGrooveArray(hits, grooveLength)

                # round to semiquavers (0.125)
                grooveArray[:,1] = grooveArray[:,1]*16
                grooveArray[:,1] = grooveArray[:,1].round(decimals=0)/16

                roundedGroove = grooveArray
                grooveList.append(roundedGroove)
                grooveNames.append(newGroove.name)
                paletteNames.append(grooveFileName)
    return grooveList, grooveNames, paletteNames

def makeCollapsedBundleFeatures(groove, featureLength):
    """
    Make feature vectors for one bundle of grooves. Collapse to 5 parts.
    :param grooveBundle: selected bundle of grooves
    :param featureLength: length of feature vector for one groove (320)
    :return: grooveFeatures -
    """
    timeIndex = np.arange(0, 8, 0.25).reshape((32, 1))  # 32 semiquavers = 2 bars
    grooveMatrix = np.zeros([32, 5])
    #print(grooveBundle[i])
    #print(grooveBundle[i].shape)
    for j in range(groove.shape[0]): #for # of hits in groove
        #put velocity value at time index of groove array in time slot in groove matrix
        timePosition = int(groove[j,1]*4)
        kitPiecePosition = int(groove[j, 3])
        if timePosition < 32:
            grooveMatrix[timePosition, kitPiecePosition] = groove[j,2]
    # features stored as stack of vectors
    features = grooveMatrix.flatten()
    return features, grooveMatrix

def makeAllFeatures(featureLength, flatAllPaletteNames, flatAllGrooves):
    """
    Generate array of feature vectors of all grooves to feed into SOM.
    :param featureLength: length of feature vector of 1 groove - 640 for 8 parts/semiquaver
    :param numberOfBundles: number of bundles of grooves
    :param grooves: name of numpy file being used
    :return: grooveFeatures
    """
    allGrooveFeatures =[]
    allGrooveMatricies =[]
    #grooveFeatures = np.empty(shape=[0, featureLength])
    print("Making features....")
    for i in range(0, len(flatAllPaletteNames)):
        #print(grooves[i])
        features, matrix = makeCollapsedBundleFeatures(flatAllGrooves[i], featureLength)
        #print("Completed pallete:", i)
        #allGrooveFeatures = np.vstack((grooveFeatures, features))
        allGrooveFeatures.append(features)
        allGrooveMatricies.append(matrix)
    return allGrooveFeatures, allGrooveMatricies

np.set_printoptions(suppress=True,precision=2)

path = "/home/fred/BFD/python/grooves/most-inc-percussion/"
grooveFileNames = getGrooveFileNames(path)

allGrooves = []
allGrooveNames =[]
paletteNames = []
evalList = []

numberOfBundles = len(grooveFileNames)

multiPalGrooveList = ['Pop HH a','Pop HH a Crash','Pop HH FBar2','Pop CH1 c','Pop CH1 c Crash','Pop CH1 FBar3','Pop CH2 a','Pop CH3 a','Pop CH3 FBar3','Pop Ride1 b Crash','Pop Ride1 FBar4','Pop Ride2 b Crash',
                      'Reggae Grooves 1','Reggae Grooves 6','Reggae Grooves 7','Reggae Grooves 11','Reggae Grooves 19','Reggae Grooves 20','Reggae Grooves Fill 1','Reggae Grooves Fill 3','Reggae Grooves Fill 5',
                      'Jungle 1','Jungle 2','Jungle 3','Jungle 8','Jungle 9','Jungle 13','Jungle 14','Jungle 19','Jungle 21','Jungle Fill 1','Jungle Fill 2',
                      'Jazz Brushes 2','Jazz Brushes 5','Jazz Brushes 18','Jazz Brushes 23','Jazz Brushes Fill 5',
                      'Rock7','Rock 17','Rock 25','Rock Ride1 FBar3',
                      'Funk CH1 a','Funk CH1 b','Funk CH1 c','Funk CH1 FBar1','Funk Ride a','Funk Ride FBar1','Funk CH a','Funk CH c','Funk CH d',
                      'Rock CH1 ba','Rock CH1 FBar1','Rock OH2 b','Rock OH2 FBar1', 'N Country Intro a']

multiPalGrooveMatchingPalettes = ['Pop V2','Pop V2','Pop V1','Pop V2','Pop V2','Pop V3','Pop V3','Pop V2','Pop V2','Pop V2','Pop V2','Pop V2',
                                  'Reggae Grooves V2','Reggae Grooves V2','Reggae Grooves V2','Reggae Grooves V2','Reggae Grooves V2','Reggae Grooves V2','Reggae Grooves V2','Reggae Grooves V2','Reggae Grooves V2',
                                  'HHM Jungle V3','HHM Jungle V1','HHM Jungle V2','HHM Jungle V2','HHM Jungle V2','HHM Jungle V1','HHM Jungle V3','HHM Jungle V1','HHM Jungle V3','HHM Jungle V2','HHM Jungle V3',
                                  'Jazz Brushes V2','Jazz Brushes V2','Jazz Brushes V2','Jazz Brushes V2','Jazz Brushes V1',
                                  'Steve Ferrone Rock V2','AFJ Rock','Steve Ferrone Rock V2', 'Rock V3',
                                  'Funk V3','Funk V3','Funk V3','Funk V3','Essential Funk','Funk V3','Essential Funk','Essential Funk','Funk V2',
                                  'Essential Rock','Rock V1','Rock V1','Essential Rock', 'New Country V1']
evalGrooveNames = []
with open("/home/fred/BFD/python/Similarity-Eval/eval-grooves-list.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        evalGrooveNames.append(row[0])
# Iterate through list of groove files in given directory, extract grooves
# in format for MusicSOM.py
k = 0
allPaletteNames = []
for i in range(numberOfBundles):
    #print(i)
    grooveDom = parse((path + grooveFileNames[i]))
    grooveList, bundleGrooveNames, paletteNames = getGroovesFromBundle(grooveFileNames[i][0:-8],grooveDom, evalGrooveNames, multiPalGrooveList,multiPalGrooveMatchingPalettes)
    for j in range(len(bundleGrooveNames)):
        # print(grooveList[0], bundleGrooveNames[0])

        #print(grooveFileNames[i][0:-8])
        #allGrooveNames.append(bundleGrooveNames[j])
        if bundleGrooveNames[j] in evalGrooveNames:
            allGrooveNames.append(bundleGrooveNames[j])
            k=k+1
            #print(bundleGrooveNames[j])
    allGrooves.append(grooveList)
    allPaletteNames.append(paletteNames)

flatAllGrooves = []
for sublist in allGrooves:
    for item in sublist:
        flatAllGrooves.append(item)

flatAllPaletteNames = []
for sublist in allPaletteNames:
    for item in sublist:
        flatAllPaletteNames.append(item)

featureLength = 160 #320 - collapsed semi mode
#print(evalGrooveNames, "\n", allGrooveNames)

grooveFeatures, grooveMatricies = makeAllFeatures(featureLength, flatAllPaletteNames, flatAllGrooves)

print(len(grooveFeatures))
print(len(flatAllGrooves))
print(len(flatAllPaletteNames))
print(len(grooveMatricies))
print(grooveMatricies[i].shape)

print(flatAllGrooves[51])
print(grooveFeatures[51])
print(grooveMatricies[51])
print(allGrooveNames[51],flatAllPaletteNames[51])

# for i in range(len(allGrooveNames)):
#     print(allGrooveNames[i], paletteNames[i])
#np.save("Eval-features.npy", grooveFeatures)
#np.save("Eval-matricies.npy", grooveMatricies)
#np.save("Eval-names.npy", allGrooveNames)
#np.save("Eval-palette-names.npy", flatAllPaletteNames)