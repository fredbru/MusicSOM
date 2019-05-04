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
    grooveArray = np.empty([len(hits), 4])
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

def getGroovesFromBundle(grooveDom):
    """ For each palette file, get a grooves from within a bundle. Extract hits and
    groove names into arrays, then put grooves and names into two separate lists
    :param grooveDom: groove object from .xml
    :return:
    """
    bundleNode = getGrooveBundleNode(grooveDom)
    grooveBundle = getGrooveNodes(bundleNode)
    grooveList = []
    grooveNames=[]
    for i in range(len(grooveBundle)):
        newGroove = getGrooveFromNode(grooveBundle[i])
        grooveLength = newGroove.lengthInBeats
        hits = newGroove.getAudibleHits()
        grooveArray = getCollapsedGrooveArray(hits, grooveLength)

        # round to semiquavers (0.125)
        grooveArray[:,1] = grooveArray[:,1]*16
        grooveArray[:,1] = grooveArray[:,1].round(decimals=0)/16

        roundedGroove = grooveArray
        grooveList.append(roundedGroove)
        grooveNames.append(newGroove.name)
    return grooveList, grooveNames

def makeBundleFeatures(grooveBundle, featureLength):
    """ Generate feature vectors for all grooves within one bundle file. Don't use collapsed features.
    This function is not recommended for generating vectors for the SOM - the collapsed features perform better in
    terms of mapping quality and computation speed.
    :param grooveBundle: selected bundle of grooves
    :param featureLength: length of feature vector for one groove (640)
    :return:
    """
    timeIndex = np.arange(0, 4, 0.0625).reshape((64, 1))  # 64 demisemiquavers = 2 bars
    bundleFeatures = np.empty(shape=[0, featureLength])
    for i in range(len(grooveBundle)):
        empty = np.zeros([64, 9])  # for mixed set - only 9 kpieces (0-8)
        grooveMatrix = np.hstack([timeIndex, empty])
        groove = grooveBundle[i]

        groove = make2Bar(groove)
        for j in range(grooveBundle[i].shape[0]):
            groove = groove
            for k in range(0, 64):
                if grooveMatrix[k, 0] == groove[j, 1]:
                    kitPiecePosition = int(groove[j, 2] + 1)
                    grooveMatrix[k, kitPiecePosition] = groove[j, 2]
        # features stored as stack of vectors
        bundleFeatures = np.vstack((grooveMatrix.flatten(), grooveFeatures))
    return bundleFeatures

def makeCollapsedBundleFeatures(grooveBundle, featureLength):
    """
    Make feature vectors for one bundle of grooves. Collapse to 5 parts.
    :param grooveBundle: selected bundle of grooves
    :param featureLength: length of feature vector for one groove (320)
    :return: grooveFeatures -
    """
    timeIndex = np.arange(0, 4, 0.125).reshape((32, 1))  # 64 demisemiquavers = 2 bars
    bundleFeatures = np.empty(shape=[0, featureLength])
    for i in range(len(grooveBundle)):
        empty = np.zeros([32, 4])  # 5 kitpieces for collapsed bundle (no percussion)
        grooveMatrix = np.hstack([timeIndex, empty])
        groove = grooveBundle[i]

        groove = make2Bar(groove)
        for j in range(grooveBundle[i].shape[0]):
            groove = groove
            for k in range(0, 32):
                if grooveMatrix[k, 0] == groove[j, 1]:
                    kitPiecePosition = int(groove[j, 2] + 1)
                    grooveMatrix[k, kitPiecePosition] = groove[j, 2]
        # features stored as stack of vectors
        bundleFeatures = np.vstack((grooveMatrix.flatten(), bundleFeatures))
    return bundleFeatures

def makeAllFeatures(featureLength, numberOfBundles, grooves):
    """
    Generate array of feature vectors of all grooves to feed into SOM.
    :param featureLength: length of feature vector of 1 groove - 640 for 8 parts/semiquaver
    :param numberOfBundles: number of bundles of grooves
    :param grooves: name of numpy file being used
    :return: grooveFeatures
    """
    grooveFeatures = np.empty(shape=[0, featureLength])
    print("Making features....")
    for i in range(0, numberOfBundles):
        bundleFeatures = makeCollapsedBundleFeatures(grooves[i], featureLength)
        #print("Completed pallete:", i)
        grooveFeatures = np.vstack((grooveFeatures, bundleFeatures))
    return grooveFeatures

np.set_printoptions(suppress=True,precision=2)

path = "/home/fred/BFD/python/grooves/All-inc-percussion/"
grooveFileNames = getGrooveFileNames(path)

allGrooves = []
allGrooveNames =[]
paletteNames = []
evalList = []

numberOfBundles = len(grooveFileNames)
evalGrooves = []
with open("/home/fred/BFD/python/Similarity-Eval/eval-grooves-list.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        evalGrooves.append(row[0])
print(evalGrooves)

# Iterate through list of groove files in given directory, extract grooves
# in format for MusicSOM.py
for i in range(numberOfBundles):
    print(i)
    grooveDom = parse((path + grooveFileNames[i]))
    grooveList, bundleGrooveNames = getGroovesFromBundle(grooveDom)
    for j in range(len(bundleGrooveNames)):
        paletteNames.append(grooveFileNames[i][0:-8])
        #print(grooveFileNames[i][0:-8])
        #allGrooveNames.append(bundleGrooveNames[j])
        if bundleGrooveNames[j] in evalGrooves:
            print(bundleGrooveNames[j])
            allGrooveNames.append(bundleGrooveNames[j])
    allGrooves.append(grooveList)

featureLength = 160 #320 - collapsed semi mode
grooveFeatures = makeAllFeatures(featureLength, numberOfBundles, allGrooves)

#np.save("Eval.npy", grooveFeatures)
#np.save("Eval-names.npy", np.vstack([allGrooveNames,paletteNames]))
