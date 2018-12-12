# Main script for calculating Self-Organising Map of music data (e.g BFD grooves).
# Take nunmpy groove and name files (output from Groove-Parse) and map/cluster using SOM.
# Arguments
# 1: Name of numpy data file and corresponding names file (e.g Erskine-Top30 for Erskine-Top30.npy and
#   Erskine-Top30Names label file)
# 2: x dimension of SOM
# 3: y dimension of SOM
# 4: number of iterations

import numpy as np
import csv
import sys
def addRelativePathToSystemPath(relPath):
    if __name__ == '__main__' and __package__ is None:
        from os import sys, path
        sys.path.append(path.join(path.dirname(path.abspath(__file__)), relPath))

addRelativePathToSystemPath("../MusicSOM")

from MusicSOM.MusicSOM import *

import random
import timeit


##
# Make map of SOM nodes to superimpose on plot
# @param dim: dimensions of SOM

def makeNodeLocations(dim):
    xNodePoints = np.arange(dim+1)
    yNodePoints = np.arange(dim+1)
    nodePoints = np.vstack([xNodePoints,yNodePoints])
    return nodePoints


##
# Get winners of SOM without plotting - for use when running on servers without matplotlib support.
#
def getWinners(features, names, som, paletteNames):
    itemIndex = range(len(names))
    weightMap = {}
    im = 0
    winners = []
    for x, g, p, t in zip(features, names, paletteNames, itemIndex):
        w = som.winner(x)
        weightMap[w] = im
        winners.append([g, p, w[0], w[1]])
        im = im+1
    return winners

##
# Plot som output as labelled winner nodes, save output to csv file. Also plots U-Matrix underneath winner map.
# @param features - array of feature vectors.
# @param labels - array of groove names in order.
# @param som - minisomWeighted SOM object
# @param showUMatrix. If True shows U-Matrix in plot. U-Matrix currently broken (not matching up to clustering)
#

def plotWinners(features, names, som, paletteNames, showUMatrix=False):
    from matplotlib import pyplot as plt

    plt.figure()
    itemIndex = range(len(names))
    weightMap = {}
    im = 0
    winners = []
    if showUMatrix == True:
        plt.pcolor(som.distance_map().T)
    for x, g, p, t in zip(features, names, paletteNames, itemIndex):  # scatterplot
        w = som.winner(x)
        weightMap[w] = im
        offsetX = random.uniform(-0.2, 0.2) #small x and y offsets to stop labels being plotted on top of each other
        offsetY = random.uniform(-0.2, 0.2)
        plt.text(w[0] + offsetX +.5, w[1] + offsetY +.5, g,
                 color=plt.cm.gist_rainbow(t / 2), fontdict={'size': 7})

        #plt.annotate(l, (w[0] + offset, w[1]+offset))
        winners.append([g, p, w[0],w[1]])
        im = im + 1
    plt.axis([0, som.getweights().shape[0], 0, som.getweights().shape[1]])
    nodes = np.indices((som.x,som.y)).reshape(2,-1)
    nx = list(nodes[0])
    nxOffset = [x+0.5 for x in nx]
    ny = list(nodes[1])
    nyOffset = [x+0.5 for x in ny]
    plt.scatter(nxOffset, nyOffset, 3) #plots SOM node positions as blue dots on grid
    plt.show(block=False)
    return winners

##
# Save SOM output mapping to .csv file
# @param winners - array groove labels and the coordinates of the winner nodes for the relevant groove
#
def saveSOM(winners):
    with open("SOM_Winners.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(winners)


np.set_printoptions(threshold=np.nan)

combinedLabels = np.load(sys.argv[1] + "Names.npy")
names = combinedLabels[0]
paletteNames = combinedLabels[1]
features = np.load(sys.argv[1] + ".npy")
features = features.astype(np.float32)
featureLength = features.shape[1]
print("Dataset = " + sys.argv[1])
print("Number of data items = " + str(features.shape[0]))
print("Feature length = " + str(features.shape[1]))
print("Iterations = " + sys.argv[4])
print("SOM size = " + sys.argv[2] + "x"+ sys.argv[3])


# minisom uses gaussian neighbourhood function. sigma = initial spread
som = MusicSOM(int(sys.argv[2]), int(sys.argv[3]), featureLength, sigma=0.3, learning_rate=0.5, perceptualWeighting=False)
som.random_weights_init(features)
trainStart = timeit.default_timer()

try:
    if sys.argv[5] == 'CPU':
        print("CPU selected")
        print("Start SOM Training...")
        som.trainCPU(features, int(sys.argv[4]))
    if sys.argv[5] == 'GPU':
        print("GPU selected")
        print("Start SOM Training...")
        som.trainGPU(features, int(sys.argv[4]))
except IndexError:
    print("No platform specified, using CPU")
    print("Start SOM Training...")
    som.trainCPU(features, int(sys.argv[4]))

from numpy import genfromtxt,array,linalg,zeros,apply_along_axis


np.save("SOM_Weights.npy",som.weights)

trainEnd = timeit.default_timer()
print('Training time = ' + str(trainEnd-trainStart))

winners = getWinners(features, names, som, paletteNames)
saveSOM(winners)

# from numpy import genfromtxt,array,linalg,zeros,apply_along_axis
#
# # reading the iris dataset in the csv format
# # (downloaded from http://aima.cs.berkeley.edu/data/iris.csv)
# data = genfromtxt('iris.csv', delimiter=',',usecols=(0,1,2,3))
# # normalization to unity of each pattern in the data
# data = apply_along_axis(lambda x: x/linalg.norm(x),1,data)
# ### Initialization and training ###
# som = MusicSOM(7,7,4,sigma=1.0,learning_rate=0.5)
# som.random_weights_init(data)
# #som.random_weights_init(data)
# print("Training...")
# som.trainCPU(data,100) # training with 100 iterations
# print("\n...ready!")
# #
# from pylab import plot,axis,show,pcolor,colorbar,bone
# bone()
# pcolor(som.distance_map().T) # distance map as background
# colorbar()
# # loading the labels
# target = genfromtxt('iris.csv',
#                     delimiter=',',usecols=(4),dtype=str)
# t = zeros(len(target),dtype=int)
# t[target == 'setosa'] = 0
# t[target == 'versicolor'] = 1
# t[target == 'virginica'] = 2
# # use different colors and markers for each label
# markers = ['o','s','D']
# colors = ['r','g','b']
# for cnt,xx in enumerate(data):
#  w = som.winner(xx) # getting the winner
#  # palce a marker on the winning position for the sample xx
#  plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
#    markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
# axis([0,som.weights.shape[0],0,som.weights.shape[1]])
# show() # show the figure