"""
    Script for generating plots of SOM output from Run-MusicSOM.py.
    Uses SOM_Winners.csv and SOM_Weights.npy files
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as matcolors
import matplotlib.lines as mlines

import csv
import random
import math

def addRelativePathToSystemPath(relPath):
    if __name__ == '__main__' and __package__ is None:
        from os import sys, path
        sys.path.append(path.join(path.dirname(path.abspath(__file__)), relPath))

addRelativePathToSystemPath("../shared")



from fx.bfd.groove import *
from fx.common.filesystem import *
from xml.dom.minidom import parse


def winners_map(csvfile, previousPalette, colour, i):
    metal = 'xkcd:blue'
    rock = 'xkcd:azure'
    jazz = 'xkcd:coral'
    reggae = 'xkcd:crimson'
    funk = 'xkcd:darkgreen'
    blues = 'xkcd:indigo'
    pop = 'xkcd:lime'
    fusion = 'xkcd:orange'
    dance = 'xkcd:gold'
    country = 'xkcd:brown'
    punk = 'xkcd:fuchsia'
    latin = 'xkcd:navy'
    hiphop = 'xkcd:maroon'


    plt.plot(0,0)
    path = "/home/fred/BFD/python/grooves/All-4-4/"

    reader = csv.reader(csvfile, delimiter=",")
    plt.plot(0,0)
    tags = ['Fill', 'Intro']

    for row in reader:
        x = int(row[2])
        y = int(row[3])
        #print(row[1])

        print(row[1])
        # print(x)
        # print(y)
        if row[1] == "Heavy Metal":
            colour = metal
        if row[1] == "Peter Erskine Rock":
            colour = rock
        if row[1] == "Smooth Jazz":
            colour = jazz
        if row[1] == "Reggae Grooves V1":
            colour = reggae
        if row[1] == "Funk V3":
            colour = funk
        if row[1] == "Chicago Blues":
            colour = blues
        if row[1] == "Pop V3":
            colour = pop
        if row[1] == "Glam Fusion":
            colour = fusion
        if row[1] == "HHM Jungle V1":
            colour = dance
        if row[1] == "Country 2Beat Shuffle":
            colour = country
        if row[1] == "Brooks Punk V1":
            colour = punk

        offsetx = random.uniform(-0.3,0.3)
        offsety = random.uniform(-0.3,0.3)
        #plt.text(x+0.5+offsetx,y+0.5+offsety, row[0],color=colour)
        if any(x in row[0] for x in tags):
            plt.scatter(x + 0.5 + offsetx, y + 0.5 + offsety, color=colour, marker=".", s=60)
        else:
            plt.scatter(x + 0.5 + offsetx, y + 0.5 + offsety, color=colour, marker="x",s=60)

        previousPalette =row[1]
        i=i+1
    # nodes = np.indices((16,16)).reshape(2,-1)
    # nx = list(nodes[0])
    # nxOffset = [x+0.5 for x in nx]
    # ny = list(nodes[1])
    # nyOffset = [x+0.5 for x in ny]
    # plt.scatter(nxOffset, nyOffset, 3) #plots SOM node positions as blue dots on grid
    # plt.show(block=False)
    danceKey = mpatches.Patch(color=dance, label='Jungle V1')
    jazzKey = mpatches.Patch(color=jazz, label='Smooth Jazz')
    rockKey = mpatches.Patch(color=rock, label='Peter Erskine Rock')
    metalKey = mpatches.Patch(color=metal, label='Heavy Metal')
    funkKey = mpatches.Patch(color=funk, label='Funk V3')
    bluesKey = mpatches.Patch(color=blues, label='Chicago Blues')
    popKey = mpatches.Patch(color=pop, label='Pop V3')
    countryKey = mpatches.Patch(color=country, label='Country 2 Beat Shuffle')



    regularKey = mlines.Line2D([],[],marker='x',label='Regular',color="k",linewidth=0.0)
    fillKey = mlines.Line2D([],[],marker='o',label='Fill',color="k",linewidth=0.0)

    plt.legend()

    plt.legend(handles=[danceKey, jazzKey, rockKey, metalKey,  funkKey, bluesKey, popKey
        , countryKey,regularKey,fillKey], loc="upper center",
               bbox_to_anchor=(0.5, -0.03),prop={'size': 10},ncol=4)
    plt.title("BFD 8 Palette SOM Output",size=12)

def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array for CPU computation.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    #return math.sqrt(np.dot(x, x.T))
    return np.linalg.norm(x)


##
# Returns the distance map of the weights as a heatmap.
# Each cell is the normalised sum of the distances between
# a neuron and its neighbours.
#

def distance_map(weights):
    """Returns the distance map of the weights.
    Each cell is the normalised sum of the distances between
    a neuron and its neighbours."""
    um = np.zeros((weights.shape[0], weights.shape[1]))
    it = np.nditer(um, flags=['multi_index'])
    while not it.finished:
        for ii in range(it.multi_index[0]-2, it.multi_index[0]+3):
            for jj in range(it.multi_index[1]-2, it.multi_index[1]+3):
                if (ii >= 0 and ii < weights.shape[0] and
                        jj >= 0 and jj < weights.shape[1]):
                    w_1 = weights[ii, jj, :]
                    w_2 = weights[it.multi_index]
                    um[it.multi_index] += fast_norm(w_1-w_2)

        it.iternext()
    um = um/um.max()

    winx = []
    winy = []
    with open("SOM_Winners_milc_final.csv") as csvfile:
     reader = csv.reader(csvfile, delimiter=",")
     for row in reader:
             winx.append(int(row[2]))
             winy.append(int(row[3]))
    pm = np.zeros((weights.shape[0], weights.shape[1]))
    it = np.nditer(um, flags=['multi_index'])

    while not it.finished:
        for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
            for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                if (ii >= 0 and ii < weights.shape[0] and
                        jj >= 0 and jj < weights.shape[1]):
                    if (ii in winx) and (jj in winy):
                        pm[it.multi_index] +=0.1
        it.iternext()

    pm=pm/pm.max()
    ustarm = um*pm
    ustarm = ustarm/ustarm.max()

    return um

weights = np.load("SOM_Weights_milc_final.npy")
colour = np.random.rand(3, 1)
previousPalette = 'none'
np.random.seed(40)
colour = np.random.rand(4)

plt.figure()
umatrix = np.flipud(distance_map(weights).T)
plt.imshow(umatrix, cmap="Greys", norm=matcolors.LogNorm(0.2, vmax=0.8), interpolation="bilinear",
           extent=[0,weights.shape[0],0,weights.shape[1]]) #need the .T for the distance map to correlate with the weights
with open("SOM_Winners_milc_final.csv") as csvfile:
    winners_map(csvfile, previousPalette, colour, 0)
plt.axis([0, 80, 0, 60])
nodes = np.indices((80, 60)).reshape(2, -1)
nx = list(nodes[0])
nxOffset = [x + 0.5 for x in nx]
ny = list(nodes[1])
nyOffset = [x + 0.5 for x in ny]
plt.scatter(nxOffset, nyOffset, 3, color="k", marker=".")  # plots SOM node positions as blue dots on grid

plt.axis([0, weights.shape[0], 0, weights.shape[1]])

plt.show()
