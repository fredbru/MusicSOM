## Take .csv file in format Groove, Palette, Xval, Yval and output JSON dict for use within Max/MSP
#  Python 3 only - uses ordered dict
# Current issues: need to replace underscores with spaces due to Max/MSP issues
# Need to manually add palettes to groove csv

import csv
import json
import collections
import sys
import os
from xml.dom.minidom import parse



def addRelativePathToSystemPath(relPath):
    if __name__ == '__main__' and __package__ is None:
        from os import sys, path
        sys.path.append(path.join(path.dirname(path.abspath(__file__)), relPath))

addRelativePathToSystemPath("../shared")

sys.path.append("/home/fred/BFD/python/grooves/All-4-4/")

from fx.bfd.groove import *
from fx.common.filesystem import *

def getGrooveFileNames(path):
    grooveFileNames = os.listdir(path)
    return grooveFileNames

# path to .csv file
path = "/home/fred/BFD/python/grooves/All-4-4/"
grooveFileNames = getGrooveFileNames(path)



genreTags = []
with open("SOM_Winners.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:

        paletteFileName = path + row[1].replace("_", " ") + ".bfd3pal"
        grooveDom = parse(paletteFileName)
        bundleNode = getGrooveBundleNode(grooveDom)
        info = grooveDom.getElementsByTagName("BFD2GrooveBundleInfo")
        genre = info[0].getAttribute("genre")
        row = [str(genre),row[0],row[1],row[2],row[3]]
        print(row)
        genreTags.append(row)

with open("SOM_Genres_milc_final.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(genreTags)
