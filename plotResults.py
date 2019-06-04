import numpy as np
import csv
import os
import pandas
np.set_printoptions(suppress=True)
from scipy import stats
from matplotlib import pyplot as plt

# part 1 - internal consistency
# extract scores for each participant in an array for each
# for each .csv file
# round ratings
# 10 levels: 0,1,2,3,4,5,6,7,8,9. 0 = 0 similarity, 9=perfect or near perfect

ratingFiles = os.listdir('/home/fred/BFD/python/MusicSOM/Ratings-by-participant-reliable')

ratingFiles.sort() #alphabetize
ratingFiles = ratingFiles[10:90]
scoresByPair = np.zeros([80,21])

def getResultsPerPair(scoresByPair):
    # tested, works
    for i in range(21): #just non-repeated ones
        with open(('/home/fred/BFD/python/MusicSOM/Ratings-by-participant-reliable/' + ratingFiles[i])) as csvfile:
            reader = csv.reader(csvfile, delimiter=",") #so far 21 participants
            j = 0
            for row in reader:
                if row[0] in (None, ""):
                    pass
                elif row[0] == "file_keys":
                    pass
                elif j < 21:
                    scoresByPair[i,j] = row[0]
                    j+=1
    return scoresByPair

def getRepeatedPairs(subject):
    first10 = np.round(subject[0:10] *10.0)+1
    last10 = np.round(subject[80:90] *10.0)+1
    return np.vstack([first10,last10])

scoresBySubject = getResultsPerPair(scoresByPair).T
plt.figure()
meanScores = np.mean(scoresBySubject, axis=0)
print(meanScores.shape)
plt.bar(np.arange(80),meanScores)
plt.show()
