import numpy as np
import csv
import os
import pandas
np.set_printoptions(suppress=True)
from scipy import stats

# part 1 - internal consistency
# extract scores for each participant in an array for each
# for each .csv file
# round ratings
# 10 levels: 0,1,2,3,4,5,6,7,8,9. 0 = 0 similarity, 9=perfect or near perfect

ratingFiles = os.listdir('/home/fred/BFD/python/MusicSOM/ratings-new')

ratingFiles.sort() #alphabetize

scoresByPair = np.zeros([90,23])
def getResultsPerPair(scoresByPair):
    # tested, works
    for i in range(90):
        with open(('/home/fred/BFD/python/MusicSOM/ratings-new/' + ratingFiles[i])) as csvfile:
            reader = csv.reader(csvfile, delimiter=",") #so far 21 participants
            j = 0
            for row in reader:
                if row[0] in (None, ""):
                    pass
                elif row[0] == "file_keys":
                    pass
                elif j < 23:
                    scoresByPair[i,j] = row[1]
                    j+=1
    return scoresByPair

def getRepeatedPairs(subject):
    first10 = subject[0:10]
    last10 = subject[80:90]
    return np.vstack([first10,last10])

scoresBySubject = getResultsPerPair(scoresByPair).T

def cronbachAlpha(stackedRepeats):
    #alpha = (number of items * average covariance between item-pairs)/average variance + ((no. items -1) * av covar)
    # itemvars = stackedRepeats.var(axis=0,ddof=1)
    # tscores = stackedRepeats.sum(axis=1)
    # nitems = 21
    # variance_sum = float(itemvars.sum())
    # total_var = float(stackedRepeats.sum(axis=1).var(ddof=1))
    # alpha = (21/float(21-1)*float(1-variance_sum/total_var))
    # print(alpha)
    # #alpha = float(nitems)/float(nitems-1.0)*(1.0-itemvars.sum()/float(tscores.var(ddof=1)))

    items = pandas.DataFrame(stackedRepeats)
    items_count = items.shape[1]
    variance_sum = float(items.var(axis=0, ddof=1).sum())
    total_var = float(items.sum(axis=1).var(ddof=1))

    return (items_count / float(items_count - 1) *
            (1 - variance_sum / total_var))

def spearmansRankReliability(subject):
    first10 = subject[0:10]
    last10 = subject[80:90]
    rank = stats.spearmanr(first10,last10)
    print(rank)
    return rank

def fleiss_kappa(s):
    """
    See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
    :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
    :type M: numpy matrix
    """
    first10 = s[0:10]
    last10 = s[80:90]
    # M = np.vstack([first10,first10])
    # N,k = M.shape  # N is # of items, k is # of categories
    #
    # n_annotators = 2  # # of annotators
    #
    # p = np.sum(M, axis=0) / (N * n_annotators)
    # P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    # Pbar = np.sum(P) / N
    # PbarE = np.sum(p * p)
    #
    # kappa = (Pbar - PbarE) / (1 - PbarE)
    import sklearn.metrics
    kappa = sklearn.metrics.cohen_kappa_score(first10,last10)

    return kappa

def fleiss_kappa2(s):
    pass

# for i in range(21):
#     s = scoresBySubject[i]
#     s = np.round(s * 5)
#     first10 = s[0:10]
#     last10 = s[80:90]
#     import sklearn.metrics
#
#     kappa = sklearn.metrics.cohen_kappa_score(first10, last10)
#     print(kappa)

for i in range(23):
    nameStr = "Participant-" + str(i) +"-New.csv"
    s = scoresBySubject[i]
    print(s)
    first10 = s[0:10]
    last10 = s[80:90]
    import sklearn.metrics
    #kappa = sklearn.metrics.cohen_kappa_score(first10, last10)
    np.savetxt(nameStr, s[10:90], delimiter=",", fmt='%f')
    # if kappa > 0.2:
    #     np.savetxt(nameStr, s[10:90], delimiter=",",fmt='%i')
    #     #print(kappa)

