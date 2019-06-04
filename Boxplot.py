import numpy as np
import csv
import os
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


np.set_printoptions(suppress=True)
from scipy import stats
ratingFiles = os.listdir('/home/fred/BFD/python/MusicSOM/Reliable-ratings-continuous')

ratingFiles.sort()

allScores = np.zeros([21,80])
# tested, works
for i in range(21):
    with open(('/home/fred/BFD/python/MusicSOM/Reliable-ratings-continuous/' + ratingFiles[i])) as csvfile:
        reader = csv.reader(csvfile, delimiter=",") #so far 21 reliable participants
        j = 0
        for row in reader:
            allScores[i,j] = row[0]
            j+=1

fig, ax = plt.subplots()
ax.boxplot(allScores,showfliers=False)
ax.set_xticklabels([])
plt.xlabel('Trials')
plt.ylabel('Similarity Rating')
ax.set_title('Inter-rater consistency for reliable participants')
plt.show()