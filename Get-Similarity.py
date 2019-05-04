import numpy as np
from matplotlib import pyplot as plt
import math

evalGrooves = np.load('Eval-matricies.npy')
evalNames = np.load('Eval-names.npy')

for i in range(evalNames.shape[0]):
    if 'Blues CH1 a' in evalNames[i]:
        grooveA = evalGrooves[i]
    if 'Blues CH1 b' in evalNames[i]:
        grooveB = evalGrooves[i]

def euclideanDistance(grooveA, grooveB):
    """Returns norm-2 of a 1-D numpy array for CPU computation.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    x = grooveA.flatten()-grooveB.flatten()
    return math.sqrt(np.dot(x, x.T))

print(grooveA-grooveB)

print(1/(1+euclideanDistance(grooveA,grooveB)))