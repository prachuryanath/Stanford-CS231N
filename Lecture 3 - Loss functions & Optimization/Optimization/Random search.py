"""
Assume X_train is the data where each column is an example  (eg 3073 * 50000)

Assume Y_train are the labels (eg 1D array of 50000)

Assume the function L evaluates the loss function
"""

import numpy as np
import random

bestloss = float("inf")     # Python assigns the highest float value

for num in range(1000):
    W = np.random.randn(10,3073) * 0.0001       # Generate random parameters
    loss = L(X_train, Y_train, W)
    if loss < bestloss:
        bestloss = loss
        bestW = W
    print('In attempt {num} the loss was {loss}, best {bestloss}'.format(num, loss, bestloss))

# Assume X_test is [3073 x 10000], Y_test [10000 x 1]
scores = Wbest.dot(Xte_cols) # 10 x 10000, the class scores for all test examples
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis = 0)
# and calculate accuracy (fraction of predictions that are correct)
np.mean(Yte_predict == Yte)
# returns 0.1555