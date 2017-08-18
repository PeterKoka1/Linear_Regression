"""
Vectorized Implementation of Linear Regression on dataset provided in Professor Ng's Machine Learning Course.

PERFORMANCE:

Starting gradient descent at b = 4, m = 2
Performing Descent...
2000 iterations - optimal theta's:
Theta 0: -3.919
Theta 1: 1.062
"""

import pandas as pd
import numpy as np

data = pd.read_csv('C:\\Users\\PeterKokalov\\lpthw\\Machine_Learning\\Week1-2\\machine-learning-ex1\\machine-learning-ex1\\ex1\\ex1data1.txt')
data.columns = ['Population', 'Profit']

iterations = 2000
alpha = 0.01

X = data
X = pd.DataFrame(X['Population'])
X = X.as_matrix()
y = pd.DataFrame(data['Profit'])
y = y.as_matrix()

def costFunctionJ(x, y, theta, x_i):
    h_x = theta[0] + theta[1] * x
    loss = h_x - y[x_i]
    return loss

def stochastic_gd(X, y, alpha, iterations):
    m = len(y)
    theta = [4,2]
    j_hist = np.ones(iterations)
    for iter in range(iterations):
        SSE = 0
        for i in range(m):
            loss = costFunctionJ(X[i], y, theta, i)
            temp0 = theta[0] - (alpha * loss)
            temp1 = theta[1] - (alpha * loss * X[i])
            theta[0] = temp0
            theta[1] = temp1
            new_loss = costFunctionJ(X[i], y, theta, i)
            SSE += new_loss**2
        j_hist[iter] = SSE

    return theta, j_hist

print("Starting gradient descent at b = 4, m = 2")
print("Performing Descent...")
theta, j_hist = stochastic_gd(X, y, alpha, iterations)
theta = [j for subj in theta for j in subj]
print("{} iterations - optimal theta's:".format(iterations))
for i in range(len(theta)):
    print("Theta {}: {}".format(i, round(theta[i],3)))
