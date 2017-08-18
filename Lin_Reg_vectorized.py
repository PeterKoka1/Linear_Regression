"""
Vectorized Implementation of Linear Regression on dataset provided in Professor Ng's Machine Learning Course.

PERFORMANCE:

Starting gradient descent at b = 4, m = 2
Performing Descent...
15000 iterations - optimal theta's:
Theta 0: -4.212
Theta 1: 1.214
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('...\\ex1data1.txt') # importing from location
data.columns = ['Population', 'Profit']
ones = np.zeros((len(data)))

for i in range(len(ones)):
    ones[i] = 1

theta = [4,2]
iters = 15000
alpha = 0.01

data['Ones'] = ones
X = data
X = pd.DataFrame(X, columns=['Ones', 'Population'])
y = pd.DataFrame(data['Profit'])
theta = pd.DataFrame(theta)

def J(X, y, theta):
    m = y.size
    error_matrix = sum((np.dot(X, theta) - y)**2) / (2*m)

    return error_matrix

def linreg(X, y, theta, alpha, iterations):
    m = y.size
    X = X.as_matrix()
    y = y.as_matrix()
    J_history = np.zeros((iterations, 1))
    theta = theta.as_matrix()
    for iter in range(iterations):
        error = (np.dot(X, theta) - y)
        theta = theta - ((alpha/m) * (np.dot(error.transpose(),X).transpose()))
        J_history[iter] = J(X, y, theta)

    theta = theta.flatten()
    print("Starting gradient descent at b = 4, m = 2")
    print("Performing Descent...")
    print("{} iterations - optimal theta's:".format(iterations))
    for i in range(len(theta)):
        print("Theta {}: {}".format(i, round(theta[i],3)))
    J_history = J_history.flatten()

    return theta, J_history

theta, J_history = linreg(X, y, theta, alpha, iters)

X_new = X.as_matrix().flatten()
X_new = [val for val in X_new if val != 1]
X_new = np.array(X_new)
y_new = [val for val in y['Profit']]
y_new = np.array(y_new)

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(X_new, y_new)
line = theta[1] * X_new + theta[0]

plt.plot(X_new, y_new, 'x', X_new, line, c='darkblue')
plt.ylim(-5,30)
plt.title('Fitted Slope on Dataset')
plt.ylabel('City Profit in $1000\'s')
plt.xlabel('City Population in 1000\'s')
plt.tight_layout()
plt.savefig('ScatterWithFittedLine_PopVsProfit')
