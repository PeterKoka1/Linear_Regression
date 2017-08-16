import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\PeterKokalov\\lpthw\\Machine_Learning\\Week1-2\\machine-learning-ex1\\machine-learning-ex1\\ex1\\ex1data1.txt')
data.columns = ['Profit', 'Population']
ones = np.zeros((len(data)))

for i in range(len(ones)):
    ones[i] = 1
rand = np.random.randint(1,10)
theta = [rand, rand]
theta = pd.DataFrame(theta)
iterations = 15000
alpha = 0.01

data['Ones'] = ones
X = data
X = pd.DataFrame(X, columns=['Ones', 'Population'])
y = pd.DataFrame(data['Profit'])

###: LOGIC IMPLEMENTATION

def costFunctionJ(X, y, theta, x_i):
    SSE = 0
    m = y.size
    for i in range(m): 
        x = X.ix[i].as_matrix() 
        x = x[x_i]
        y_i = y.ix[i].as_matrix() 
        h_x_i = theta[0] + theta[1] * x 
        loss = (h_x_i - y_i) * x 
        SSE += loss
    total_loss = SSE * (1 / m)

    return total_loss 

def gradientDescent_1(X, y, alpha, theta):
    cost = 0
    for j in range(len(theta)): # for all theta's
        cost += costFunctionJ(X, y, theta, j)

    return cost

def gradientDescent_2(X, y, alpha, theta):
    m = y.size
    cost = gradientDescent_1(X, y, alpha, theta)
    temp0 = theta[0] - alpha / m * cost
    temp1 = theta[1] - alpha / m * cost
    theta[0] = temp0
    theta[1] = temp1

    return theta

def linreg(X, y, alpha, theta, iterations):

    for i in range(iterations):
        desc_theta = gradientDescent_2(X, y, alpha, theta)
        theta = desc_theta
        print("Iteration {}, theta: {}".format(i,theta))
    theta = tuple(map(tuple, theta))
    return theta

###: VECTORIZED IMPLEMENTATION

def J(X, y, theta):
    m = y.size
    X = X.as_matrix()
    y = y.as_matrix()

    error = sum((np.dot(X, theta) - y)**2) / (2*m)

    return error

def gradient_descent(X, y, theta, alpha, iterations):
    m = y.size
    J_history = np.zeros((iterations, 1))
    theta = theta.as_matrix()
    for iter in range(iterations):
        error = (np.dot(X.as_matrix(), theta) - y)
        theta = theta - ((alpha/m) * (np.dot(error.transpose(),X).transpose()))
        J_history[iter] = J(X, y, theta)

    plt.plot(J_history)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()

    return theta

plt.scatter(X['Population'], y, marker='x')
plt.show()

print(gradient_descent(X, y, theta, alpha, iterations))
print(linreg(X, y, alpha, theta, iterations))

