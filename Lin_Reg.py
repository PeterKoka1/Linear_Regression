import pandas as pd
import numpy as np

def data():
    data = pd.read_csv('ex1data1.txt')
    data.columns = ['Profit','Population']
    ones = np.zeros((len(data)))
    for i in range(len(ones)):
        ones[i] = 1

    data['Ones'] = ones
    X = data
    X = pd.DataFrame(X, columns=['Profit', 'Ones'])
    y = data['Population']

    return X, y

def hypothesis(theta, X):
    obs = theta[0] + theta[1] * X
    return obs

def costFunctionJ(X, y, theta):
    SSE = 0
    m = len(y)
    X = X['Profit']
    for i in range(m):
        x = X.ix[i]
        y_i = y.ix[i]
        h_x_i = hypothesis(theta, x)
        Error = (h_x_i - y_i)
        SSE += Error
    SSE = SSE**2
    cost_out = SSE * (1/(2 * m))

    return cost_out

def costFunctionJ_mult(X, y, theta, x_i):
    SSE = 0
    m = len(y)
    for i in range(m):
        x = X.ix[i] 
        y_i = y.ix[i] 
        h_x_i = hypothesis(theta, x) 
        Error = (h_x_i - y_i) * x[x_i]
        SSE += Error
    cost_out = SSE * (1/(2 * m))

    return cost_out

def gradientDescent(X, y, alpha, theta):
    thetas = [] 
    m = len(y)
    constant = (alpha * (1/m))
    for x_i in range(len(theta)):
        cost = costFunctionJ_mult(X, y, theta, x_i)
        convergence_alpha = theta[x_i] - (constant * cost)
        thetas.append(convergence_alpha)

    return thetas

def linreg(X, y, alpha, theta, iterations):
    for i in range(iterations):
        optimal_theta = gradientDescent(X, y, alpha, theta)
        theta = optimal_theta
        
    return theta

def main():
    X, y = data()
    theta = [0,0]
    alpha = 0.01
    iterations = 1000
