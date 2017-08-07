import pandas as pd
import numpy as np

data = pd.read_csv('C:\\Users\\PeterKokalov\\lpthw\\Machine_Learning\\Week1-2\\machine-learning-ex1\\ex1\\ex1data1.txt')
data.columns = ['Profit','Population']
ones = np.zeros((len(data)))
for i in range(len(ones)):
    ones[i] = 1

theta = np.zeros(shape=(2,1))
iterations = 1500
alpha = 0.01

data['Ones'] = ones
X = data
X = pd.DataFrame(X, columns=['Ones', 'Profit'])
y = data['Population']

def cost_func(X, y, theta, x_i):
    SSE = 0
    m = y.size
    for i in range(m):
        x = X.ix[i] 
        y_i = y.ix[i] 
        h_x_i = theta[0] + theta[1] * x 
        loss = (h_x_i - y_i) * x[x_i] 
        SSE += loss
    total_loss = SSE * (1/m)

    return total_loss

def gradientDescent(X, y, alpha, theta):
    
    thetas = [] 
    
    m = y.size
    for x_i in range(len(theta)):
        cost = cost_func(X, y, theta, x_i)
        convergence_alpha = theta[x_i] - ((alpha * (1/m)) * cost)
        thetas.append(convergence_alpha)

    return thetas

def linreg(X, y, alpha, theta, iterations):
    
    descent_markers = []
    
    for i in range(iterations):
        optimal_theta = gradientDescent(X, y, alpha, theta)
        descent_markers.append(optimal_theta)
        theta = optimal_theta

    return theta

def main():
    X, y = data()
    theta = [0,0]
    alpha = 0.01
    iterations = 1500
    opt_theta = linreg(X, y, alpha, theta, iterations)

    return opt_theta
# 
# if __name__ == '__main__':
#     main()
