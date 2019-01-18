import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def linearRegCostFunction(theta, X, y, lamb):
    theta = theta.reshape(len(theta), 1)
    m = X.shape[0]
    h = X @ theta - y
    theta_r = theta[1:, :]

    J = h.T @ h / (2 * m) + theta_r.T @ theta_r * (lamb / (2 * m))

    return J


def linearRegGradient(theta, X, y, lamb):
    theta = theta.reshape(len(theta), 1)
    m = X.shape[0]
    grad = X.T @ (X @ theta - y) / m
    grad[1:, :] += theta[1:, :] * (lamb / m)

    return grad


def trainLinearReg(X, y, lamb):
    init_theta = np.zeros(X.shape[1])

    tnc = minimize(linearRegCostFunction, init_theta, (X, y, lamb), jac=linearRegGradient, method='TNC')

    return tnc.x


def learningCurves(X, y, X_val, y_val):
    pass


if __name__ == '__main__':
    data_path = r'D:\ML\AndrewNg\machine-learning-ex5\ex5\ex5data1.mat'
    data = sio.loadmat(data_path)

    X = data['X']
    y = data['y']

    X_test = data['Xtest']
    y_test = data['ytest']

    X_val = data['Xval']
    y_val = data['yval']

    # plt.figure()
    # plt.plot(X, y, 'xr')
    # plt.xlabel('Change in water level(x)')
    # plt.ylabel('Water flowing out of the dam(y)')
    # plt.show()
    #
    # lamb = 1
    # theta = np.ones((2, 1))
    #
    # J = linearRegCostFunction(theta, np.c_[np.ones(X.shape[0]), X], y, lamb)
    # grad = linearRegGradient(theta, np.c_[np.ones(X.shape[0]), X], y, lamb)
    # print(J)
    # print(grad)

    lamb = 0
    theta = trainLinearReg(np.c_[np.ones(X.shape[0]), X], y, lamb)

    plt.figure()
    plt.plot(X, y, 'xr')
    plt.plot(X, np.c_[np.ones(X.shape[0]), X] @ theta, '--')
    plt.xlabel('Change in water level(x)')
    plt.ylabel('Water flowing out of the dam(y)')
    plt.show()

