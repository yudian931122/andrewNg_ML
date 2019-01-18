import pandas
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def sigmoid(z):
    """
    sigmoid函数
    :param z:
    :return:
    """

    return numpy.power(1 + numpy.exp(-z), -1)


def costFunction(theta, X, y):
    """
    损失函数
    :param theta:
    :param X:
    :param y:
    :return:
    """
    m = y.shape[0]

    h = sigmoid(numpy.dot(X, theta))

    J = (numpy.dot((-y).T, numpy.log(h)) - numpy.dot((1 - y).T, numpy.log(1 - h))) / m

    return J


def grad(theta, X, y):
    """
    梯度计算
    :param theta:
    :param X:
    :param y:
    :return:
    """
    m = len(y)

    h = sigmoid(numpy.dot(X, theta))

    grad = numpy.dot((h - y).T, X).T / m

    return grad


def plotData(X, y):
    """
    展示数据
    :param X:
    :param y:
    :return:
    """
    plt.figure()

    # 分别找出正例和反例的索引
    pos = numpy.where(y == 1)
    neg = numpy.where(y == 0)

    plt.scatter(X[pos, 0], X[pos, 1], marker="+", edgecolors='k', label='y=1')
    plt.scatter(X[neg, 0], X[neg, 1], marker="o", edgecolors='k', label='y=0')


def plotDecisionBoundary(theta, X, y):
    """
    绘制决策边界
    :param theta:
    :param X:
    :param y:
    :return:
    """
    plotData(X[:, 1:], y)

    if X.shape[1] <= 3:
        # 特征数量为2，且假设函数最高为1次幂时的边界绘制，由于手动添加了一个特征，所以这里判断条件为3
        plot_x = numpy.array([numpy.min(X[:, 1]) - 2, numpy.max(X[:, 1])])
        plot_y = -(theta[0] + theta[1] * plot_x) / theta[2]

        plt.plot(plot_x, plot_y, 'r', label='boundary')

    else:
        # 生成坐标
        u = numpy.linspace(-1, 1.5, 50)
        v = numpy.linspace(-1, 1.5, 50)

        # 生成坐标网格，这里不是很好解释，打印看一下就懂了
        t1, t2 = numpy.meshgrid(u, v)

        # ravel()是将矩阵拉平，变成一个一维数组
        # 拉平后各作为一个特征，生成多项式特征
        mapping = featureMapping(t1.ravel(), t2.ravel())

        # 计算
        z = numpy.dot(mapping, theta)

        # 绘图
        plt.contour(u, v, z.reshape(t1.shape), [0, ])


def predict(theta, X):
    """
    预测
    :param theta:
    :param X:
    :return:
    """
    m = X.shape[0]

    p = numpy.zeros((m,))

    # 值大于0.5的索引
    pos = numpy.where(sigmoid(numpy.dot(X, theta)) >= 0.5)

    # 将大于0.5的值替换成1
    p[pos] = 1

    return p


def logisticRegression():
    """
    逻辑回归
    :return:
    """
    dataPath = r"D:\ML\AndrewNg\machine-learning-ex2\ex2\ex2data1.txt"
    data = numpy.loadtxt(dataPath, delimiter=",")

    X = data[:, 0:2]
    y = data[:, 2]

    # 数据可视化
    plotData(X, y)
    plt.title('lr')
    plt.xlabel('Exam 1 score')
    plt.ylabel('exam 2 score')
    plt.legend()
    plt.show()

    # 假设函数中考虑截距的情况下，给每个样本增加一个为1的特征
    ones = numpy.ones((X.shape[0], 1))
    X = numpy.c_[ones, X]

    # 初始化theta
    theta = numpy.zeros((X.shape[1],))

    # 初始theta计算出的损失和梯度
    J = costFunction(theta, X, y)
    g = grad(theta, X, y)

    print('Cost at initial theta (zeros): %f' % J)
    print('Expected cost (approx): 0.693')
    print('Gradient at initial theta (zeros):')
    print(g)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

    # 使用最优化方法求解，类似于octave中的fminumc, 要注意的是，计算损失和计算梯度是分开的
    bfgs = minimize(costFunction, theta, args=(X, y), jac=grad, method='BFGS')

    # 最优参数theta
    theta = bfgs.x

    # 绘制决策边界
    plotDecisionBoundary(theta, X, y)
    plt.title('lr')
    plt.xlabel('Exam 1 score')
    plt.ylabel('exam 2 score')
    plt.legend()
    plt.show()

    # 预测
    prob = sigmoid(numpy.dot(numpy.array([1, 45, 85]), theta))
    print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob)
    print('Expected value: 0.775 +/- 0.002\n')

    p = predict(theta, X)

    print('Train Accuracy: ', numpy.mean(numpy.double(p == y)) * 100)
    print('Expected accuracy (approx): 89.0\n')


def featureMapping(X1, X2):
    """
    根据两个特征生成多项式特征， 最多到6次幂
    也可以使用现有机器学习包里面的方法生成多项式特征，比如：sklearn.preprocessing中的PolynomialFeatures
    :param X1:
    :param X2:
    :return:
    """
    degree = 6
    out = numpy.ones((X1.shape[0], 1))

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            out = numpy.column_stack((out, numpy.power(X1, i - j) * numpy.power(X2, j)))

    return out


def costFunctionReg(theta, X, y, lamb):
    """
    使用规则化项时的损失计算
    :param theta:
    :param X:
    :param y:
    :param lamb:
    :return:
    """
    m = X.shape[0]
    # theta_r = theta 这样的直接赋值是不行的，需要使用copy
    theta_r = theta.copy()
    theta_r[0] = 0

    h = sigmoid(numpy.dot(X, theta))

    J = (numpy.dot((-y).T, numpy.log(h)).T - numpy.dot((1 - y).T, numpy.log(1 - h))) / m + \
        numpy.dot(theta_r.T, theta_r) * (lamb / (2 * m))

    return J


def gradReg(theta, X, y, lamb):
    """
    使用规则化项时的梯度计算
    :param theta:
    :param X:
    :param y:
    :param lamb:
    :return:
    """
    m = X.shape[0]
    # theta_r = theta 这样的直接赋值是不行的，需要使用copy
    theta_r = theta.copy()
    theta_r[0] = 0

    h = sigmoid(numpy.dot(X, theta))

    grad = numpy.dot((h - y).T, X).T / m + theta_r * (lamb / m)

    return grad


def logisticRegressionWithRegularized():
    """
    使用规则化的逻辑回归
    :return:
    """
    dataPath = r"D:\ML\AndrewNg\machine-learning-ex2\ex2\ex2data2.txt"
    data = numpy.loadtxt(dataPath, delimiter=',')

    X = data[:, 0:2]
    y = data[:, 2]

    # 数据可视化
    plotData(X, y)
    plt.title("lr_reg")
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.show()

    # 生成多项式特征, 在这个函数中，已经添加了一列全为1的特征，不用再额外添加
    # 也可以使用现有机器学习包里面的方法生成多项式特征，比如：sklearn.preprocessing中的PolynomialFeatures
    X = featureMapping(X[:, 0], X[:, 1])

    # 初始化theta
    theta = numpy.zeros((X.shape[1],))

    # 初始化规则化系数lambda
    lamb = 1

    # 初始theta计算出的损失和梯度
    J = costFunctionReg(theta, X, y, lamb)
    g = gradReg(theta, X, y, lamb)

    print('Cost at initial theta (zeros): %f' % J)
    print('Expected cost (approx): 0.693 \n')
    print('Gradient at initial theta (zeros) - first five values only:')
    print(g[0:5])
    print('Expected gradients (approx) - first five values only:')
    print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

    # 使用BFGS算法，也可以使用不同的算法，使用method参数指定
    bfgs = minimize(costFunctionReg, theta, (X, y, lamb), jac=gradReg, method='BFGS')

    theta = bfgs.x

    # 绘制决策边界
    plotDecisionBoundary(theta, X, y)
    plt.title('lr_reg, lambda = %f' % lamb)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.show()

    # 预测及评估
    p = predict(theta, X)

    print('Train Accuracy: ', numpy.mean(numpy.double(p == y)) * 100)
    print('Expected accuracy (with lambda = 1): 83.1 (approx)')


if __name__ == '__main__':
    print("********************************** Logistic Regression With **********************************\n")
    logisticRegression()
    print("\n********************************** Logistic Regression With Regularized **********************************\n")
    logisticRegressionWithRegularized()
