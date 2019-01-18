import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def displayData(data):
    """
    数据可视化
    :param data:
    :return:
    """
    rows = np.int(np.sqrt(data.shape[0]))
    cols = np.int(data.shape[0] / rows)

    width = np.int(np.sqrt(data.shape[1]))

    fig, ax_array = plt.subplots(rows, cols, sharex='all', sharey='all', figsize=(8, 8))

    for r in range(rows):
        for c in range(cols):
            ax_array[r, c].matshow(data[r * rows + c].reshape(width, width), cmap='gray_r')

    plt.xticks()
    plt.yticks()
    plt.show()


def labelMapping(a):
    """
    标签映射
    :param a:
    :return:
    """
    zeros = np.zeros(10)
    zeros[a[0] - 1] = 1

    return zeros


def sigmoid(z):
    """
    sigmoid函数
    :param z:
    :return:
    """

    return 1.0 / (1 + np.exp(-z))


def sigmoidGradient(z):
    """
    sigmoid求导
    :param z:
    :return:
    """

    return sigmoid(z) * (1 - sigmoid(z))


def randInitializeWeights(L_out, L_in):
    """
    生成随机权重矩阵
    :param L_out:
    :param L_in:
    :return:
    """
    epsilon_init = 0.12
    # 服从均匀分布的随机权重矩阵
    return np.random.uniform(-epsilon_init, epsilon_init, (L_out, L_in + 1))


def debugInitializeWeights(fan_out, fan_in):
    """
    生成调试用的参数矩阵
    :param fan_out:
    :param fan_in:
    :return:
    """
    W = np.zeros((fan_in + 1, fan_out))
    return np.sin(range(1, W.size + 1)).reshape(W.shape).T / 10


def gradientChecking(lamb=0):
    """
    梯度检验
    :param lamb:
    :return:
    """
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    X = debugInitializeWeights(m, input_layer_size - 1)

    y = 1 + np.mod(range(1, m + 1), num_labels).reshape(m, 1)

    y_m = np.zeros((m, num_labels))
    for i in range(m):
        zeros = np.zeros((num_labels,))
        zeros[y[i] - 1] = 1
        y_m[i] = zeros

    params = np.r_[Theta1.flatten(), Theta2.flatten()]

    grad = gradientVectorized(params, input_layer_size, hidden_layer_size, num_labels, X, y_m, lamb)

    numgrad = np.zeros(params.shape)
    perturb = np.zeros(params.shape)

    e = pow(10, -4)

    for p in range(params.size):
        perturb[p] = e
        loss1 = costFunction(params - perturb, input_layer_size, hidden_layer_size, num_labels, X, y_m, lamb)
        loss2 = costFunction(params + perturb, input_layer_size, hidden_layer_size, num_labels, X, y_m, lamb)

        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    print(np.c_[numgrad, grad])

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print(diff)


def feedForward(Theta1, Theta2, X):
    """
    前向传播
    :param Theta1:
    :param Theta2:
    :param X:
    :return:
    """
    m = X.shape[0]

    a1 = np.c_[np.ones(m), X]

    # 隐含层, @表示矩阵点乘
    z2 = a1 @ Theta1.T
    a2 = np.c_[np.ones(m), sigmoid(z2)]

    # 输出层
    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)

    return a1, z2, a2, z3, a3


def costFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    """
    损失函数计算，这里只针对简单的三层神经网络模型
    :param nn_params: 展开的所有模型参数
    :param input_layer_size: 输入层神经元数量
    :param hidden_layer_size: 隐含层神经元数量
    :param num_labels: 标签数量，也就是输出层神经元数量
    :param X: 样本
    :param y: 标签
    :param lamb: 规则化系数 lambda
    :return: J，损失
    """
    # 样本数量
    m = X.shape[0]

    # 先将展开的所有模型参数，根据神经网络的结构，分解成对应的参数矩阵
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

    _, _, _, _, h = feedForward(Theta1, Theta2, X)

    # 计算损失，在矩阵计算中，*表示矩阵的对应元素相乘
    J = np.sum((-y) * np.log(h) - (1 - y) * np.log(1 - h)) / m

    # 损失加上规则化项
    J = J + (lamb / (2 * m)) * (np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:])))

    return J


def gradientVectorized(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    """
    矢量化的梯度计算
    :param nn_params:
    :param input_layer_size:
    :param hidden_layer_size:
    :param num_labels:
    :param X:
    :param y:
    :param lamb:
    :return:
    """
    # 样本数量
    m = X.shape[0]

    # 先将展开的所有模型参数，根据神经网络的结构，分解成对应的参数矩阵
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

    # 前向传播
    a1, z2, a2, z3, a3 = feedForward(Theta1, Theta2, X)

    # 反向传播
    delta3 = a3 - y

    delta2 = delta3 @ Theta2[:, 1:] * sigmoidGradient(z2)

    # 计算梯度
    D2 = delta3.T @ a2 / m
    D1 = delta2.T @ a1 / m

    # 加上规则化项
    D1[:, 1:] += Theta1[:, 1:] * (lamb / m)
    D2[:, 1:] += Theta2[:, 1:] * (lamb / m)

    return np.r_[D1.ravel(), D2.ravel()]


def gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    """
    梯度计算
    :param nn_params: 展开的所有模型参数
    :param input_layer_size: 输入层神经元数量
    :param hidden_layer_size: 隐含层神经元数量
    :param num_labels: 标签数量，也就是输出层神经元数量
    :param X: 样本
    :param y: 标签
    :param lamb: 规则化系数 lambda
    :return: J，损失
    """
    # 样本数量
    m = X.shape[0]

    # 先将展开的所有模型参数，根据神经网络的结构，分解成对应的参数矩阵
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

    # 初始化参数的梯度矩阵
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    for i in range(m):
        # 前向传播
        a1 = X[i]
        a1_1 = np.r_[1, a1]

        z2 = Theta1 @ a1_1
        a2 = sigmoid(z2)
        a2_1 = np.r_[1, a2]

        z3 = Theta2 @ a2_1
        a3 = sigmoid(z3)

        # 反向传播
        delta3 = a3 - y[i]
        delta2 = (Theta2.T @ delta3)[1:] * sigmoidGradient(
            z2)  # 这里直接忽略掉了隐含层中截距神经元的delta，因为在隐含层将delta2反向传播给输入层的过程中，隐含层的截距神经元的误差不会传播过去

        # 计算并更新梯度
        Theta2_grad += delta3.reshape(num_labels, 1) @ a2_1.reshape(hidden_layer_size + 1, 1).T
        Theta1_grad += delta2.reshape(hidden_layer_size, 1) @ a1_1.reshape(input_layer_size + 1, 1).T

    # 计算最终的梯度结果
    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m

    # 加上正则化项
    Theta1_grad[:, 1:] += Theta1[:, 1:] * (lamb / m)
    Theta2_grad[:, 1:] += Theta2[:, 1:] * (lamb / m)

    return np.r_[Theta1_grad.ravel(), Theta2_grad.ravel()]


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    # 前向传播
    _, _, _, _, h = feedForward(Theta1, Theta2, X)

    pre_y = np.argmax(h, axis=1) + 1

    return pre_y.reshape(m, 1)


if __name__ == '__main__':
    data_path = r"D:\ML\AndrewNg\machine-learning-ex4\ex4\ex4data1.mat"
    weight_path = r"D:\ML\AndrewNg\machine-learning-ex4\ex4\ex4weights.mat"
    data = sio.loadmat(data_path)
    weights = sio.loadmat(weight_path)

    X = data['X']
    y = data['y']
    # 将y标签映射成10维的向量
    y_m = np.apply_along_axis(labelMapping, 1, y)

    # 随机选择100组数据进行可视化
    choice = np.random.choice(range(5000), 100)
    images = X[choice]

    displayData(images)

    Theta1 = weights['Theta1']
    displayData(Theta1[:, 1:])
    Theta2 = weights['Theta2']
    params = np.r_[Theta1.ravel(), Theta2.ravel()]

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    # lamb = 0
    # J = costFunction(params, input_layer_size, hidden_layer_size, num_labels, X, y_m, lamb)
    # print(J)

    # lamb = 1
    # J = costFunction(params, input_layer_size, hidden_layer_size, num_labels, X, y_m, lamb)
    # print(J)

    print('gradientChecking...')
    print('gradientChecking in lambda=0')
    # gradientChecking()

    lamb = 3
    print('gradientChecking in lambda=3')
    # gradientChecking(3)

    print('cost J in lambda=3')
    debug_J = costFunction(params, input_layer_size, hidden_layer_size, num_labels, X, y_m, lamb)
    print(debug_J)

    print('training neural network...')
    # 随机初始化参数
    initialTheta1 = randInitializeWeights(hidden_layer_size, input_layer_size)
    initialTheta2 = randInitializeWeights(num_labels, hidden_layer_size)
    initialParams = np.r_[initialTheta1.flatten(), initialTheta2.flatten()]

    lamb = 1
    # 优化参数
    tnc = minimize(costFunction, initialParams, args=(input_layer_size, hidden_layer_size, num_labels, X, y_m, lamb),
                   jac=gradientVectorized, method='TNC', options={'maxiter': 500})

    print(tnc)

    nn_params = tnc.x

    Theta1 = nn_params[:initialTheta1.size].reshape(initialTheta1.shape)
    Theta2 = nn_params[initialTheta1.size:].reshape(initialTheta2.shape)

    displayData(Theta1[:, 1:])

    pre_y = predict(Theta1, Theta2, X)

    print('Train Accuracy: ', np.mean(np.double(pre_y == y)) * 100)
