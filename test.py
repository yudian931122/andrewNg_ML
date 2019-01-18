import pandas
import numpy as np


def labelMapping(a):
    """
    标签映射
    :param a:
    :return:
    """
    zeros = np.zeros(10)
    zeros[a[0] - 1] = 1

    return zeros


a = np.array([[1., 4., 2., 6., 5., 3.], [4., 2., 6., 8., 3., 1.]])
print(a)
argmin = np.argmin(a, axis=1, )
print(argmin.shape)
