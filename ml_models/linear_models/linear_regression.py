"""
线性回归模型实现
"""

import numpy as np
import matplotlib.pyplot as plt
import utils.utils as utils

class LinearRegression(object):
    """线性回归模型实现

    该类实现了线性回归算法，支持配置是否拟合截距项、选择优化器、
    特征标准化处理等功能，通过迭代优化方法求解模型参数。
    """
    def __init__(self, fit_intercept = True, optimizer = 'sgd', if_standard = True,
                 epochs = 10, lr = 1e-2, batch_size = 1, l1_ratio = None, l2_ratio = None):
        """初始化线性回归模型参数

        :param fit_intercept: bool类型，可选，默认值为True
            是否在模型中拟合截距项（偏置项）。若为True，模型会额外学习一个常数项；
            若为False，模型仅学习特征的系数，不包含截距项。

        :param optimizer: str类型，可选，默认值为'sgd'
            模型训练使用的优化器名称。目前默认支持'sgd'（随机梯度下降），
            可扩展支持其他优化器（如'gd'梯度下降、'momentum'动量法等）。

        :param if_standard: bool类型，可选，默认值为True
            是否对输入特征进行标准化处理。若为True，训练过程中会计算每个特征的均值和标准差，
            并将特征转换为均值为0、标准差为1的标准化特征；若为False，则不进行标准化。

        :param epochs: int类型，可选，默认值为10
            训练迭代的轮数（完整遍历训练集的次数）。轮数越多，模型在训练集上的迭代次数越多，
            需注意避免过拟合，通常需要根据验证集性能调整。

        :param lr: float类型，可选，默认值为1e-2（即0.01）
            学习率（步长），控制优化过程中参数更新的幅度。学习率过大会导致收敛不稳定，
            过小则会延长训练时间，需根据实际场景调整（通常为0.001~0.1之间）。

        :param batch_size: int类型，可选，默认值为1
            每次参数更新使用的样本批量大小。当为1时，为随机梯度下降（SGD）；
            当等于训练集大小时，为批量梯度下降（BGD）；当为1~训练集大小之间时，
            为小批量梯度下降（MBGD）。需为正整数，且不超过训练样本总数。

        :ivar w: 模型参数（系数），初始化为None，训练后存储特征对应的权重
        :ivar feature_mean: 特征均值，当if_standard为True时有效，存储各特征的均值
        :ivar feature_std: 特征标准差，当if_standard为True时有效，存储各特征的标准差
        """
        self.w = None
        self.fit_intercept = fit_intercept
        self.optimizer = optimizer
        self.if_standard = if_standard
        if if_standard:
            self.feature_mean = None
            self.feature_std = None
        self.epochs = epochs
        self.eta = lr
        self.batch_size = batch_size
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        # 注册sign函数
        self.sign_func = np.vectorize(utils.sign)

    def init_params(self, n_features):
        """初始化模型参数（权重矩阵）

        该方法根据输入特征的数量，使用随机数初始化模型的权重参数w，
        生成形状为(n_features, 1)的初始权重矩阵，为后续模型训练做准备。

        :param n_features: int类型，正整数
            输入特征的数量（特征维度）。决定了权重矩阵的行数，需与训练数据的特征维度一致。

        :return: None
            无返回值，初始化结果直接存储在实例变量self.w中。

        :ivar self.w: numpy数组，形状为(n_features, 1)
            初始化后的值为[0, 1)区间内的随机浮点数，存储模型的特征权重参数。
        """
        self.w = np.random.random(size = (n_features, 1))


    def _fit_closed_form_solution(self, x, y):
        """
        直接求闭式解
        :param x:
        :param y:
        :return:
        """
        if self.l1_ratio is None and self.l2_ratio is None:
            self.w = np.linalg.pinv(x).dot(y)
        elif self.l1_ratio is None and self.l2_ratio is not None:
            self.w = np.linalg.inv(x.T.dot(x) + self.l2_ratio * np.eye(x.shape[1])).dot(x.T).dot(y)
        else:
            self._fit_sgd(x, y)

    def _fit_sgd(self, x, y):
        """
        随机梯度下降求解
        :param x:
        :param y:
        :param epochs:
        :param eta:
        :param batch_size:
        :return:
        """
        x_y = np.c_[x, y]
        # 按batch_size更新w,b
        for _ in range(self.epochs):
            np.random.shuffle(x_y)
            for index in range(x_y.shape[0] // self.batch_size):
                batch_x_y = x_y[self.batch_size * index:self.batch_size * (index + 1)]
                batch_x = batch_x_y[:, :-1]
                batch_y = batch_x_y[:, -1:]

                dw = -2 * batch_x.T.dot(batch_y - batch_x.dot(self.w)) / self.batch_size

                # add the function of using L1/L2
                dw_reg = np.zeros(shape=(x.shape[1] - 1, 1))
                if self.l1_ratio is not None:
                    dw_reg += self.l1_ratio * self.sign_func(self.w[:-1]) / self.batch_size
                if self.l2_ratio is not None:
                    dw_reg += 2 * self.l2_ratio * self.w[:-1] / self.batch_size
                dw_reg = np.concatenate([dw_reg, np.asarray([[0]])], axis=0)
                dw += dw_reg

                self.w = self.w - self.eta * dw

    def fit(self, x, y):
        # 是否归一化feature
        if self.if_standard:
            self.feature_mean = np.mean(x, axis=0)
            self.feature_std = np.std(x, axis=0) + 1e-8
            x = (x - self.feature_mean) / self.feature_std
        # 是否训练bias
        if self.fit_intercept:
            x = np.c_[x, np.ones_like(y)]
        # 初始化参数
        self.init_params(x.shape[1])
        # 训练模型
        if self.optimizer == 'closed_form':
            self._fit_closed_form_solution(x, y)
        elif self.optimizer == 'sgd':
            self._fit_sgd(x, y)

    def get_params(self):
        """
        输出原始的系数
        :return: w,b
        """
        if self.fit_intercept:
            w = self.w[:-1]
            b = self.w[-1]
        else:
            w = self.w
            b = 0
        if self.if_standard:
            w = w / self.feature_std.reshape(-1, 1)
            b = b - w.T.dot(self.feature_mean.reshape(-1, 1))
        return w.reshape(-1), b

    def predict(self, x):
        """
        :param x:ndarray格式数据: m x n
        :return: m x 1
        """
        if self.if_standard:
            x = (x - self.feature_mean) / self.feature_std
        if self.fit_intercept:
            x = np.c_[x, np.ones(shape=x.shape[0])]
        return x.dot(self.w)

    def plot_fit_boundary(self, x, y):
        """
        绘制拟合结果
        :param x:
        :param y:
        :return:
        """
        plt.scatter(x[:, 0], y)
        plt.plot(x[:, 0], self.predict(x), 'r')

if __name__ == '__main__':
    # create some samples
    X = np.linspace(0, 100, 100)  # [0,100], step equal, 100 numbers
    X = np.c_[X, np.ones(100)]  # create a matrix, the second column is all 1

    w = np.asarray([2, 3])  # create `w` matrix,

    Y = X.dot(w) # X 和 w 进行矩阵乘法

    X = X.astype('float')
    Y = Y.astype('float')
    X[:, 0] += np.random.normal(size=X[:, 0].shape) * 3  # 添加噪声
    Y = Y.reshape(100, 1)

    # 拟合数据并可视化
    plt.title('Linear Regression initialize')
    linear_model = LinearRegression()
    linear_model.fit(X[:, :-1], Y)
    linear_model.plot_fit_boundary(X[:, :-1], Y)
    plt.show()

    # 加入异常点
    X = np.concatenate([X, np.asanyarray([[100, 1], [101, 1], [102, 1], [103, 1], [104, 1]])])
    Y = np.concatenate([Y, np.asanyarray([[3000], [3300], [3600], [3800], [3900]])])
    plt.title('Linear Regression initialize with wired points data')
    linear_model = LinearRegression()
    linear_model.fit(X[:, :-1], Y)
    linear_model.plot_fit_boundary(X[:, :-1], Y)
    plt.show()

    # lasso
    lasso = LinearRegression(l1_ratio = 100)
    plt.title('Lasso initialize with lasso fix')
    lasso.fit(X[:, :-1], Y)
    lasso.plot_fit_boundary(X[:, :-1], Y)
    plt.show()

    # ridge
    ridge = LinearRegression(l2_ratio=10)
    plt.title('Linear Regression initialize with ridge fix')
    ridge.fit(X[:, :-1], Y)
    ridge.plot_fit_boundary(X[:, :-1], Y)
    plt.show()

    # elastic net(both lasso and ridge)
    elastic = LinearRegression(l1_ratio=100, l2_ratio=10)
    plt.title('Linear Regression initialize with elastic fix')
    elastic.fit(X[:, :-1], Y)
    elastic.plot_fit_boundary(X[:, :-1], Y)
    plt.show()