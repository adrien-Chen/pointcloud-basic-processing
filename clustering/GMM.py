# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

from KMeans import K_Means

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    # 更新W : NX3 表示点属于每个类的概率
    def update_W(self, data, pi, Mu, Var):
        pdfs = np.zeros((len(data), self.n_clusters))  # NX3
        for cluster in range(self.n_clusters):
            if np.array(Var[cluster]).any() == 0:
                # reset Mu to randomly chosen values, and Var to a large value
                Mu[cluster] = data[np.random.randint(0, len(data), 1)]
                Var[cluster] = [1, 1]
            pdfs[:, cluster] = pi[cluster] * multivariate_normal.pdf(data, Mu[cluster], np.diag(Var[cluster]))
        W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        return W


    # 更新pi
    def update_pi(self, W):
        N_K = W.sum(axis=0)
        N = W.sum()
        pi = N_K / N
        return pi
        
    # 更新Mu
    def update_Mu(self, data, W):
        Mu = np.zeros((self.n_clusters, data.shape[1]))
        for i in range(self.n_clusters):
            Mu[i] = np.average(data, axis=0, weights=W[:, i])
        return Mu

    # 更新Var
    def update_Var(self, data, W, Mu):
        Var = np.zeros((self.n_clusters, data.shape[1]))
        for i in range(self.n_clusters):
            Var[i] = np.average((data - Mu[i])**2, axis=0, weights=W[:, i])
        return Var


    def fit(self, data):
        # 1. 初始化参数
        self.W = np.ones((len(data), self.n_clusters)) / self.n_clusters
        self.pi = [1 / self.n_clusters] * self.n_clusters
        self.Var = [[1, 1], [1, 1], [1, 1]] # 假设协方差矩阵都是对角阵，且各方向方差一致，可视化为圆形，只存储对角元素

        # rand_idx = np.arange(data.shape[0])
        # np.random.shuffle(rand_idx)
        # self.Mu = data[rand_idx[0: self.n_clusters]]

        # 用k-means初始化均值
        my_kmeans = K_Means(n_clusters=3)
        my_kmeans.fit(data)
        self.Mu = my_kmeans.get_centers()

        # 迭代优化：
        iters = 0
        while iters < self.max_iter:
            # E-step: 计算并更新后验概率 W
            self.W = self.update_W(data, self.pi, self.Mu, self.Var)

            # M-step: MLE 更新模型参数
            self.pi = self.update_pi(self.W)
            self.Mu = self.update_Mu(data, self.W)
            self.Var = self.update_Var(data, self.W, self.Mu)

            iters += 1
            print("iters = ",  iters)

    
    def predict(self, data):
        W = self.update_W(data, self.pi, self.Mu, self.Var)
        result = [np.argmax(W[i]) for i in range(len(data))]
        return result

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

