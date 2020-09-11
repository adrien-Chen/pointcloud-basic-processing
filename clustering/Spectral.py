# 文件功能： 实现 Spectral Clustering 算法

import numpy as np
from sklearn.cluster import KMeans


class Spectral(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def calDistanceMatrix(self, data):
        n_points = data.shape[0]
        S = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = data[i] - data[j]
                dist = np.sum(dist**2)
                S[i][j] = dist
                S[j][i] = S[i][j]
        return S

    def build_W_L(self, S, k, sigma=1.0):
        n_points = len(S)
        # 计算相似度矩阵W
        W = np.zeros((n_points, n_points))

        for i in range(n_points):
            # 求第i个点到其余点的距离，并排序：
            dist_with_idx = zip(S[i], range(n_points))
            dist_with_idx = sorted(dist_with_idx, key=lambda x: x[0])
            neighbour_idx = [dist_with_idx[m][1] for m in range(1, k+1)]  # k+1因为自身距离为0

            for j in neighbour_idx:
                W[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
                # 权重/相似度 距离越大权重越小：exp^(-dist/(2*sigma*sigma))
                W[j][i] = W[i][j]

        # 计算D
        D = np.zeros((n_points, n_points))
        for i in range(n_points):
            D[i][i] = W.sum(axis=1)[i]

        # 计算L
        L = D - W
        # # 归一化 L:
        L_rw = np.linalg.inv(D).dot(L)

        return W, L, L_rw

    def fit(self, data):
        # 1. 初始化 W, L:
        S = self.calDistanceMatrix(data)
        self.W, self.L, self.L_rw = self.build_W_L(S, k=10)

        # 2. L_rw 的前K个最小特征向量, 得到新到特征点:
        eigenvalues, eigenvectors = np.linalg.eig(self.L_rw)
        sorted_idx = np.argsort(eigenvalues)
        V = eigenvectors[:, sorted_idx[0:self.k_]]

        # # # # # # # # # #
        # 3. 对Y 做K-means:#
        # # # # # # # # # #

        # sp_kmeans = KMeans(self.k_).fit(V)
        # self.result = sp_kmeans.labels_
        sp_kmeans = KMeans(self.k_)
        sp_kmeans.fit(V)
        self.result = sp_kmeans.labels_



    def predict(self, data):
        return self.result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    spectral = Spectral()
    spectral.fit(x)

    cat = spectral.predict(x)
    print(cat)