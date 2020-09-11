# 文件功能： 实现 K-Means 算法

import numpy as np

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def calculate_r_nk(self, point):
        dists = self.centers - point
        dists = np.linalg.norm(dists, axis=1)
        return np.argmin(dists) # 返回所属聚类的索引下标

    def fit(self, data):
        # 拟合模型参数：计算均值、方差等

        # 1. 初始化 中心点 & r_nk:
        # 随机选K个点：
        rand_idx = np.arange(data.shape[0])
        np.random.shuffle(rand_idx)
        self.centers = data[rand_idx[0:self.k_]]

        # 2. 进行迭代：
        '''
        # 终止条件：
            1. tolerance 很小或不变
            2. r_nk 不变
            3. 超过迭代次数
        '''
        iters = 0
        old_r_nk = None
        while iters < self.max_iter_:

            # E-step:
            new_r_nk = [self.calculate_r_nk(point) for point in data]

            # r_nk不变则终止:
            if old_r_nk == new_r_nk:
                return
            old_r_nk = new_r_nk

            # M-step:更新聚类中心
            old_centers = self.centers
            for idx in range(self.k_):
                points_idx = np.where(np.array(new_r_nk) == idx)
                points = data[points_idx]
                self.centers[idx] = points.mean(axis=0)
            # 如果中心误差很小则终止：
            if abs(old_centers - self.centers).all() < self.tolerance_:
                return

            iters += 1


    def predict(self, p_datas): # p_data: array of shape [n_samples, n_features]
        result = []
        result = [self.calculate_r_nk(point) for point in p_datas]
        return result

    def get_centers(self):
        return self.centers

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)
