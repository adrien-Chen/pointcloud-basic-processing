# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    '''
    :param correlation: 是否使用相关系数矩阵计算特征值和特征向量
    原始数据的相关系数阵 == 标准化数据的协方差矩阵；
    当不同维度数据的量纲相差很大时，协方差矩阵不适用，不能很好反应数据特点，用相关系数矩阵比较合适；
    '''
    # 使用相关系数矩阵计算
    if(correlation == True):
        cov_data = np.corrcoef(data.T)
    else:
        # 1. normalized  by the center:
        m, n = np.shape(data)
        x_mean = np.mean(data, axis=0) # 1X3
        avgs = np.tile(x_mean, (m, 1))
        data = data - avgs
        # 2. compute covMatrix：
        cov_data = np.cov(data.T)
    # 3. compute eigenvalues & eigenvectors:
    eigenvalues, eigenvectors = np.linalg.eig(cov_data)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    classes = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone",
               "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
               "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
               "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]

    dir = "./modelnet40_normal_resampled/"

    for name in classes:

        # 加载原始点云
        filename = dir + name + "/" + name + "_0001.txt"
        points = np.genfromtxt(filename, delimiter=",")
        # points = np.genfromtxt("./modelnet40_normal_resampled/airplane/airplane_0001.txt", delimiter=",")
        points = pd.DataFrame(points[:, 0:3])
        points.columns = ['x', 'y', 'z']
        point_cloud_pynt = PyntCloud(points)
        #point_cloud_pynt = PyntCloud.from_file("/Users/renqian/Downloads/program/cloud_data/11.ply")
        point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
        # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

        # 从点云中获取点，只对点进行处理
        points = point_cloud_pynt.points
        print('total points number is:', points.shape[0])

        # 用PCA分析点云主方向
        w, v = PCA(points)
        # test cov and corrcoef:
        w2, v2 = PCA(points, True)

        point_cloud_vector = v[:, 0]  # 点云主方向对应的向量
        print('the main orientation of this pointcloud is: ', point_cloud_vector)
        # TODO: 此处只显示了点云，还没有显示PCA
        # o3d.visualization.draw_geometries([point_cloud_o3d])

        # # PCA 投影可视化：1. encoder 2. decoder
        # for k in range(3):
        #     pca_matrix = v[:, 0:k+1]
        #
        #     points_ = points.dot(pca_matrix) # encoder
        #     points_ = points_.dot(pca_matrix.T) # decoder
        #
        #     points_ = pd.DataFrame(points_)
        #     points_.columns = ['x', 'y', 'z']
        #     point_cloud_pynt_pca = PyntCloud(points_)
        #
        #     point_cloud_o3d_pca = point_cloud_pynt_pca.to_instance("open3d", mesh=False)
        #     o3d.visualization.draw_geometries([point_cloud_o3d_pca])
        #
        # # test: using corrcoef:
        # for k in range(3):
        #     pca_matrix = v2[:, 0:k+1]
        #
        #     points_ = points.dot(pca_matrix) # encoder
        #     points_ = points_.dot(pca_matrix.T) # decoder
        #
        #     points_ = pd.DataFrame(points_)
        #     points_.columns = ['x', 'y', 'z']
        #     point_cloud_pynt_pca = PyntCloud(points_)
        #
        #     point_cloud_o3d_pca = point_cloud_pynt_pca.to_instance("open3d", mesh=False)
        #     o3d.visualization.draw_geometries([point_cloud_o3d_pca])

        # 循环计算每个点的法向量
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
        normals = []

        # 1. select point p and find neighborhood:
        for p in range(0, len(points)):  # for loop all points
            # 1.1 using knn: choose 200 nearest neighborhood
            # 最近邻搜索此处直接调用open3d中的函数
            [k, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[p], 200)

            # # 1.2 using radius: distance less than 0.2:
            # [k, idx, _] = pcd_tree.search_radius_vector_3d(point_cloud_o3d.points[p], 0.2)

        # 2. do PCA for all neighborhood of point p:
            neighborhood = points.iloc[idx[:], :]
            w, v = PCA(neighborhood)
            # normal is the least significant vector:
            normals.append(v[:, 2])

        normals = np.array(normals, dtype=np.float64)
        # TODO: 此处把法向量存放在了normals中
        point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
