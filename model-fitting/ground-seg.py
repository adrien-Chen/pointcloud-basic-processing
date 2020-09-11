# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from Spectral import Spectral
from KMeans import K_Means
from DBSCAN import Dbscan
from voxel_filter import  voxel_filter
from pyntcloud import PyntCloud
import open3d as o3d
import pandas as pd

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：点到面的距离
# 输入：
#     point： 三维点
#     params：ax + by + cz + d = 0 平面方程参数
# 输出：
#     距离
def point2plane(point, params):
    a, b, c, d = params
    x, y, z = point
    dist = abs(a*x + b*y + c*z + d)
    dist = dist / np.sqrt(a*a + b*b + c*c)
    return dist

'''
# model fitting:
'''
# 功能：模型拟合
# 输入：
#     data： 一帧完整点云
# 输出：
#     平面模型
def LSQ(data):
    # model: ax + by + cz + d = 0
    H = np.cov(data.T)
    eigenvalues, eigenvectors = np.linalg.eig(H)
    sorted_idx = np.argsort(eigenvalues)
    a, b, c = eigenvectors[sorted_idx[0]]
    print("means of xyz: ", data.mean(axis=0))
    xyz_means = data.mean(axis=0)
    d = -(a*xyz_means[0] + b*xyz_means[1] + c*xyz_means[2])
    params = [a, b, c, d]
    print("params = ", params)
    return params

def RANSAC(data, tao=0.2, max_iters=35):
    num_points = data.shape[0]

    # 5. iterate N times:
    iters = 0
    max_count = 2
    best_n, p = np.zeros((1, 3)), np.zeros((1, 3))

    while iters < max_iters:
        # 1. random sample
        idx = random.sample(range(num_points), 3)
        p1, p2, p3 = data[idx]

        # 2. solve model: 计算平面单位法向量 n
        p12 = p2 - p1
        p13 = p3 - p1
        n = np.cross(p12, p13)
        n = n / np.linalg.norm(n)  # 单位化

        # 3. computer distance(error function):
        count = 0
        for point in data:
            d = abs(np.dot((point-p1), n))
        # 4. count points
            if d < tao:
                count += 1

        if count > max_count:
            max_count = count
            best_n = n
            p = p1

        iters += 1

    return best_n, p, max_count

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    segmengted_cloud = []

    # set params
    tao = 0.2
    max_iters = 35
    ground_index = []
    no_ground_index = []

    '''
    # 1. For RANSAC:
    '''
    n, p, max_count = RANSAC(data, tao, max_iters)
    for idx, point in enumerate(data):
        d = abs(np.dot((point-p), n))
        if d < tao:
            ground_index.append(idx)
        else:
            no_ground_index.append(idx)
            segmengted_cloud.append(point)

    '''
    # 2. For LSQ:
    '''
    # params = LSQ(data=data)
    # for idx, point in enumerate(data):
    #     dist = point2plane(point, params)
    #     if dist < tao:
    #         ground_index.append(idx)
    #     else:
    #         no_ground_index.append(idx)
    #         segmengted_cloud.append(point)

    '''
        # 3. For RANSAC + LSQ: 
        #       refine model after RANSAC
    '''
    # ground_data = data[ground_index, :]
    # params = LSQ(ground_data)
    # ground_index = []
    # no_ground_index = []
    # segmengted_cloud = []
    # for idx, point in enumerate(data):
    #     dist = point2plane(point, params)
    #     if dist < tao:
    #        ground_index.append(idx)
    #     else:
    #         no_ground_index.append(idx)
    #         segmengted_cloud.append(point)


    segmengted_cloud = np.asarray(segmengted_cloud, dtype=np.float32)
    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud, ground_index, no_ground_index

# 功能：显示点云与地面分割
# 输入：
#      data：点云数据
#      ground_index：点云数据（滤除地面后的点云索引）
def plot_ground(data, ground_index, no_ground_index):
    ax = plt.figure().add_subplot(111, projection='3d')

    ax.scatter(data[ground_index, 0], data[ground_index, 1], data[ground_index, 2], s=2, c='b')
    ax.scatter(data[no_ground_index, 0], data[no_ground_index, 1], data[no_ground_index, 2], s=2, c='0.8')
    plt.show()

'''
# clustering:
'''
# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    n_clusters = 30

    # spectral clustering:
    # spectral = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
    #     affinity="nearest_neighbors")
    # spectral.fit(data)
    # clusters_index = spectral.labels_

    # my spectral:
    # my_spectral = Spectral(n_clusters=n_clusters)
    # my_spectral.fit(data)
    # clusters_index = my_spectral.predict(data)

    # K-means：
    # kmeans = cluster.KMeans(n_clusters=n_clusters)
    # kmeans.fit(data)
    # clusters_index = kmeans.labels_

    # My kmeans：
    # kmeans = K_Means(n_clusters=n_clusters)
    # kmeans.fit(data)
    # clusters_index = kmeans.predict(data)

    # DBSCAN:
    # cluster.DBSCAN()
    # dbscan = cluster.DBSCAN(3, 30)
    # dbscan.fit(data)
    # clusters_index = dbscan.labels_

    # # my DBSCAN
    my_dbscan = Dbscan(r=1, min_sample=20)
    my_dbscan.fit(data)
    clusters_index = my_dbscan.label_

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    root_dir = 'data/' # 数据集路径
    cat = os.listdir(root_dir)
    # cat = cat[1:]
    iteration_num = len(cat)


    '''
        # test clustering:
    '''
    filename = os.path.join(root_dir, cat[0])
    print('clustering pointcloud file:', filename)
    origin_points = read_velodyne_bin(filename)  # N*3

    # 1. 先做降采样
    points = pd.DataFrame(origin_points)
    points.columns = ['x', 'y', 'z']
    point_cloud_pynt = PyntCloud(points)

    # 调用voxel滤波函数，实现滤波
    print("num of originpoints = ", len(origin_points))
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 1)
    print("num of filtered_points = ", len(filtered_cloud))

    # do RANSAC:
    iters = 2
    segmengted_cloud = []

    # set params
    tao = 0.2
    max_iters = 35
    ground_index = []
    no_ground_index = []

    for i in range(iters):
        n, p, max_count = RANSAC(filtered_cloud, tao, max_iters)
        for idx, point in enumerate(origin_points):
            d = abs(np.dot((point - p), n))
            if d < tao:
                ground_index.append(idx)
            else:
                no_ground_index.append(idx)
                segmengted_cloud.append(point)

    segmengted_cloud = np.asarray(segmengted_cloud, dtype=np.float32)
    print('origin data points num:', origin_points.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])

    # show ground:
    points = pd.DataFrame(segmengted_cloud)
    points.columns = ['x', 'y', 'z']
    point_cloud_pynt = PyntCloud(points)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云

    # segmented_points, ground_index, no_ground_index = ground_segmentation(data=origin_points)
    # cluster_index = clustering(segmented_points)
    #
    # # save point in txt:
    # print("cluster_index: ", cluster_index[0:5])
    # np.savetxt("cluster_index.txt", cluster_index, fmt='%d')
    # np.savetxt("no_ground_index.txt", no_ground_index, fmt='%d')


    # for i in range(iteration_num):
    #     filename = os.path.join(root_dir, cat[i])
    #     print('clustering pointcloud file:', filename)
    #
    #     origin_points = read_velodyne_bin(filename)  # N*3
    #     t0 = time.time()
    #     segmented_points, ground_index, no_ground_index = ground_segmentation(data=origin_points)
    #     t1 = time.time()
    #     print("time of RANSAC: ", t1 - t0)
    #     plot_ground(origin_points, ground_index, no_ground_index)
    #
    #     # # save point in txt:
    #     # np.savetxt(cat[i] + "_ground_index.txt", ground_index, fmt='%d')
    #     # np.savetxt(cat[i] + "_no_ground_index.txt", no_ground_index, fmt='%d')
    #
    #
    #     cluster_index = clustering(segmented_points)
    #     # # save point in txt:
    #     # print("cluster_index: ", cluster_index[0:5])
    #     # np.savetxt(cat[i]+"_cluster_index.txt", cluster_index, fmt='%d')
    #
    #     plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
