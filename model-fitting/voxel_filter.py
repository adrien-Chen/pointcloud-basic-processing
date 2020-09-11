# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
import math
import random

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []

    # 1. compute boundary:
    x_max = point_cloud.loc[:, 'x'].max()
    x_min = point_cloud.loc[:, 'x'].min()
    y_max = point_cloud.loc[:, 'y'].max()
    y_min = point_cloud.loc[:, 'y'].min()
    z_max = point_cloud.loc[:, 'z'].max()
    z_min = point_cloud.loc[:, 'z'].min()

    # 2. compute dim of voxel grid:
    d_x = (x_max-x_min)/leaf_size
    d_y = (y_max-y_min)/leaf_size

    # 3. compute index for each point:
    for index, row in point_cloud.iterrows():
        # floor, because voxel index begin from 0
        point_cloud.loc[index, 'h_x'] = h_x = math.floor((row['x']-x_min)/leaf_size)
        point_cloud.loc[index, 'h_y'] = h_y = math.floor((row['y']-y_min)/leaf_size)
        point_cloud.loc[index, 'h_z'] = h_z = math.floor((row['z']-z_min)/leaf_size)
        point_cloud.loc[index, 'h'] = math.floor(h_x + h_y*d_x + h_z*d_x*d_y)  # also floor

    # 4. sort the points with the voxel index:
    points_sorted = point_cloud.sort_values(by='h').loc[:, ['x', 'y', 'z', 'h']]
    points_sorted.index = range(len(points_sorted))  # after sorted, index have changed

    # 5. select points:

    # solution 1: centroid:
    count = points_sorted.loc[0, 'h']
    x_, y_, z_ = points_sorted.loc[0, ['x', 'y', 'z']]
    num = 1
    for index, row in points_sorted.iterrows():
        if row['h'] == count and index != 0:
            x_ += row['x']
            y_ += row['y']
            z_ += row['z']
            num += 1
            continue
        elif index != 0:
            filtered_points.append([x_ / num, y_ / num, z_ / num])
            count = row['h']
            x_, y_, z_ = row['x'], row['y'], row['z']
            num = 1
    # add last point:
    filtered_points.append([x_ / num, y_ / num, z_ / num])

    # solution 2: random:
    # start = end = 0
    # count = points_sorted.loc[0, 'h']
    # for index, row in points_sorted.iterrows():
    #     if row['h'] == count:
    #         end = index
    #         continue
    #     idx = random.randint(start, end)
    #     filtered_points.append(points_sorted.loc[idx, ['x', 'y', 'z']])
    #     start = end = index
    #     count = row['h']
    # idx = random.randint(start, end)
    # filtered_points.append(points_sorted.loc[idx, ['x', 'y', 'z']])

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

# 功能：对点云进行voxel approximated滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
#     contain_size: 容器个数
def voxel_filter_approximated(point_cloud, leaf_size, contain_size):
    filtered_points = []

    # 1. compute boundary:
    x_max = point_cloud.loc[:, 'x'].max()
    x_min = point_cloud.loc[:, 'x'].min()
    y_max = point_cloud.loc[:, 'y'].max()
    y_min = point_cloud.loc[:, 'y'].min()
    z_max = point_cloud.loc[:, 'z'].max()
    z_min = point_cloud.loc[:, 'z'].min()

    # 2. compute dim of voxel grid: length should be a int
    d_x = math.ceil((x_max-x_min)/leaf_size)
    d_y = math.ceil((y_max-y_min)/leaf_size)

    # 3. compute index for each point:
    for index, row in point_cloud.iterrows():
        # floor, because voxel index begin from 0
        point_cloud.loc[index, 'h_x'] = h_x = math.floor((row['x']-x_min)/leaf_size)
        point_cloud.loc[index, 'h_y'] = h_y = math.floor((row['y']-y_min)/leaf_size)
        point_cloud.loc[index, 'h_z'] = h_z = math.floor((row['z']-z_min)/leaf_size)
        # compute index also floor:
        point_cloud.loc[index, 'h'] = math.floor(h_x + h_y*d_x + h_z*d_x*d_y)

    # 4. do hash map and solve hash conflict:
    # conflict happened when hash_idx and h are both same!
    containers = [[] for i in range(contain_size)]
    for index, row in point_cloud.iterrows():
        voxel_idx = row['h']
        hash_idx = int(row['h'] % contain_size)  # index should be int!

        # if conflict happened, select one them empty:
        if containers[hash_idx] and voxel_idx != point_cloud.loc[containers[hash_idx][0], 'h']:
            # select one point: centroid(or random)
            if len(containers[hash_idx]) > 1:
                point = [point_cloud.loc[containers[hash_idx], 'x'].mean(0),
                         point_cloud.loc[containers[hash_idx], 'y'].mean(0),
                         point_cloud.loc[containers[hash_idx], 'z'].mean(0)]
            else:
                point = [point_cloud.loc[containers[hash_idx], 'x'],
                         point_cloud.loc[containers[hash_idx], 'y'],
                         point_cloud.loc[containers[hash_idx], 'z']]
            filtered_points.append(point)
            # empty this container:
            containers[hash_idx] = []
        containers[hash_idx].append(index)

    # final output all points in containers:
    while containers:
        points_idx = containers.pop()
        point = [point_cloud.loc[points_idx, 'x'].mean(0),
                 point_cloud.loc[points_idx, 'y'].mean(0),
                 point_cloud.loc[points_idx, 'z'].mean(0)]
        filtered_points.append(point)

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    print("before filter: ", point_cloud.shape[0])
    print("after filter: ", len(filtered_points))
    print(filtered_points[0:5])
    return filtered_points

def main():
    classes = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone",
               "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
               "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
               "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]

    dir = "./modelnet40_normal_resampled/"

    for name in classes:

        filename = dir + name + "/" + name + "_0001.txt"
        points = np.genfromtxt(filename, delimiter=",")
        # points = np.genfromtxt("./modelnet40_normal_resampled/airplane/airplane_0001.txt", delimiter=",")
        points = pd.DataFrame(points[:, 0:3])
        points.columns = ['x', 'y', 'z']
        point_cloud_pynt = PyntCloud(points)

        # 转成open3d能识别的格式
        point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
        # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

        # 调用voxel滤波函数，实现滤波
        # filtered_cloud = voxel_filter(point_cloud_pynt.points, 0.1)  # 100 cause d_x,d_y too samll

        # 调用voxel approximated 滤波函数，实现滤波
        contain_size = 200
        filtered_cloud = voxel_filter_approximated(point_cloud_pynt.points, 0.1, contain_size)
        point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
        # 显示滤波后的点云
        o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
