import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle, islice


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

def plot_ground(data, ground_index, no_ground_index):
    ax = plt.figure().add_subplot(111, projection='3d')
    print(data.shape)
    ax.scatter(data[ground_index, 0], data[ground_index, 1], data[ground_index, 2], s=2, c='b')
    ax.scatter(data[no_ground_index, 0], data[no_ground_index, 1], data[no_ground_index, 2], s=2, c='0.8')
    plt.show()

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
    root_dir = 'data'  # 数据集路径
    cat = os.listdir(root_dir)
    iteration_num = len(cat)


    # # test:
    filename = os.path.join(root_dir, "000001.bin")
    print(filename)
    origin_points = read_velodyne_bin(filename)  # N*3

    # ground_index = np.loadtxt("ground_index.txt", dtype=np.int)
    # no_ground_index = np.loadtxt("no_ground_index.txt", dtype=np.int)
    # plot_ground(origin_points, ground_index, no_ground_index)

    cluster_index = np.loadtxt("cluster_index.txt", dtype=np.int)
    no_ground_index = np.loadtxt("no_ground_index.txt", dtype=np.int)
    data = origin_points[no_ground_index]
    plot_clusters(data, cluster_index)


    # for i in range(iteration_num):
    #     filename = os.path.join(root_dir, cat[i])
    #     print('Fitting pointcloud file:', filename)
    #     origin_points = read_velodyne_bin(filename)  # N*3
    #
    #     cluster_index = np.loadtxt(cat[i]+"_cluster_index.txt", dtype=np.int)
    #     no_ground_index = np.loadtxt(cat[i]+"_no_ground_index.txt", dtype=np.int)
    #     data = origin_points[no_ground_index]
    #     plot_clusters(data, cluster_index)



if __name__ == '__main__':
    main()