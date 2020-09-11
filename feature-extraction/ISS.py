# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d
import numpy as np
import time
import math
import pandas as pd


# 功能：加载点云文件
# 输入：
#     filename：点云文件
# 输出：
#     pcd：point_cloud: numpy.ndarray
def get_pcd(filename):
    points = np.genfromtxt(filename, delimiter=",")
    points = pd.DataFrame(points)
    points.columns = ['x', 'y', 'z', 'nx', 'ny', 'nz']

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[['x', 'y', 'z']].values)
    pcd.normals = o3d.utility.Vector3dVector(points[['nx', 'ny', 'nz']].values)

    return pcd


# 功能：计算球体IOU：半径相等，除非圆心重合，否则相离（切）或相交
# 输入：
#     p1：球1的圆心
#     p2：球2的圆心
#     r：radius
# 输出：
#     IOU：Intersection of Union
def compute_iou(p1, p2, r):
    dist = np.linalg.norm(p1-p2)
    if dist == 0:
        iou = 1
    elif dist == 2*r or dist > 2*r:
        iou = 0
    else:
        # 1. area of overlap:
        x = dist/2.
        h = r - x
        v_overlap = math.pi * h * h * (r - h/3.) * 2
        # 2. area of union:
        v_sphere = 4 * math.pi * r * r * r / 3.
        v_union = 2 * v_sphere - v_overlap

        iou = v_overlap / v_union # should be 0~1
    return iou

# 功能：计算特征点的函数
# 输入：
#     data：点云，NX3的矩阵
#     r：radius
# 输出：
#     feature_points：特征点
def ISS(root, data, r=0.2, threshold=0.4):

    # 求特征点 特征值：
    print("start find feature points...")
    feature_points = []
    input_list = {}
    output_list = []
    for i in range(len(data)):
        # 1. 求点的radius领域
        point_i = data[i]
        [k, idx_neighs, _] = root.search_radius_vector_3d(point_i, r)  # index of neighs

        # 2. 计算加权协方差矩阵：权重 = 这里用的领域内点个数点倒数（L1/L2也OK）
        w = []
        deviation = []

        for j in idx_neighs:
            node_j = data[j]
            # s1：领域内点个数点倒数
            [k, idx_neighs_j, _] = root.search_radius_vector_3d(node_j, r)  # index of neighs_j
            w_j = 1.0 / len(idx_neighs_j)

            # s2：L1/L2
            # d_j = np.linalg.norm(node_j-point_i)
            # # print("L2: ", d_j)
            # d_j = np.exp(d_j)
            # w_j = 1. / d_j

            w.append(w_j)
            deviation.append(node_j-point_i)

            # print("L2 after exp", d_j)
            # print("w : ", w_j)


        w = np.asarray(w)
        deviation = np.asarray(deviation)
        cov = (1. / w.sum()) * np.dot(deviation.T, np.dot(np.diag(w), deviation))

        # 3. 计算特征值：从大到小
        eigenvalues, _ = np.linalg.eig(cov)
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]

        # 4. 过滤平面 或 直线的点
        v1, v2, v3 = eigenvalues
        # if (v1 > v2) and (v2 > v3):
        #     input_list[i] = v3  # {index: score}
        # 加强条件：
        if (v1 * 0.875 > v2) and (v2*0.875 > v3):
            input_list[i] = v3


    input_list = sorted(input_list.items(), key=lambda item:item[1]) # list of tuple
    print("size of input list: ", len(input_list))

    # NMS 过滤特征点：
    print("start do NMS ...")
    print("r, threshold = ", r, threshold)

    while input_list:
        key_point = input_list.pop()
        output_list.append(key_point)
        for idx, p in enumerate(input_list):
            p1 = data[key_point[0]]
            p2 = data[p[0]]
            iou = compute_iou(p1, p2, r)

            if iou > threshold:
                input_list.pop(idx)

    for p in output_list:
        feature_points.append(p[0])

    print("num of feature points: ", len(feature_points))
    return feature_points




def main():

    classes = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone",
               "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
               "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
               "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]

    dir = "/home/modelnet40_normal_resampled/"  # set for your data dir

    for name in classes[0:3]:

        # 加载原始点云
        filename = dir + name + "/" + name + "_0001.txt"
        points = np.genfromtxt(filename, delimiter=",")
        points = points[:, 0:3]

        print('total points number is:', points.shape[0])

        # feature detection:
        pcd = get_pcd(filename)
        points = pcd.points
        root = o3d.geometry.KDTreeFlann(pcd)

        threshold = 0.6
        r = 0.25
        t1 = time.time()
        feature_points = ISS(root, points, r, threshold)
        t2 = time.time()
        print("cost time: %.3fs" % (t2-t1))

        np.savetxt(name+'_key_points.txt', feature_points, fmt='%d')

        # visualize:
        # pcd.paint_uniform_color([0.95, 0.95, 0.95]) # background like grey
        # np.asarray(pcd.colors)[feature_points, :] = [1.0, 0.0, 0.0]
        # o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
