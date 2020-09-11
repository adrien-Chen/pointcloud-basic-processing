import numpy as np
import pandas as pd
import open3d as o3d

# 功能：计算特征点的spfh
# 输入：
#     pcd：点云
#     root：search_tree
#     key_point_id ：关键点的id
#     r：radius
#     B：直方图bin的个数
# 输出：
#     spfh：simplified pfh
def do_spfh(pcd, root, key_point_id, r, B):
    # 1. rnn 搜索邻居：
    data = np.asarray(pcd.points)
    key_point = data[key_point_id]
    [k, idx_neighs, _] = root.search_radius_vector_3d(key_point, r)
    # remove query point:
    idx_neighs = idx_neighs[1:]

    # 2. 计算表面属性的三元组：（alpha phi theta）：
    normals = np.asarray(pcd.normals)
    n1 = normals[key_point_id]
    u = n1
    diff = data[idx_neighs] - key_point
    diff /= np.linalg.norm(diff, ord=2, axis=1)[:, None]
    v = np.cross(u, diff) # shape(k, 3)
    w = np.cross(u, v)

    n2 = normals[idx_neighs]
    alpha = (v * u).sum(axis=1)
    phi = (u * diff).sum(axis=1)
    theta = np.arctan2(((w*n2).sum(axis=1)), (u*n2).sum(axis=1))

    # 3. 连接三个直方图并返回：
    # np.histogram return 2 array : hist and bin_edges
    alpha_h = np.histogram(alpha, bins=B, range=(-1.0, 1.0))[0] # just need hist, alpha is cos,should in (-1,1)
    alpha_h = alpha_h / alpha_h.sum() # do histogram normalize

    phi_h = np.histogram(phi, bins=B, range=(-1.0, 1.0))[0]
    phi_h = phi_h / phi_h.sum()

    theta_h = np.histogram(theta, bins=B, range=(-np.pi, np.pi))[0] # angel should in (-pi, pi)
    theta_h = theta_h / theta_h.sum()

    # concat 3 histogram and return:
    spfh = np.hstack((alpha_h, phi_h, theta_h))
    return spfh

def descriptor(pcd, root, key_point_id, r, B):
    # 1. 计算邻居们的spfh
    data = np.asarray(pcd.points)
    key_point = data[key_point_id]
    [k, idx_neighs, _] = root.search_radius_vector_3d(key_point, r)
    if k <= 1:
        return None
    # remove query point:
    idx_neighs = idx_neighs[1:]

    # get weight:
    w = 1.0 / np.linalg.norm(key_point-data[idx_neighs], ord=2, axis=1)

    # spfh of neighbor points:
    X = np.asarray([do_spfh(pcd, root, idx, r, B) for idx in idx_neighs])
    spfh_neighs = 1.0 / (k-1) * np.dot(w, X)

    # 2. 计算关键点的spfh
    spfh_key_point = do_spfh(pcd, root, key_point_id, r, B)

    # 3. 计算FPSH：对spfh的加权求和
    fpsh = spfh_key_point + spfh_neighs
    fpsh = fpsh / np.linalg.norm(fpsh) # normalize

    return fpsh
    