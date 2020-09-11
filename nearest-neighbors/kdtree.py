# kdtree的具体实现，包括构建和查找

import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet

# Node类，Node是tree的基本组成元素
class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output

# 功能：构建树之前需要对value进行排序，同时对一个的key的顺序也要跟着改变
# 输入：
#     key：键
#     value:值
# 输出：
#     key_sorted：排序后的键
#     value_sorted：排序后的值
def sort_key_by_vale(key, value):
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted

'''
快速选择算法：
# 输入：
#     key：键
#     value:值
# 输出：
#     key_middle：中值的键
#     value_middle：中值的值
'''

# 划分算法：
def partition(start, end, value):
    if start == end:
        return start

    p_value = value[start]

    while start < end:
        while start < end and value[end] >= p_value:
            end -= 1
        value[start] = value[end]

        while start < end and value[start] <= p_value:
            start += 1
        value[end] = value[start]

    value[start] = p_value
    return start


def quick_select(value):
    middle_idx = math.ceil(value.shape[0] / 2) - 1
    i, j = 0, value.shape[0]-1

    while i <= j:
        partition_idx = partition(i, j, value)
        if partition_idx == middle_idx:
            return value[middle_idx]
        elif partition_idx < middle_idx:
            i = partition_idx + 1
        else:
            j = partition_idx - 1


# 计算下一次要切的轴～
def axis_round_robin(axis, dim):
    if axis == dim-1:
        return 0
    else:
        return axis + 1

# 功能：通过递归的方式构建树
# 输入：
#     root: 树的根节点
#     db: 点云数据
#     point_indices：排序后的键
#     axis: scalar
#     leaf_size: scalar
# 输出：
#     root: 即构建完成的树
def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        # solution 1: sorted with O(nlogn):
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # M
        
        # 1. 找切割点（排序中值）：由于切的时候，不经过点，需要对中间左右两个点平均～
        # left point:
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1
        middle_left_point_idx = point_indices_sorted[middle_left_idx]
        middle_left_point = db[middle_left_point_idx, axis]
        # right point:
        middle_right_idx = math.ceil(point_indices_sorted.shape[0] / 2) + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx]
        middle_right_point = db[middle_right_point_idx, axis]

        # set value:
        root.value = (middle_left_point + middle_right_point) * 0.5
        # --- get the split position ---

        # 2. 递归左右子树～
        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           point_indices_sorted[0:middle_right_idx],
                                           axis_round_robin(axis, dim=db.shape[1]),
                                           leaf_size)
        root.right = kdtree_recursive_build(root.right,
                                            db,
                                            point_indices_sorted[middle_right_idx:],
                                            axis_round_robin(axis, dim=db.shape[1]),
                                            leaf_size)

    return root

# without sort:
def kdtree_recursive_build_2(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        # solution 2: sorted with O(n): quick_select~
        # 1. 找切割点（找中值）：快速选择算法
        root.value = quick_select(db[point_indices, axis])

        # 2. 递归左右子树～
        point_indices_left = [idx for idx in point_indices if db[idx][axis] <= root.value]
        point_indices_right = [idx for idx in point_indices if db[idx][axis] > root.value]
        root.left = kdtree_recursive_build_2(root.left,
                                             db,
                                             point_indices_left,
                                             axis_round_robin(axis, dim=db.shape[1]),
                                             leaf_size)
        root.right = kdtree_recursive_build_2(root.right,
                                              db,
                                              point_indices_right,
                                              axis_round_robin(axis, dim=db.shape[1]),
                                              leaf_size)
    return root


# 功能：翻转一个kd树
# 输入：
#     root：kd树
#     depth: 当前深度
#     max_depth：最大深度
def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1

# 功能：构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
# 输入：
#     db_np：原始数据
#     leaf_size：scale
# 输出：
#     root：构建完成的kd树
def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root

# without sort:
def kdtree_construction_2(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build_2(root,
                                    db_np,
                                    np.arange(N),
                                    axis=0,
                                    leaf_size=leaf_size)
    return root


# 功能：通过kd树实现knn搜索，即找出最近的k个近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set：搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 左小右大，递归直到叶子节点；是否回溯另一侧，根据射程范围确定～
    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)

    return False

# 功能：通过kd树实现radius搜索，即找出距离radius以内的近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set:搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 和KNN一样，只是worstdist = radius 是固定的，但是都用worstDist()封装了，代码就完全一样了
    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)

    return False



def main():
    # configuration
    db_size = 64
    dim = 3
    leaf_size = 4
    k = 1

    db_np = np.random.rand(db_size, dim)

    begin_t = time.time()
    root = kdtree_construction(db_np, leaf_size=leaf_size)
    print("construction with sort: %.3f", (time.time() - begin_t)*1000)

    begin_t = time.time()
    root2 = kdtree_construction_2(db_np, leaf_size=leaf_size)
    print("construction without sort: %.3f", (time.time() - begin_t)*1000)

    # depth = [0]
    # max_depth = [0]
    # traverse_kdtree(root, depth, max_depth)
    # print("tree max depth: %d" % max_depth[0])

    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # knn_search(root, db_np, result_set, query)
    #
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])
    #
    #
    print("Radius search:")
    query = np.asarray([0, 0, 0])
    result_set = RadiusNNResultSet(radius=0.5)
    result_set2 = RadiusNNResultSet(radius=0.5)

    begin_t = time.time()
    kdtree_radius_search(root, db_np, result_set, query)
    print("search time with sort: %.3fms", (time.time() - begin_t)*1000)
    print(result_set)

    begin_t = time.time()
    kdtree_radius_search(root2, db_np, result_set2, query)
    print("search time without sort: %.3fms", (time.time() - begin_t)*1000)
    print(result_set2)


if __name__ == '__main__':
    main()