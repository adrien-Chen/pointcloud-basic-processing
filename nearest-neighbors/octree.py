# octree的具体实现，包括构建和查找

import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet

# 节点，构成OCtree的基本元素
class Octant:
    def __init__(self, children, center, extent, point_indices, is_leaf):
        self.children = children
        self.center = center
        self.extent = extent
        self.point_indices = point_indices
        self.is_leaf = is_leaf

    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "
        output += 'point_indices: ' + str(self.point_indices)
        return output

# 功能：翻转octree
# 输入：
#     root: 构建好的octree
#     depth: 当前深度
#     max_depth：最大深度
def traverse_octree(root: Octant, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root is None:
        pass
    elif root.is_leaf:
        print(root)
    else:
        for child in root.children:
            traverse_octree(child, depth, max_depth)
    depth[0] -= 1

# 功能：通过递归的方式构建octree
# 输入：
#     root：根节点
#     db：原始数据
#     center: 中心
#     extent: 当前分割区间
#     point_indices: 点的key
#     leaf_size: scale
#     min_extent: 最小分割区间
def octree_recursive_build(root, db, center, extent, point_indices, leaf_size, min_extent):
    if len(point_indices) == 0:
        return None

    if root is None:
        root = Octant([None for i in range(8)], center, extent, point_indices, is_leaf=True)

    # determine whether to split this octant
    if len(point_indices) <= leaf_size or extent <= min_extent:
        root.is_leaf = True
    else:
        root.is_leaf = False
        # 1. split points into different sub-octant:
        children_point_indices = [[] for i in range(8)]
        for point_idx in point_indices:
            point_db = db[point_idx]
            # 莫顿码，从上到下,Z字形走向索引：可以将多维数据（索引的二进制表示），转化为一维编码（十进制）
            morton_code = 0
            if point_db[0] > center[0]:
                morton_code = morton_code | 1
            if point_db[1] > center[1]:
                morton_code = morton_code | 2
            if point_db[2] > center[2]:
                morton_code = morton_code | 4
            children_point_indices[morton_code].append(point_idx)

        # 2. create children:
        factor = [-0.5, 0.5] # 中心点位移系数
        for i in range(8):
            # & 按位与：1，2，4 分别是001，010，100
            # 与操作后分别取索引二进制的1，2，3位的值：0 或 1
            # 值为1 表示分别在 center 的x,y,z方向的正方向侧
            child_center_x = center[0] + factor[(i & 1) > 0] * extent
            child_center_y = center[1] + factor[(i & 2) > 0] * extent
            child_center_z = center[2] + factor[(i & 4) > 0] * extent
            child_extent = extent * 0.5
            child_center = np.asarray([child_center_x, child_center_y, child_center_z])
            root.children[i] = octree_recursive_build(root.children[i],
                                                      db,
                                                      child_center,
                                                      child_extent,
                                                      children_point_indices[i],
                                                      leaf_size,
                                                      min_extent)
    return root

# 功能：判断当前query区间是否在octant内
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def inside(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball is inside the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    possible_space = query_offset_abs + radius
    return np.all(possible_space < octant.extent)

# 功能：判断当前query区间是否和octant有重叠部分
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def overlaps(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball overlaps with the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    # completely outside, since query is outside the relevant area
    max_dist = radius + octant.extent
    if np.any(query_offset_abs > max_dist):
        return False

    # if pass the above check, consider the case that the ball is contacting the face of the octant
    if np.sum((query_offset_abs < octant.extent).astype(np.int)) >= 2:
        return True

    # conside the case that the ball is contacting the edge or corner of the octant
    # since the case of the ball center (query) inside octant has been considered,
    # we only consider the ball center (query) outside octant
    x_diff = max(query_offset_abs[0] - octant.extent, 0)
    y_diff = max(query_offset_abs[1] - octant.extent, 0)
    z_diff = max(query_offset_abs[2] - octant.extent, 0)

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius


def overlaps_plus(query: np.ndarray, inv_radius: float, octant: Octant):
    xyz_min = octant.center - octant.extent
    xyz_max = octant.center + octant.extent

    t_enter = np.array(xyz_min - query) * inv_radius
    t_exit = np.array(xyz_max - query) * inv_radius

    return np.sum((t_enter < t_exit).astype(np.int) & (t_exit >= 0).astype(np.int) & (t_enter <= 1).astype(np.int)) >= 2



# 功能：判断当前query是否包含octant
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def contains(query: np.ndarray, radius: float, octant:Octant):
    """
    Determine if the query ball contains the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    query_offset_to_farthest_corner = query_offset_abs + octant.extent
    return np.linalg.norm(query_offset_to_farthest_corner) < radius

# 功能：在octree中查找信息
# 输入：
#    root: octree
#    db：原始数据
#    result_set: 索引结果
#    query：索引信息
def octree_radius_search_fast(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    # 1. if query ball contains the octant, no need to check child, just compare all point in it!
    # 只是不需要向下递归，这里的child是只当前节点的下一层，回溯还是有可能的！所以这里return的false是有原因的嗷！
    if contains(query, result_set.worstDist(), root):
        # compare all points:
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # no need to check child
        return False

    # consider leaf point:
    if root.is_leaf and len(root.point_indices) > 0:
        # compare all points:
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return inside(query, result_set.worstDist(), root)

    # 2. check all children
    for c, child in enumerate(root.children):
        if child is None:
            continue
        if False == overlaps(query, result_set.worstDist(), child):
            continue
        if octree_radius_search_fast(child, db, result_set,query):
            return True

    return inside(query, result_set.worstDist(), root)

# version plus:
def octree_radius_search_plus(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 提前结束：核心-八叉树对3个维度有限制；
    # 结束条件：其实是两个方向：向下递归，向上回溯
    #   1-当一个节点返回True时，表示找到knn了，维度限制导致不会有更优的了，不再向下，立即结束；
    #   2-最坏距离球如果 inside 当前节点，不需要再向上回溯检查其他节点，立即结束；

    # 跳过条件：在检查
    # 1. search the first relevant child:
    # 找到最近的孩子，根据查询点的莫顿码
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4
    if octree_knn_search(root.children[morton_code],
                         db,
                         result_set,
                         query):
        return True

    # 2. check other children
    inv_worstDist = 1 / result_set.worstDist()
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if False == overlaps_plus(query, inv_worstDist, child):
            continue
        if octree_knn_search(child, db, result_set, query):
            return True

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)


# 功能：在octree中查找radius范围内的近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_radius_search(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 提前结束：核心-八叉树对3个维度有限制；
    # 结束条件：其实是两个方向：向下递归，向上回溯
    #   1-当一个节点返回True时，表示找到knn了，维度限制导致不会有更优的了，不再向下，立即结束；
    #   2-最坏距离球如果 inside 当前节点，不需要再向上回溯检查其他节点，立即结束；

    # 跳过条件：在检查
    # 1. search the first relevant child:
    # 找到最近的孩子，根据查询点的莫顿码
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4
    if octree_knn_search(root.children[morton_code],
                         db,
                         result_set,
                         query):
        return True

    # 2. check other children
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if False == overlaps(query, result_set.worstDist(), child):
            continue
        if octree_knn_search(child, db, result_set, query):
            return True

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)

# 功能：在octree中查找最近的k个近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_knn_search(root: Octant, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 提前结束：核心-八叉树对3个维度有限制；
    # 结束条件：其实是两个方向：向下递归，向上回溯
    #   1-当一个节点返回True时，表示找到knn了，维度限制导致不会有更优的了，不再向下，立即结束；
    #   2-最坏距离球如果 inside 当前节点，不需要再向上回溯检查其他节点，立即结束；

    # 跳过条件：在检查
    # 1. search the first relevant child:
    # 找到最近的孩子，根据查询点的莫顿码
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4
    if octree_knn_search(root.children[morton_code],
                         db,
                         result_set,
                         query):
        return True

    # 2. check other children
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if False == overlaps(query, result_set.worstDist(), child):
            continue
        if octree_knn_search(child, db, result_set, query):
            return True
            
    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)

# 功能：构建octree，即通过调用octree_recursive_build函数实现对外接口
# 输入：
#    dp_np: 原始数据
#    leaf_size：scale
#    min_extent：最小划分区间
def octree_construction(db_np, leaf_size, min_extent):
    N, dim = db_np.shape[0], db_np.shape[1]
    db_np_min = np.amin(db_np, axis=0)
    db_np_max = np.amax(db_np, axis=0)
    db_extent = np.max(db_np_max - db_np_min) * 0.5
    db_center = db_np_min + db_extent

    root = None
    root = octree_recursive_build(root, db_np, db_center, db_extent, list(range(N)),
                                  leaf_size, min_extent)

    return root

def main():
    # configuration
    db_size = 64000
    dim = 3
    leaf_size = 4
    min_extent = 0.0001
    k = 8

    db_np = np.random.rand(db_size, dim)

    root = octree_construction(db_np, leaf_size, min_extent)

    # depth = [0]
    # max_depth = [0]
    # traverse_octree(root, depth, max_depth)
    # print("tree max depth: %d" % max_depth[0])

    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # octree_knn_search(root, db_np, result_set, query)
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])

    # begin_t = time.time()
    # print("Radius search normal:")
    # for i in range(100):
    #     query = np.random.rand(3)
    #     result_set = RadiusNNResultSet(radius=0.5)
    #     octree_radius_search(root, db_np, result_set, query)
    # # print(result_set)
    # print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))

    # begin_t = time.time()
    # print("Radius search fast:")
    # for i in range(100):
    #     query = np.random.rand(3)
    #     result_set = RadiusNNResultSet(radius = 0.5)
    #     octree_radius_search_fast(root, db_np, result_set, query)
    # # print(result_set)
    # print("Search takes %.3fms\n" % ((time.time() - begin_t)*1000))

    query = np.random.rand(3)
    result_set1 = RadiusNNResultSet(radius=0.5)
    result_set2 = RadiusNNResultSet(radius=0.5)
    result_set3 = RadiusNNResultSet(radius=0.5)


    print("Radius search normal:")
    begin_t = time.time()
    octree_radius_search(root, db_np, result_set1, query)
    # print(result_set1)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))

    print("Radius search fast:")
    begin_t = time.time()
    octree_radius_search_fast(root, db_np, result_set2, query)
    # print(result_set2)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))

    print("Radius search normal plus:")
    begin_t = time.time()
    octree_radius_search_plus(root, db_np, result_set3, query)
    # print(result_set3)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))


if __name__ == '__main__':
    main()