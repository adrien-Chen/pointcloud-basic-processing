import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time

from ISS import ISS
from FPFH import descriptor

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

# 功能：命令行参数
# 输入：
#     None
# 输出：
#     参数
def get_arguments():
    # initial:
    parser = argparse.ArgumentParser("Description ISS keypoints on ModelNet40 dataset.")

    # add required and optional groups:
    required = parser.add_argument_group("Required")
    optional = parser.add_argument_group("Optional")

    # # add required:
    # required.add_argument(
    #     "-i", dest="input", help="Input path of ModelNet40 sample.",
    #     required=True
    # )

    required.add_argument(
        "-r", dest="radius", help="Radius for radius nearest neighbor definition.",
        required=True, type=float
    )

    return parser.parse_args()

def main():
    # # parse arguments:
    # arguments = get_arguments()

    dir = "/home/dw/adrien/cloud_lesson/data/modelnet40_normal_resampled/"

    name = 'chair'
    # 加载原始点云
    filename = dir + name + "/" + name + "_0001.txt"
    points = np.genfromtxt(filename, delimiter=",")

    # feature detection:
    pcd = get_pcd(filename)
    points = pcd.points
    root = o3d.geometry.KDTreeFlann(pcd)

    threshold = 0.4
    r = 0.2
    # t1 = time.time()
    # feature_points = ISS(root, points, r, threshold)
    # t2 = time.time()
    # print("# do detection # : ISS cost time: %.3fs" % (t2 - t1))
    # np.savetxt(name + '_key_points.txt', feature_points, fmt='%d')

    # get feature points from txt
    idx_file = name + '_key_points.txt'
    idx_feature_points = np.loadtxt(idx_file, dtype=np.int)

    # find bounding_box of similar areas:
    pcd.paint_uniform_color([0.5, 0.5, 0.5]) # background as grey
    max_bound = pcd.get_max_bound()
    min_bound = pcd.get_min_bound()

    min_bound[1] = min_bound[1]
    max_bound[1] = min_bound[1] + 0.2
    min_bound[2] = min_bound[2]
    max_bound[2] = min_bound[2] + 0.2

    bounding_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=min_bound,
        max_bound=max_bound
    )

    # find key points in the similar area:
    points = pd.DataFrame(points)
    feature_points = points.iloc[list(idx_feature_points)]
    feature_points.columns = ['x', 'y', 'z']

    feature_points_in_roi = feature_points.loc[(
            ((feature_points['x'] >= min_bound[0]) & (feature_points['x'] <= max_bound[0])) &
            ((feature_points['y'] >= min_bound[1]) & (feature_points['y'] <= max_bound[1])) &
            ((feature_points['z'] >= min_bound[2]) & (feature_points['z'] <= max_bound[2]))),
            :]

    print(len(feature_points_in_roi))

    # feature description:
    B = 6
    df_feature_description = []
    for idx in feature_points_in_roi.index:
        fpfh = descriptor(pcd, root, idx, r, B)
        df_idx = pd.DataFrame.from_dict(
            {
                'index': np.arange(len(fpfh)),
                'feature': fpfh
            }
        )
        df_idx['key_idx'] = idx
        df_feature_description.append(df_idx)
    df_feature_description = pd.concat(df_feature_description)
    df_feature_description.to_csv("description.csv")

    # draw the plot:
    plt.figure(num=None, figsize=(16, 9))

    sns.lineplot(
        x="index", y="feature",
        hue="key_idx", style="key_idx",
        markers=True, dashes=False, data=df_feature_description
    )

    plt.title('Description Visualization for Keypoints')
    plt.show()



if __name__ == '__main__':
    main()