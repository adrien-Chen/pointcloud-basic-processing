import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import seaborn as sns

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

def main():
    # 加载原始点云
    name = "chair"
    dir = "./"
    filename = dir + name + "_0001.txt"
    points = np.genfromtxt(filename, delimiter=",")

    pcd = get_pcd(filename)

    # get feature points from txt
    idx_file = name + '_key_points.txt'
    idx_feature_points = np.loadtxt(idx_file, dtype=np.int)

    # visualize:
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # background as grey
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

    roi = pcd.crop(bounding_box)
    roi.paint_uniform_color([1.00, 0.00, 0.00])

    # o3d.visualization.draw_geometries([roi])

    # find key points in the similar area:
    points = pcd.points
    points = pd.DataFrame(points)

    feature_points = points.iloc[list(idx_feature_points)]

    feature_points.columns = ['x', 'y', 'z']
    print(feature_points.iloc[0:3])
    feature_points_in_roi = feature_points.loc[(
            ((feature_points['x'] >= min_bound[0]) & (feature_points['x'] <= max_bound[0])) &
            ((feature_points['y'] >= min_bound[1]) & (feature_points['y'] <= max_bound[1])) &
            ((feature_points['z'] >= min_bound[2]) & (feature_points['z'] <= max_bound[2]))),
            :]

    print(len(feature_points_in_roi))
    np.asarray(pcd.colors)[feature_points_in_roi.index.tolist(), :] = [1.0, 0, 0]
    o3d.visualization.draw_geometries([pcd])

    df_feature_description = pd.read_csv("description.csv")
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