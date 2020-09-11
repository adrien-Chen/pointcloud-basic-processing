# pointcloud-basic-processing
关于点云的一些基础性、传统处理方法的学习实现和总结，希望能帮助到大家～

数据集主要包括：ModelNet40、KITTI

主要对以下算法进行学习和实现，

- [PCA](/Users/Adrienchen/Desktop/点云/pointcloud-basic-processing/pca)

  - 对ModelNet40数据做PCA

  ![pca](/pca.png)

  - 对ModelNet40数据做voxel filter

![](/voxel_filter.png)

- [Nearest Neighbors Algorithm](/Users/Adrienchen/Desktop/点云/pointcloud-basic-processing/nearest-neighbors)

  - KD-Tree实现
  - Octree实现
  - KNN、Radius-NN实现

- [Clustering](/Users/Adrienchen/Desktop/点云/pointcloud-basic-processing/clustering)

  - K-means
  - GMM
  - Spectral

  compare ours implements with sklearn：show spectral like this

  ![](/show_spectral.png)

- [Model Fitting](/Users/Adrienchen/Desktop/点云/pointcloud-basic-processing/model-fitting)

  - LSQ（最小二乘法）
  - RANSAC
  - 实现简单的地面分割（[more results](/Users/Adrienchen/Desktop/点云/pointcloud-basic-processing/model-fitting/result-imgs)）

![](/show_ground_seg.png)

- [Feature Extraction](/Users/Adrienchen/Desktop/点云/pointcloud-basic-processing/feature-extraction): 
  - implete ISS
- [Feature Description](/Users/Adrienchen/Desktop/点云/pointcloud-basic-processing/feature-description)  
  - implete FPFH

![](/show_description.png)

- [Supplementary](/Users/Adrienchen/Desktop/点云/pointcloud-basic-processing/supplementary-notes)
  - 数学基础
  - GNN简介
  - 知识点总结









