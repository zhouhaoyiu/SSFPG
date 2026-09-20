# SSFPG

一种用于大规模非负地震震源反演的加速投影梯度方法
（An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions，
即一种快速收敛的非负最小二乘矩阵求解算法）

本文件夹包含实现 SSFPG 方法的若干程序：

1. **SSFPG**：求解非负或带界约束的线性最小二乘问题。

2. **SSFPG_spar**：面向系数矩阵同时包含稠密块与稀疏块的问题。

3. **SSFPG_spar_mob**：面向多组数据集的同时反演。各数据向量被合并为一个矩阵并一同反演；
   借助 MATLAB 的矩阵-矩阵乘法，反演速度可得到显著提升。

4. **SSFPG_spar_mweits**：面向对稀疏约束矩阵施加多重权重的问题。
   一次反演即可同时得到不同权重所对应的全部解。

5. **examples.m**：提供一个数值反卷积测试，其中调用了上述四个函数，
   并绘制所得解的对比图。

## 参考文献

Zhang, Yong. (2026). An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions.
