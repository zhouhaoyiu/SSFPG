# SSFPG
An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions. (一种快速收敛的非负最小二乘矩阵求解算法)

This folder contains several programs for the SSFPG method:

1. SSFPG: Solves nonnegative or bound-constrained linear least-squares problems.

2. SSFPG_spar: Designed for problems in which the coefficient matrix consists of both dense and sparse components.

3. SSFPG_spar_mob: Designed for the simultaneous inversion of multiple datasets. The data vectors are combined into a matrix and inverted together. By exploiting MATLAB matrix-matrix multiplication, the inversion can be substantially accelerated.

4. SSFPG_spar_mweits: Designed for problems with multiple weights applied to the sparse constraint matrix. Solutions corresponding to different weights can be obtained simultaneously in a single inversion.

5. examples.m: Provides a numerical deconvolution test in which the four functions above are applied, and the resulting solutions are plotted for comparison.

Reference
Zhang, Yong. (2026). An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions.