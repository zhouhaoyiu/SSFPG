# SSFPG

[English](#english) | [中文](#中文)

---

<a name="english"></a>

## English

An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions.

### Programs in this branch

This folder contains several programs for the SSFPG method:

1. SSFPG: Solves nonnegative or bound-constrained linear least-squares problems.

2. SSFPG_spar: Designed for problems in which the coefficient matrix consists of both dense and sparse components.

3. SSFPG_spar_mob: Designed for the simultaneous inversion of multiple datasets. The data vectors are combined into a matrix and inverted together. By exploiting MATLAB matrix-matrix multiplication, the inversion can be substantially accelerated.

4. SSFPG_spar_mweits: Designed for problems with multiple weights applied to the sparse constraint matrix. Solutions corresponding to different weights can be obtained simultaneously in a single inversion.

5. examples.m: Provides a numerical deconvolution test in which the four functions above are applied, and the resulting solutions are plotted for comparison.

### Python and Rust implementations

The four solvers are also implemented in Python (`python/`) and Rust (`rust/`). Each port keeps
one file per solver, mirroring the MATLAB layout:

| MATLAB | Python | Rust |
| --- | --- | --- |
| `SSFPG.m` | `python/SSFPG.py` | `rust/src/ssfpg.rs` |
| `SSFPG_spar.m` | `python/SSFPG_spar.py` | `rust/src/ssfpg_spar.rs` |
| `SSFPG_spar_mob.m` | `python/SSFPG_spar_mob.py` | `rust/src/ssfpg_spar_mob.rs` |
| `SSFPG_spar_mweits.m` | `python/SSFPG_spar_mweits.py` | `rust/src/ssfpg_spar_mweits.rs` |

Both ports reproduce the original algorithm: nonnegativity projection, optional scaling to the
best fit, the pre-computed step table from `t0_1e6_30.mat`, the safeguard against steps above
the stability bound `2/L`, and the fallback step search. `L` (the largest eigenvalue of `G'*G`,
or of `G'*G + D'*D`) and the step table are supplied by the caller, so spectral preparation
stays outside the iteration loop.

As in MATLAB, the dense and sparse variants differ only in the initial solution (`ones` versus
`zeros`). `SSFPG_spar_mweits` additionally skips the stability test, keeps a single pivot at
the largest `L`, and writes the revised steps back to the step table.

Both ports were checked against the original kernels on identical inputs: the four
implementations reproduce the same solutions, normalised misfits, used steps and revision
flags.

**Python.** Requires Python >= 3.10, `numpy` and `scipy`. Each module is self-contained — it
carries its own `Result` container and `L` estimator — and exposes a single entry point named
`solve`:

```python
import SSFPG_spar
result = SSFPG_spar.solve(G, D, ob, rawsteps, L, iterations=100, scaling=True)
```

`result` holds the best solution, the normalised misfit, the used steps, the revision flags
and, optionally, the full history. Reading `t0_1e6_30.mat` needs `h5py`, because the file is
stored in the MATLAB v7.3 (HDF5) format.

**Rust.**

```bash
cd rust
cargo build --release
```

`rust/src/lib.rs` holds only what the four solvers share: the Accelerate BLAS bindings, the
forward / gradient / candidate operator, and the panic guard. The crate builds as a `cdylib`
and currently targets macOS only, because the kernel links Apple's Accelerate framework. It
exports one C entry point per solver:

```
ssfpg_run_dense   ssfpg_run_spar   ssfpg_run_mob   ssfpg_run_mweits
```

### Accelerated variants on the `rust` branch

Three additional solvers are kept on the `rust` branch together with `README_FAST.md`; they are
not present in this branch:

| Solver | Intended use | Main acceleration | Main limitation |
| --- | --- | --- | --- |
| `SSFPG_fast.m` | General replacement for `SSFPG.m` | Single safeguarded fallback, reusable largest eigenvalue, warm start, optional history allocation | Speedup is moderate when `G` changes every solve |
| `SSFPG_gram.m` | Repeated inversions with the same `G` | Reuses `H = G'*G` and its largest eigenvalue | Normal equations can lose accuracy for severely ill-conditioned matrices |
| `SSFPG_gram_rust.m` | Repeated dense nonnegative inversions | Runs the cached Gram iteration loop in an optimized Rust MEX kernel | Full real `double` matrices and nonnegative projection only |

Synthetic benchmark on Apple M4 Max with MATLAB R2024a (dense convolution matrix, `N=2000`,
`M=1000`, fixed 100 iterations): `SSFPG_fast` 1.72x including eigenvalue estimation and 2.05x
with a cached `L`; `SSFPG_gram` 4.23x with cached `H` and `L`; `SSFPG_gram_rust` 13.11x. The
Rust MEX kernel took `0.0177 s`, and its relative solution difference from MATLAB `SSFPG_gram`
was `2.6e-8`. These numbers are synthetic and machine-specific; run the supplied benchmarks on
the target matrices before drawing conclusions for a rupture-inversion project.

### Reference

Zhang, Yong. (2026). An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions.

---

<a name="中文"></a>

## 中文

一种用于大规模非负地震震源反演的加速投影梯度方法
（An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions，
即一种快速收敛的非负最小二乘矩阵求解算法）。

### 本分支的程序

本文件夹包含实现 SSFPG 方法的若干程序：

1. **SSFPG**：求解非负或带界约束的线性最小二乘问题。

2. **SSFPG_spar**：面向系数矩阵同时包含稠密块与稀疏块的问题。

3. **SSFPG_spar_mob**：面向多组数据集的同时反演。各数据向量被合并为一个矩阵并一同反演；
   借助 MATLAB 的矩阵-矩阵乘法，反演速度可得到显著提升。

4. **SSFPG_spar_mweits**：面向对稀疏约束矩阵施加多重权重的问题。
   一次反演即可同时得到不同权重所对应的全部解。

5. **examples.m**：提供一个数值反卷积测试，其中调用了上述四个函数，并绘制所得解的对比图。

### Python 与 Rust 实现

这四个求解器另有 Python（`python/`）与 Rust（`rust/`）两份实现，都与 MATLAB 一样
**一个求解器一个文件**：

| MATLAB | Python | Rust |
| --- | --- | --- |
| `SSFPG.m` | `python/SSFPG.py` | `rust/src/ssfpg.rs` |
| `SSFPG_spar.m` | `python/SSFPG_spar.py` | `rust/src/ssfpg_spar.rs` |
| `SSFPG_spar_mob.m` | `python/SSFPG_spar_mob.py` | `rust/src/ssfpg_spar_mob.rs` |
| `SSFPG_spar_mweits.m` | `python/SSFPG_spar_mweits.py` | `rust/src/ssfpg_spar_mweits.rs` |

两份实现都保留了原算法：非负投影、可选的「缩放到最佳拟合」、来自 `t0_1e6_30.mat` 的
预计算步长表、对超过稳定性上限 `2/L` 的步长的保护，以及回退步长搜索。
`L`（`G'*G` 或 `G'*G + D'*D` 的最大特征值）与步长表由调用方给出，
因此特征值准备与迭代过程是分离的。

与 MATLAB 一致，稠密版与稀疏版的**唯一区别是初始解**（全 1 与全 0）。
`SSFPG_spar_mweits` 另外还：跳过稳定性上限判断、只保留最大 `L` 那一列作为主元、
并把修正后的步长写回步长表。

两份实现都在相同输入下与原实现做过核对：四个求解器给出的解、归一化残差、
实际使用的步长与回退标记完全一致。

**Python**：需要 Python ≥ 3.10、`numpy` 与 `scipy`。每个模块都是自包含的
（自带 `Result` 容器与 `L` 估计函数），并只暴露一个入口 `solve`：

```python
import SSFPG_spar
result = SSFPG_spar.solve(G, D, ob, rawsteps, L, iterations=100, scaling=True)
```

`result` 里包含最优解、归一化残差、实际使用的步长、回退标记，以及（可选的）完整迭代历史。
读取 `t0_1e6_30.mat` 需要 `h5py`，因为该文件是 MATLAB v7.3（HDF5）格式。

**Rust**：

```bash
cd rust
cargo build --release
```

`rust/src/lib.rs` 只放四个求解器共用的部分：Accelerate 的 BLAS 绑定、
正演 / 梯度 / 候选解算子，以及 panic 兜底。这个 crate 编译为 `cdylib`，
因为内核链接了 Apple 的 Accelerate 框架，**目前只能在 macOS 上构建**。
它按求解器导出四个 C 入口：

```
ssfpg_run_dense   ssfpg_run_spar   ssfpg_run_mob   ssfpg_run_mweits
```

### `rust` 分支上的加速版本

另有三个求解器连同 `README_FAST.md` 一起放在 `rust` 分支上，**不在本分支**：

| 求解器 | 适用场景 | 主要加速手段 | 主要限制 |
| --- | --- | --- | --- |
| `SSFPG_fast.m` | 可作为 `SSFPG.m` 的通用替代 | 单次带保护的回退、复用最大特征值、热启动、可选的历史记录分配 | 当每次求解的 `G` 都在变化时，加速幅度有限 |
| `SSFPG_gram.m` | 使用同一个 `G` 反复反演 | 复用 `H = G'*G` 及其最大特征值 | 矩阵严重病态时，正规方程可能损失精度 |
| `SSFPG_gram_rust.m` | 反复进行的稠密非负反演 | 用优化过的 Rust MEX 内核执行已缓存的 Gram 迭代循环 | 仅支持完整的实数双精度矩阵与非负投影 |

合成数据基准（Apple M4 Max + MATLAB R2024a，稠密卷积矩阵 `N=2000`、`M=1000`，固定 100 次迭代）：
`SSFPG_fast` 含特征值估计 1.72 倍、复用 `L` 后 2.05 倍；`SSFPG_gram` 复用 `H` 与 `L` 后 4.23 倍；
`SSFPG_gram_rust` 13.11 倍。Rust MEX 内核耗时 `0.0177 s`，其解与 MATLAB 版 `SSFPG_gram`
的相对差异为 `2.6e-8`。以上都是合成数据、且与具体机器相关，**不能**据此推断实际破裂反演
项目也能获得同等加速——请在目标矩阵上运行随附的基准脚本。

### 参考文献

Zhang, Yong. (2026). An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions.
