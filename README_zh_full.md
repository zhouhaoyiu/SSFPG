# SSFPG

An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions

一种面向大规模非负震源反演的快速收敛投影梯度算法

本分支（fork）保留上游原始 SSFPG 求解器，并在 `rust` 分支上新增了三条实验性加速路线。
上游项目为 [ZhangYongGroupPKU/SSFPG](https://github.com/ZhangYongGroupPKU/SSFPG)。

加速相关的源码、测试与基准测试维护在 [`rust`](https://github.com/zhouhaoyiu/SSFPG/tree/rust) 分支：

```bash
git fetch origin
git switch rust
```

## 包含的求解器

上游原始程序保持不变：

- `SSFPG.m`：非负或带界约束的线性最小二乘。
- `SSFPG_spar.m`：系数矩阵由稠密块与稀疏块组成的情形。
- `SSFPG_spar_mob.m`：多组数据集同时反演。
- `SSFPG_spar_mweits.m`：一次求得多个稀疏约束权重所对应的解。
- `examples.m`：面向原始求解器的数值反卷积示例。

`rust` 分支新增：

| 求解器 | 适用场景 | 主要加速手段 | 主要限制 |
| --- | --- | --- | --- |
| `SSFPG_fast.m` | 可作为 `SSFPG.m` 的通用替代 | 单次带保护的回退、复用最大特征值、热启动、可选的历史记录分配 | 当每次求解的 `G` 都在变化时，加速幅度有限 |
| `SSFPG_gram.m` | 使用同一个 `G` 反复反演 | 复用 `H = G'*G` 及其最大特征值 | 矩阵严重病态时，正规方程可能损失精度 |
| `SSFPG_gram_rust.m` | 反复进行的稠密非负反演 | 用优化过的 Rust MEX 内核执行已缓存的 Gram 迭代循环 | 仅支持完整的实数双精度矩阵与非负投影 |

## MATLAB 快速上手

```matlab
project = @(x) max(x,0);

% 直接梯度实现；当 G 不变时可复用 L。
[x1,misfit1,~,~,~,~,L] = SSFPG_fast(G,b1,1e-8,1000,project,1);
[x2,misfit2] = SSFPG_fast(G,b2,1e-8,1000,project,1,L,x1);

% Gram 实现；当 G 不变时可复用 H 与 L。
[x3,misfit3,~,~,~,~,cache] = SSFPG_gram(G,b1,1e-8,1000,project,1);
[x4,misfit4] = SSFPG_gram(G,b2,1e-8,1000,project,1,cache,x3);
```

`SSFPG_fast` 与 `SSFPG_gram` 同时保留了原有的「投影指令字符串」调用方式。

## Rust MEX 快速上手

先安装 Rust，用 `mex -setup C` 配置 MATLAB 支持的 C 编译器，然后运行：

```matlab
build_SSFPG_gram_rust

[x1,misfit1,~,~,~,~,cache] = SSFPG_gram_rust(G,b1,1e-8,1000,1);
[x2,misfit2] = SSFPG_gram_rust(G,b2,1e-8,1000,1,cache,x1);
```

构建过程直接调用 `rustc`，不依赖任何第三方 Rust 包。它会在本地生成平台相关的 MEX 二进制文件；
二进制文件不入库。

## 已验证的合成数据基准

环境：Apple M4 Max、MATLAB R2024a，稠密卷积矩阵 `N=2000`、`M=1000`，固定 100 次迭代：

| 实现 | 相对原始 `SSFPG` 的加速比 |
| --- | ---: |
| `SSFPG_fast`（含特征值估计） | 1.72x |
| `SSFPG_fast`（复用 L） | 2.05x |
| `SSFPG_gram`（复用 H 与 L） | 4.23x |
| `SSFPG_gram_rust`（复用 H 与 L） | 13.11x |

在 Rust 对比中，MEX 内核耗时 `0.0177 s`，比 MATLAB 版 `SSFPG_gram` 快 2.60 倍。
其目标函数值与 MATLAB 版 `SSFPG_gram` 在打印精度内一致；解的相对差异为 `2.6e-8`，
投影梯度残差为 `1.0e-9`。

以上均为合成数据、且与具体机器相关，**不能**据此推断在实际破裂反演项目中也能获得同等加速。
请在目标矩阵上运行随附的基准脚本，并比较收敛性、波形拟合、地震矩、滑动分布与停止容差。

## 验证

```matlab
test_SSFPG_fast
test_SSFPG_gram
test_SSFPG_gram_rust

benchmark_SSFPG_fast
benchmark_SSFPG_gram
benchmark_SSFPG_gram_rust
```

在条件数从 `10` 到 `1e8` 的构造算例中，MATLAB 版与 Rust 版 Gram 求解器的残差一致。
1% 的特征值安全裕度也通过了 75 个构造谱算例。这些测试属于数值证据，并不构成普适的条件数保证。

实现细节、完整基准结果与数值限制，见 [`rust` 分支上的 `README_FAST.md`](https://github.com/zhouhaoyiu/SSFPG/blob/rust/README_FAST.md)。

## 参考文献与许可

Zhang, Yong. (2026). *An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions*.

本分支仍遵循本仓库的 [MIT 许可证](LICENSE)。
