# SSFPG

An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions

一种面向大规模非负震源反演的快速收敛投影梯度算法

This fork preserves the original SSFPG solvers and adds three experimental acceleration paths on the `rust` branch. The upstream project is [ZhangYongGroupPKU/SSFPG](https://github.com/ZhangYongGroupPKU/SSFPG).

## Included solvers

The original upstream programs remain unchanged:

- `SSFPG.m`: nonnegative or bound-constrained linear least squares.
- `SSFPG_spar.m`: dense and sparse coefficient matrices.
- `SSFPG_spar_mob.m`: simultaneous inversion of multiple datasets.
- `SSFPG_spar_mweits.m`: simultaneous solutions for multiple sparse-constraint weights.
- `examples.m`: numerical deconvolution examples for the original solvers.

The fork adds:

| Solver | Intended use | Main acceleration | Main limitation |
| --- | --- | --- | --- |
| `SSFPG_fast.m` | General replacement for `SSFPG.m` | Single safeguarded fallback, reusable largest eigenvalue, warm start, optional history allocation | Speedup is moderate when `G` changes every solve |
| `SSFPG_gram.m` | Repeated inversions with the same `G` | Reuses `H = G'*G` and its largest eigenvalue | Normal equations can lose accuracy for severely ill-conditioned matrices |
| `SSFPG_gram_rust.m` | Repeated dense nonnegative inversions | Runs the cached Gram iteration loop in an optimized Rust MEX kernel | Full real `double` matrices and nonnegative projection only |

## MATLAB quick start

```matlab
project = @(x) max(x,0);

% Direct-gradient implementation; reuse L when G is unchanged.
[x1,misfit1,~,~,~,~,L] = SSFPG_fast(G,b1,1e-8,1000,project,1);
[x2,misfit2] = SSFPG_fast(G,b2,1e-8,1000,project,1,L,x1);

% Gram implementation; reuse H and L when G is unchanged.
[x3,misfit3,~,~,~,~,cache] = SSFPG_gram(G,b1,1e-8,1000,project,1);
[x4,misfit4] = SSFPG_gram(G,b2,1e-8,1000,project,1,cache,x3);
```

`SSFPG_fast` and `SSFPG_gram` also retain the original projection-string interface.

## Rust MEX quick start

Install Rust, configure a MATLAB-supported C compiler with `mex -setup C`, then run:

```matlab
build_SSFPG_gram_rust

[x1,misfit1,~,~,~,~,cache] = SSFPG_gram_rust(G,b1,1e-8,1000,1);
[x2,misfit2] = SSFPG_gram_rust(G,b2,1e-8,1000,1,cache,x1);
```

The build uses `rustc` directly and has no third-party Rust package dependencies. It creates a platform-specific MEX binary locally; binaries are not committed.

## Verified synthetic benchmark

Apple M4 Max, MATLAB R2024a, dense convolution matrix with `N=2000`, `M=1000`, fixed 100 iterations:

| Implementation | Speed relative to original `SSFPG` |
| --- | ---: |
| `SSFPG_fast`, including eigenvalue estimation | 1.72x |
| `SSFPG_fast`, cached `L` | 2.05x |
| `SSFPG_gram`, cached `H` and `L` | 4.23x |
| `SSFPG_gram_rust`, cached `H` and `L` | 13.11x |

In the Rust comparison, the MEX kernel took `0.0177 s` and was 2.60x faster than MATLAB `SSFPG_gram`. Its objective matched MATLAB `SSFPG_gram` at printed precision; the relative solution difference was `2.6e-8`, and the projected-gradient residual was `1.0e-9`.

These measurements are synthetic and machine-specific. They do not establish a speedup for a rupture-inversion project. Run the supplied benchmarks on the target matrices and compare convergence, waveform fit, moment, slip distribution, and stopping tolerance.

## Validation

```matlab
test_SSFPG_fast
test_SSFPG_gram
test_SSFPG_gram_rust

benchmark_SSFPG_fast
benchmark_SSFPG_gram
benchmark_SSFPG_gram_rust
```

The MATLAB and Rust Gram solvers matched residuals in constructed tests with condition numbers from `10` through `1e8`. The 1% eigenvalue safety margin also passed 75 constructed spectral tests. These tests are numerical evidence, not a universal conditioning guarantee.

See [README_FAST.md](README_FAST.md) for implementation details, complete benchmark results, and numerical limits.

## Reference and license

Zhang, Yong. (2026). *An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions*.

The fork remains under the repository's [MIT License](LICENSE).
