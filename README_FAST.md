# SSFPG_fast

`SSFPG_fast.m` is an experimental, interface-compatible acceleration of Zhang Yong Group's MIT-licensed SSFPG implementation.

## Changes

- Replaces the 40-candidate safeguard with one stable fallback step, `1.5/L`. The official multi-observation solver already uses this fallback.
- Estimates `L = lambda_max(G'*G)` with a shorter `eigs` run, adds a 1% safety margin, and falls back to a strict estimate if `eigs` does not converge.
- Accepts a cached `L` as the seventh input for repeated inversions with the same `G`.
- Accepts an optional warm-start solution as the eighth input.
- Allocates `Xall` only when the caller requests the fifth output.
- Tracks the best solution without retaining every iterate.
- Handles zero predictions, negative scaling factors, zero data, and zero-solution stopping without NaN values.
- Accepts a projection function handle while retaining the original code-string interface.

## Usage

```matlab
project = @(X) max(X,0);

% First inversion; return the estimated L for reuse.
[x1,misfit1,~,~,~,~,L] = SSFPG_fast(G,b1,1e-8,1000,project,1);

% Later inversions with the same G skip eigenvalue estimation.
[x2,misfit2] = SSFPG_fast(G,b2,1e-8,1000,project,1,L);

% Optional warm start.
[x3,misfit3] = SSFPG_fast(G,b3,1e-8,1000,project,1,L,x2);
```

The original call remains valid:

```matlab
[x,misfit] = SSFPG_fast(G,b,1e-99,100,'X=max(X,0);',1);
```

## Verified results

MATLAB R2024a, dense convolution problem with `N=2000`, `M=1000`, 100 iterations, seven timing repetitions:

- Full call including eigenvalue estimation: 1.72 times faster than the original SSFPG in the packaged benchmark run.
- Reusing `L`: 2.05 times faster for fixed 100 iterations.
- Relative objective difference from the original: about `1e-15`.
- Relative solution difference from the original: about `4e-9`.

With `Xtol0=1e-8`, the original stopped after 58 iterations and the fast version after 64. The fast version was 1.68 times faster including eigenvalue estimation and 2.30 times faster with cached `L`; its solution differed from the original by about `1.7e-7`.

## Repeated-inversion backend

`SSFPG_gram.m` caches `H = G'*G` and its largest eigenvalue. It is intended for repeated inversions with the same `G`:

```matlab
[x1,misfit1,~,~,~,~,cache] = SSFPG_gram(G,b1,1e-8,1000,project,1);
[x2,misfit2] = SSFPG_gram(G,b2,1e-8,1000,project,1,cache,x1);
```

On the same `N=2000`, `M=1000` benchmark:

- Cached `SSFPG_gram`: 4.23 times faster than the original and 2.02 times faster than cached-`L` `SSFPG_fast`.
- Relative objective difference from `SSFPG_fast`: `7.5e-16` in magnitude.
- Relative solution difference from `SSFPG_fast`: `1.4e-12`.
- Cache preparation: about `0.20 s`; cache memory: `8 MB`.

The cache preparation cost broke even after about two solves relative to the original, or about five solves relative to cached-`L` `SSFPG_fast`, on this matrix.

## Rust MEX backend

`SSFPG_gram_rust.m` keeps cache construction and data preparation in MATLAB and runs the dense nonnegative iteration loop in Rust. It has no Rust package dependencies. Build it locally with `rustc` and MATLAB's configured C MEX compiler:

```matlab
build_SSFPG_gram_rust
[x1,misfit1,~,~,~,~,cache] = SSFPG_gram_rust(G,b1,1e-8,1000,1);
[x2,misfit2] = SSFPG_gram_rust(G,b2,1e-8,1000,1,cache,x1);
```

On Apple M4 Max with MATLAB R2024a, the same `N=2000`, `M=1000`, 100-iteration interleaved benchmark gave:

- Rust MEX: `0.0177 s`, 2.60 times faster than MATLAB `SSFPG_gram` and 13.11 times faster than the original SSFPG.
- Relative objective difference from MATLAB `SSFPG_gram`: zero at printed precision.
- Relative solution difference: `2.6e-8`; projected-gradient residual: `1.0e-9`.

The Rust and MATLAB Gram solvers also matched residuals in constructed condition-number tests from `10` through `1e8`. Run `benchmark_SSFPG_gram_rust` on the target machine because BLAS and compiler performance vary.

Run:

```matlab
test_SSFPG_fast
test_SSFPG_gram
test_SSFPG_gram_rust
benchmark_SSFPG_fast
benchmark_SSFPG_gram
benchmark_SSFPG_gram_rust
```

## Limits

- The 1% eigenvalue safety margin passed 75 constructed full-rank, clustered-spectrum, ill-conditioned, rank-deficient, and rescaled cases. This is numerical evidence, not a universal proof. Pass a trusted `L` for critical inversions.
- The current version accelerates `SSFPG.m`. Sparse, multi-observation, and multi-weight variants are unchanged.
- `SSFPG_gram` uses normal equations. This can lose numerical accuracy for severely ill-conditioned matrices. Use `SSFPG_fast` when preserving the original `G'*(b-G*x)` calculation is required.
- `SSFPG_gram_rust` currently supports full real double matrices and the nonnegative projection only. The repository contains source code, not a platform-specific MEX binary; run the build script locally.
- Ill-conditioned inverse problems still require suitable regularization or preconditioning. Faster iterations do not recover poorly resolved model components.
- The benchmark is synthetic. Validate runtime, waveform fit, moment, slip distribution, and stopping tolerance on the actual rupture-inversion matrices before scientific use.
