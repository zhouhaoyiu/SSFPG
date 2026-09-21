//! SSFPG 的 Rust 移植：四个核共用的底座。
//!
//! 四个求解器各占一个文件，本文件只放它们共用的部分：
//! - macOS Accelerate 的 BLAS 绑定（`cblas_dgemm` / `cblas_dgemv`）
//! - 正演、梯度、候选解算子 `Op`
//! - C ABI 入口的公共辅助 `finish`（错误码映射与 panic 兜底）
//!
//! 所有矩阵按列主序 `f64` 存放。特征值准备（`L`）与步长表由调用方
//! （Python / MATLAB）给出，本 crate 只负责迭代循环。
//!
//! | 文件 | 对应 MATLAB | 导出的 C 符号 |
//! | --- | --- | --- |
//! | `src/ssfpg.rs` | `SSFPG.m` | `ssfpg_run_dense` |
//! | `src/ssfpg_spar.rs` | `SSFPG_spar.m` | `ssfpg_run_spar` |
//! | `src/ssfpg_spar_mob.rs` | `SSFPG_spar_mob.m` | `ssfpg_run_mob` |
//! | `src/ssfpg_spar_mweits.rs` | `SSFPG_spar_mweits.m` | `ssfpg_run_mweits` |

pub mod ssfpg;
pub mod ssfpg_spar;
pub mod ssfpg_spar_mob;
pub mod ssfpg_spar_mweits;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_dgemm(
        order: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
    fn cblas_dgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );
}

/// 候选解的返回类型：`(投影并缩放后的解, 残差, 各列误差能量)`。
pub(crate) type Candidate = Result<(Vec<f64>, Vec<f64>, Vec<f64>), i32>;

/// 系数矩阵与观测算子的统一表示：稠密块 `g` 加 CSC 稀疏块 `dv/dr/dp`。
pub(crate) struct Op<'a> {
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) p: usize,
    pub(crate) g: &'a [f64],
    pub(crate) dv: &'a [f64],
    pub(crate) dr: &'a [i64],
    pub(crate) dp: &'a [i64],
}

impl Op<'_> {
    /// 正演：稠密块走 Accelerate 的 dgemv/dgemm，稀疏块手工累加。
    pub(crate) fn forward(&self, x: &[f64], w: &[f64]) -> Vec<f64> {
        let k = w.len();
        let h = self.n + self.p;
        let mut y = vec![0.; h * k];
        unsafe {
            if k == 1 {
                cblas_dgemv(
                    102,
                    111,
                    self.n as i32,
                    self.m as i32,
                    1.,
                    self.g.as_ptr(),
                    self.n as i32,
                    x.as_ptr(),
                    1,
                    0.,
                    y.as_mut_ptr(),
                    1,
                );
            } else {
                cblas_dgemm(
                    102,
                    111,
                    111,
                    self.n as i32,
                    k as i32,
                    self.m as i32,
                    1.,
                    self.g.as_ptr(),
                    self.n as i32,
                    x.as_ptr(),
                    self.m as i32,
                    0.,
                    y.as_mut_ptr(),
                    h as i32,
                );
            }
        }
        for c in 0..k {
            for j in 0..self.m {
                let z = x[c * self.m + j] * w[c];
                for a in self.dp[j] as usize..self.dp[j + 1] as usize {
                    y[c * h + self.n + self.dr[a] as usize] += self.dv[a] * z;
                }
            }
        }
        y
    }

    /// 梯度：残差先经稠密块转置乘，再叠加稀疏块的贡献，最后按列乘权重。
    pub(crate) fn gradient(&self, e: &[f64], w: &[f64]) -> Vec<f64> {
        let k = w.len();
        let h = self.n + self.p;
        let mut x = vec![0.; self.m * k];
        unsafe {
            if k == 1 {
                cblas_dgemv(
                    102,
                    112,
                    self.n as i32,
                    self.m as i32,
                    1.,
                    self.g.as_ptr(),
                    self.n as i32,
                    e.as_ptr(),
                    1,
                    0.,
                    x.as_mut_ptr(),
                    1,
                );
            } else {
                cblas_dgemm(
                    102,
                    112,
                    111,
                    self.m as i32,
                    k as i32,
                    self.n as i32,
                    1.,
                    self.g.as_ptr(),
                    self.n as i32,
                    e.as_ptr(),
                    h as i32,
                    0.,
                    x.as_mut_ptr(),
                    self.m as i32,
                );
            }
        }
        for c in 0..k {
            for j in 0..self.m {
                let mut v = 0.;
                for a in self.dp[j] as usize..self.dp[j + 1] as usize {
                    v += self.dv[a] * e[c * h + self.n + self.dr[a] as usize];
                }
                x[c * self.m + j] += v * w[c];
            }
        }
        x
    }

    /// 候选解：先做非负投影，再（可选）整体缩放到最佳拟合，最后返回残差与误差能量。
    pub(crate) fn candidate(
        &self,
        mut x: Vec<f64>,
        b: &[f64],
        w: &[f64],
        scaling: bool,
    ) -> Candidate {
        for v in &mut x {
            *v = v.max(0.);
        }
        let h = self.n + self.p;
        let mut syn = self.forward(&x, w);
        let mut errors = vec![0.; w.len()];
        for c in 0..w.len() {
            if scaling {
                let mut den = 0.;
                let mut num = 0.;
                for r in 0..h {
                    let t = c * h + r;
                    den += syn[t] * syn[t];
                    num += b[t] * syn[t];
                }
                if den == 0. {
                    return Err(2);
                }
                let fac = num / den;
                for r in 0..h {
                    syn[c * h + r] *= fac;
                }
                for r in 0..self.m {
                    x[c * self.m + r] *= fac;
                }
            }
            for r in 0..h {
                let t = c * h + r;
                syn[t] = b[t] - syn[t];
                errors[c] += syn[t] * syn[t];
            }
        }
        Ok((x, syn, errors))
    }
}

/// 按列下标从列主序数组里挑出若干列（多组观测 / 多权重时只处理需要重试的那些列）。
pub(crate) fn select(a: &[f64], rows: usize, cols: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * cols.len());
    for &c in cols {
        out.extend_from_slice(&a[c * rows..(c + 1) * rows]);
    }
    out
}

/// C ABI 入口的统一收尾：把内部错误码映射为返回码，并兜住 panic。
///
/// 返回 0 成功，1 表示能量或 L 非法，2 表示缩放时合成能量为 0，3 表示发生 panic。
pub(crate) fn finish<F>(f: F) -> i32
where
    F: FnOnce() -> Result<(), i32> + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(f) {
        Ok(Ok(())) => 0,
        Ok(Err(code)) => code,
        Err(_) => 3,
    }
}
