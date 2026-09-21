//! SSFPG_spar_mweits：对稀疏约束块施加多重权重并一次求解（对应 MATLAB 的 `SSFPG_spar_mweits.m`）。
//!
//! 解 `[G; D*weits] * X = ob`，每一列对应一个权重，一次迭代得到全部权重的解。
//! 与其它三个核的区别：
//! - 残差上升时不做稳定性上限判断（`ts > 2/L`），一涨就修正；
//! - 默认只保留最大 `L` 那一列（对应 MATLAB dense mrdivide 的基本解），其余取 0；
//! - 修正后的步长会写回步长表。
//!
//! 导出 C 符号 `ssfpg_run_mweits`。

use std::slice;

use crate::{finish, select, Op};

#[allow(clippy::too_many_arguments)]
fn run(
    op: Op,
    b: &[f64],
    w: &[f64],
    raw: &[f64],
    l: &[f64],
    it: usize,
    scaling: bool,
    history: bool,
    out: &mut [f64],
    mis: &mut [f64],
    ts: &mut [f64],
    rev: &mut [i32],
) -> Result<(), i32> {
    let k = w.len();
    let m = op.m;
    let h = op.n + op.p;
    let energy: Vec<f64> = b.chunks(h).map(|v| v.iter().map(|a| a * a).sum()).collect();
    if energy.iter().any(|v| *v <= 0.) || l.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return Err(1);
    }
    let mut x = vec![0.; m * k];
    let syn = op.forward(&x, w);
    let mut e: Vec<f64> = b.iter().zip(syn).map(|(bb, ss)| bb - ss).collect();
    let mut grad = op.gradient(&e, w);
    let mut best = vec![f64::INFINITY; k];
    let mut hist = if history {
        vec![0.; m * k * it]
    } else {
        Vec::new()
    };
    for i in 0..it {
        let old = x;
        let mut trial = vec![0.; m * k];
        for c in 0..k {
            ts[i * k + c] = raw[i % raw.len()] / l[c];
            for r in 0..m {
                let t = c * m + r;
                trial[t] = old[t] + grad[t] * ts[i * k + c];
            }
        }
        let (xx, ee, err) = op.candidate(trial, b, w, scaling)?;
        x = xx;
        e = ee;
        for c in 0..k {
            mis[i * k + c] = err[c] / energy[c];
        }
        if i > 0 {
            // 多权重版不看稳定性上限：残差一涨就修正
            let idx: Vec<usize> = (0..k)
                .filter(|&c| mis[i * k + c] - mis[(i - 1) * k + c] > f64::EPSILON)
                .collect();
            if !idx.is_empty() {
                let wi: Vec<f64> = idx.iter().map(|&c| w[c]).collect();
                let bi = select(b, h, &idx);
                let mut trial = select(&old, m, &idx);
                // R2024a mrdivide basic solution: one pivot at the largest L.
                let pivot = *idx.iter().max_by(|&&a, &&b| l[a].total_cmp(&l[b])).unwrap();
                for (j, &c) in idx.iter().enumerate() {
                    let t = if c != pivot { 0. } else { 1.5 / l[c] };
                    ts[i * k + c] = t;
                    for r in 0..m {
                        trial[j * m + r] += grad[c * m + r] * t;
                    }
                }
                let (xx, ee, err) = op.candidate(trial, &bi, &wi, scaling)?;
                for (j, &c) in idx.iter().enumerate() {
                    x[c * m..(c + 1) * m].copy_from_slice(&xx[j * m..(j + 1) * m]);
                    e[c * h..(c + 1) * h].copy_from_slice(&ee[j * h..(j + 1) * h]);
                    mis[i * k + c] = err[j] / energy[c];
                }
                for c in idx {
                    rev[i * k + c] = 1;
                }
            }
        }
        grad = op.gradient(&e, w);
        if history {
            hist[i * m * k..(i + 1) * m * k].copy_from_slice(&x);
        }
        for c in 0..k {
            if mis[i * k + c] < best[c] {
                best[c] = mis[i * k + c];
                out[c * m..(c + 1) * m].copy_from_slice(&x[c * m..(c + 1) * m]);
            }
        }
    }
    // Observe history so optimized compilation cannot remove the benchmark storage.
    std::hint::black_box(&hist);
    Ok(())
}

/// C ABI 入口（仅由随附的调用方使用）。
///
/// 返回 0 成功，1 表示能量或 L 非法，2 表示缩放时合成能量为 0，3 表示 panic。
///
/// # Safety
///
/// 调用方必须保证：`g/dv/dr/dp/b/w/raw/l` 非空且指向足够长的连续内存
/// （长度分别为 `n*m`、`nnz`、`nnz`、`m+1`、`(n+p)*k`、`k`、`nraw`、`k` 个元素），
/// 且 `out/mis/ts/rev` 可写、元素数分别不小于 `m*k`、`it*k`、`it*k`、`it*k`。
#[no_mangle]
pub unsafe extern "C" fn ssfpg_run_mweits(
    n: usize,
    m: usize,
    p: usize,
    k: usize,
    nnz: usize,
    g: *const f64,
    dv: *const f64,
    dr: *const i64,
    dp: *const i64,
    b: *const f64,
    w: *const f64,
    raw: *const f64,
    nraw: usize,
    l: *const f64,
    it: usize,
    scaling: i32,
    history: i32,
    out: *mut f64,
    mis: *mut f64,
    ts: *mut f64,
    rev: *mut i32,
) -> i32 {
    if n == 0 || m == 0 || k == 0 || nraw == 0 || it == 0 {
        return 1;
    }
    let op = Op {
        n,
        m,
        p,
        g: slice::from_raw_parts(g, n * m),
        dv: slice::from_raw_parts(dv, nnz),
        dr: slice::from_raw_parts(dr, nnz),
        dp: slice::from_raw_parts(dp, m + 1),
    };
    finish(std::panic::AssertUnwindSafe(|| {
        run(
            op,
            slice::from_raw_parts(b, (n + p) * k),
            slice::from_raw_parts(w, k),
            slice::from_raw_parts(raw, nraw),
            slice::from_raw_parts(l, k),
            it,
            scaling != 0,
            history != 0,
            slice::from_raw_parts_mut(out, m * k),
            slice::from_raw_parts_mut(mis, it * k),
            slice::from_raw_parts_mut(ts, it * k),
            slice::from_raw_parts_mut(rev, it * k),
        )
    }))
}
