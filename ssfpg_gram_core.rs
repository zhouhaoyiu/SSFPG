use std::slice;

#[inline]
fn matvec(h: &[f64], x: &[f64], y: &mut [f64], m: usize) {
    y.fill(0.0);
    for j in 0..m {
        let xj = x[j];
        let col = &h[j * m..(j + 1) * m];
        for i in 0..m {
            y[i] = col[i].mul_add(xj, y[i]);
        }
    }
}

#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

fn true_residual(g: &[f64], ob: &[f64], x: &[f64], syn: &mut [f64], n: usize) -> f64 {
    syn.fill(0.0);
    for (j, &xj) in x.iter().enumerate() {
        let col = &g[j * n..(j + 1) * n];
        for i in 0..n {
            syn[i] = col[i].mul_add(xj, syn[i]);
        }
    }
    ob.iter().zip(syn).map(|(a, b)| (*a - *b) * (*a - *b)).sum()
}

#[no_mangle]
pub unsafe extern "C" fn ssfpg_gram_core(
    h_ptr: *const f64,
    c_ptr: *const f64,
    g_ptr: *const f64,
    ob_ptr: *const f64,
    steps_ptr: *const f64,
    m: usize,
    n: usize,
    iter: usize,
    tmax: f64,
    ob_e: f64,
    xtol: f64,
    scaling: i32,
    xinit_ptr: *const f64,
    xout_ptr: *mut f64,
    misfit_ptr: *mut f64,
    used_steps_ptr: *mut f64,
    revised_ptr: *mut f64,
    xdif_ptr: *mut f64,
    xall_ptr: *mut f64,
) -> usize {
    let h = slice::from_raw_parts(h_ptr, m * m);
    let c = slice::from_raw_parts(c_ptr, m);
    let g = slice::from_raw_parts(g_ptr, n * m);
    let ob = slice::from_raw_parts(ob_ptr, n);
    let steps = slice::from_raw_parts(steps_ptr, iter);
    let xinit = slice::from_raw_parts(xinit_ptr, m);
    let xout = slice::from_raw_parts_mut(xout_ptr, m);
    let misfit = slice::from_raw_parts_mut(misfit_ptr, iter);
    let used_steps = slice::from_raw_parts_mut(used_steps_ptr, iter);
    let revised = slice::from_raw_parts_mut(revised_ptr, iter);
    let xdif = slice::from_raw_parts_mut(xdif_ptr, iter);
    let mut xall = if xall_ptr.is_null() {
        None
    } else {
        Some(slice::from_raw_parts_mut(xall_ptr, m * iter))
    };

    used_steps.copy_from_slice(steps);
    let mut x = xinit.to_vec();
    let mut x0 = vec![0.0; m];
    let mut hx = vec![0.0; m];
    let mut gradient = vec![0.0; m];
    let mut syn = vec![0.0; n];
    matvec(h, &x, &mut hx, m);
    for j in 0..m {
        gradient[j] = c[j] - hx[j];
    }

    let mut best = f64::INFINITY;
    for k in 0..iter {
        x0.copy_from_slice(&x);
        let mut step = used_steps[k];
        for j in 0..m {
            x[j] = (x[j] + step * gradient[j]).max(0.0);
        }

        let mut evaluate = |x: &mut [f64], hx: &mut [f64]| {
            matvec(h, x, hx, m);
            let mut p = dot(x, c);
            let mut q = dot(x, hx);
            if scaling != 0 {
                if q > 0.0 {
                    let factor = (p / q).max(0.0);
                    for j in 0..m {
                        x[j] *= factor;
                        hx[j] *= factor;
                    }
                    p *= factor;
                    q *= factor * factor;
                } else {
                    x.fill(0.0);
                    hx.fill(0.0);
                    p = 0.0;
                    q = 0.0;
                }
            }
            let mut value = (ob_e - 2.0 * p + q).max(0.0) / ob_e;
            if value < f64::EPSILON.sqrt() {
                value = true_residual(g, ob, x, &mut syn, n) / ob_e;
            }
            value
        };

        let mut value = evaluate(&mut x, &mut hx);
        if step > tmax && k > 0 && value - misfit[k - 1] > f64::EPSILON {
            step = 1.5 * tmax / 2.0;
            used_steps[k] = step;
            for j in 0..m {
                x[j] = (x0[j] + step * gradient[j]).max(0.0);
            }
            value = evaluate(&mut x, &mut hx);
            revised[k] = 1.0;
        }
        misfit[k] = value;

        if value < best {
            best = value;
            xout.copy_from_slice(&x);
        }
        for j in 0..m {
            gradient[j] = c[j] - hx[j];
        }
        if let Some(xall) = xall.as_deref_mut() {
            xall[k * m..(k + 1) * m].copy_from_slice(&x);
        }

        let mut delta2 = 0.0;
        let mut norm2 = 0.0;
        for j in 0..m {
            delta2 += (x[j] - x0[j]) * (x[j] - x0[j]);
            norm2 += x[j] * x[j];
        }
        xdif[k] = delta2.sqrt() / norm2.sqrt().max(f64::EPSILON);
        if xdif[k] < xtol {
            return k + 1;
        }
    }
    iter
}
