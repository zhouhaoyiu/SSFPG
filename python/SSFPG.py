"""SSFPG：非负约束线性最小二乘的投影梯度求解（Python 版，对应 MATLAB 的 SSFPG.m）。

四个核里的基础版：系数矩阵只有稠密块 G，单组观测。
特征值准备（L）与步长表由调用方给出，与迭代过程分离，保留原始算法的
非负投影、缩放、步长序列与回退行为。不接受 MATLAB 的 eval 字符串，
零能量输入会被拒绝。
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh_tridiagonal


@dataclass
class Result:
    x: np.ndarray
    misfit: np.ndarray
    steps: np.ndarray
    revisions: np.ndarray
    history: np.ndarray | None


def solve(G, ob, rawsteps, L, iterations=100, scaling=True, history=True, xtol=1e-99):
    G = np.asarray(G, dtype=np.float64, order='F')
    n, m = G.shape
    B = np.asarray(ob, dtype=np.float64).reshape(n, -1, order='F')
    k = B.shape[1]
    w = np.ones(k)
    B = np.broadcast_to(B, (B.shape[0], k))
    energy = np.sum(B*B, axis=0)
    if np.any(energy == 0):
        raise ValueError('Zero observation energy has no normalized misfit')
    eig = np.broadcast_to(np.asarray(L).ravel(), (k,))
    if np.any(eig <= 0) or not np.all(np.isfinite(eig)):
        raise ValueError('L must be positive and finite')
    raw = np.asarray(rawsteps).ravel()
    steps = np.resize(raw, iterations)[:, None] / eig[None, :]
    x = np.ones((m,k), order='F')            # 基础版初值取全 1
    hist = np.empty((m,k,iterations), order='F') if history else None
    mis = np.empty((iterations,k))
    rev = np.zeros((iterations,k), dtype=np.int32)
    best = np.full(k, np.inf)
    bestx = x.copy()

    def forward(v, ww):
        return G@v

    def candidate(v, bb, ww):
        np.maximum(v, 0, out=v)
        syn = forward(v, ww)
        if scaling:
            den = np.sum(syn*syn, axis=0)
            if np.any(den == 0):
                raise ValueError('Zero synthetic energy; original scaling is undefined')
            fac = np.sum(bb*syn, axis=0)/den
            syn *= fac
            v *= fac
        ex = bb-syn
        return v, ex, np.sum(ex*ex, axis=0)

    ex = B-forward(x,w)
    grad = G.T@ex
    for i in range(iterations):
        old = x
        x, ex, error = candidate(x+grad*steps[i],B,w)
        mis[i] = error/energy
        if i:
            mask = (mis[i]-mis[i-1]) > np.finfo(float).eps
            mask &= steps[i] > 2/eig
            idx = np.flatnonzero(mask)
            if len(idx):
                trials = np.logspace(np.log10(1/eig[0]),np.log10(steps[i,0]),40)
                xx, ee, errors = candidate(old+grad*trials,B,w)
                j = int(np.argmin(errors))
                x, ex = xx[:,j:j+1], ee[:,j:j+1]
                mis[i,0] = errors[j]/energy[0]
                steps[i,0] = trials[j]
                rev[i,idx] = 1
        grad = G.T@ex
        if history:
            hist[:,:,i] = x
        improved = mis[i] < best
        best[improved] = mis[i,improved]
        bestx[:,improved] = x[:,improved]
        if np.linalg.norm(x-old)/np.linalg.norm(x) < xtol:
            mis, steps, rev = mis[:i+1], steps[:i+1], rev[:i+1]
            if history:
                hist=hist[:,:,:i+1]
            break
    return Result(bestx,mis,steps,rev,hist)


def estimate_l(G,D=None,weights=None,seed=0):
    """SciPy preparation; same estimator families, not identical random starts."""
    n,m = G.shape
    D = sparse.csc_matrix((0,m)) if D is None else D
    rng=np.random.default_rng(seed)
    if weights is None:
        def mv(x):
            return G.T@(G@x)+D.T@(D@x)
        x=rng.normal(size=m); x/=np.linalg.norm(x)
        for _ in range(2):
            x=mv(x); x/=np.linalg.norm(x)
        rough=np.linalg.norm(G@x)**2+np.linalg.norm(D@x)**2
        return eigsh(LinearOperator((m,m),matvec=mv),k=1,which='LM',
                     tol=rough*1e-8,maxiter=100,v0=rng.normal(size=m),
                     return_eigenvectors=False)
    w=np.asarray(weights).ravel(); k=len(w)
    q=rng.normal(size=(m,k)); q/=np.linalg.norm(q,axis=0)
    old=np.zeros_like(q); beta_old=np.zeros(k)
    aa=np.zeros((20,k)); bb=np.zeros((20,k))
    L=np.zeros(k); residual=np.zeros(k)
    for j in range(20):
        z=G.T@(G@q)+D.T@((D@q)*(w*w))
        if j:
            z-=old*beta_old
        a=np.sum(q*z,axis=0); z-=q*a
        if j:
            z-=old*np.sum(old*z,axis=0)
        b=np.linalg.norm(z,axis=0); aa[j]=a; bb[j]=b
        for c in range(k):
            vals,vec=eigh_tridiagonal(aa[:j+1,c],bb[:j,c])
            L[c]=vals[-1]
            residual[c]=b[c]*abs(vec[-1,-1])/max(abs(L[c]),np.finfo(float).eps)
        if (j>=2 and np.all(residual<1e-10)) or max(b)<100*np.finfo(float).eps:
            break
    return L
