"""SSFPG_spar_mob：多组观测数据同时反演（Python 版，对应 MATLAB 的 SSFPG_spar_mob.m）。

解 [G; D] * X = ob，ob 可以是多列矩阵（每列一组独立观测），借助矩阵-矩阵乘法
一次算完整批数据。步长在整批数据上共用，残差上升的数据组用固定的 1.5/L 收缩步长重试。
特征值准备（L）与步长表由调用方给出。
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


def solve(G, D, ob, rawsteps, L, iterations=100, scaling=True, history=True):
    G = np.asarray(G, dtype=np.float64, order='F')
    n, m = G.shape
    D = sparse.csc_matrix((0, m)) if D is None else sparse.csc_matrix(D)
    B = np.asarray(ob, dtype=np.float64).reshape(n+D.shape[0], -1, order='F')
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
    raw = raw[raw != 0]                      # 多观测版剔除步长表里的 0
    steps = np.resize(raw, iterations)[:, None] / eig[None, :]
    x = np.zeros((m,k), order='F')
    hist = np.empty((m,k,iterations), order='F') if history else None
    mis = np.empty((iterations,k))
    rev = np.zeros((iterations,k), dtype=np.int32)
    best = np.full(k, np.inf)
    bestx = x.copy()

    def forward(v, ww):
        return np.vstack((G@v, (D@v)*ww))

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
    grad = G.T@ex[:n] + (D.T@ex[n:])*w
    for i in range(iterations):
        old = x
        x, ex, error = candidate(x+grad*steps[i],B,w)
        mis[i] = error/energy
        if i:
            mask = (mis[i]-mis[i-1]) > np.finfo(float).eps
            mask &= steps[i] > 2/eig
            idx = np.flatnonzero(mask)
            if len(idx):
                # 残差上升的数据组：改用固定的 1.5/L 收缩步长重试
                t = 1.5/eig[idx]
                xx, ee, errors = candidate(old[:,idx]+grad[:,idx]*t,B[:,idx],w[idx])
                x[:,idx], ex[:,idx] = xx, ee
                mis[i,idx] = errors/energy[idx]
                rev[i,idx] = 1
        grad = G.T@ex[:n] + (D.T@ex[n:])*w
        if history:
            hist[:,:,i] = x
        improved = mis[i] < best
        best[improved] = mis[i,improved]
        bestx[:,improved] = x[:,improved]
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
