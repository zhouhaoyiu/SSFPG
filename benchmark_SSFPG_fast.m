rng(20260904);
N=2000;
M=1000;
g=randn(N,1);
G=convmtx(g,M);
G(N+1:end,:)=[];
xtrue=max(sin(linspace(0,8*pi,M)'),0);
b0=G*xtrue;
b=b0+0.1*max(abs(b0))*randn(size(b0));

[~,~,~,~,~,~,L]=SSFPG_fast(G,b,1e-99,5,@(X)max(X,0),1);
SSFPG(G,b,1e-99,5,'X=max(X,0);',1);

repeats=7;
original=zeros(repeats,1);
fast=zeros(repeats,1);
cached=zeros(repeats,1);
original_tol=zeros(repeats,1);
fast_tol=zeros(repeats,1);
cached_tol=zeros(repeats,1);
for j=1:repeats
    rng(j);tic;xo=SSFPG(G,b,1e-99,100,'X=max(X,0);',1);original(j)=toc;
    rng(j);tic;xf=SSFPG_fast(G,b,1e-99,100,@(X)max(X,0),1);fast(j)=toc;
    tic;xfc=SSFPG_fast(G,b,1e-99,100,@(X)max(X,0),1,L);cached(j)=toc;
    rng(j);tic;[xot,mot]=SSFPG(G,b,1e-8,1000,'X=max(X,0);',1);original_tol(j)=toc;
    rng(j);tic;[xft,mft]=SSFPG_fast(G,b,1e-8,1000,@(X)max(X,0),1);fast_tol(j)=toc;
    tic;[xfct,mfct]=SSFPG_fast(G,b,1e-8,1000,@(X)max(X,0),1,L);cached_tol(j)=toc;
end

Ltrue=eigs(G'*G,1,'largestabs');
ro=norm(G*xo-b)^2;
rf=norm(G*xfc-b)^2;
pg=@(x)norm(x-max(x-(G'*(G*x-b))/Ltrue,0),inf)/max(1,norm(x,inf));

fprintf('MATLAB %s, N=%d, M=%d, repeats=%d\n',version,N,M,repeats);
fprintf('original median=%g s\n',median(original));
fprintf('fast full median=%g s, speedup=%g x\n',median(fast),median(original)/median(fast));
fprintf('fast cached-L median=%g s, speedup=%g x\n',median(cached),median(original)/median(cached));
fprintf('tol=1e-8 original median=%g s, iterations=%d\n',median(original_tol),numel(mot));
fprintf('tol=1e-8 fast full median=%g s, speedup=%g x, iterations=%d\n',median(fast_tol),median(original_tol)/median(fast_tol),numel(mft));
fprintf('tol=1e-8 fast cached-L median=%g s, speedup=%g x, iterations=%d\n',median(cached_tol),median(original_tol)/median(cached_tol),numel(mfct));
fprintf('relative objective gap=%g, original PG=%g, fast PG=%g, relative x difference=%g\n', ...
    (rf-ro)/ro,pg(xo),pg(xfc),norm(xfc-xo)/norm(xo));
fprintf('tol=1e-8 original-fast relative x difference=%g, cached-fast relative x difference=%g\n', ...
    norm(xft-xot)/norm(xot),norm(xfct-xot)/norm(xot));
