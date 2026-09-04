rng(20260904);N=2000;M=1000;
g=randn(N,1);G=convmtx(g,M);G(N+1:end,:)=[];
xtrue=max(sin(linspace(0,8*pi,M)'),0);b0=G*xtrue;b=b0+0.1*max(abs(b0))*randn(size(b0));
project=@(X)max(X,0);cache.H=G'*G;cache.L=1.01*eigs(cache.H,1,'largestabs');
SSFPG(G,b,1e-99,5,'X=max(X,0);',1);SSFPG_fast(G,b,1e-99,5,project,1,cache.L);
SSFPG_gram(G,b,1e-99,5,project,1,cache);SSFPG_gram_rust(G,b,1e-99,5,1,cache);
repeats=9;original=zeros(repeats,1);fast=zeros(repeats,1);matlab=zeros(repeats,1);rust=zeros(repeats,1);
for j=1:repeats
    if mod(j,2)
        tic;xo=SSFPG(G,b,1e-99,100,'X=max(X,0);',1);original(j)=toc;
        tic;xf=SSFPG_fast(G,b,1e-99,100,project,1,cache.L);fast(j)=toc;
        tic;xm=SSFPG_gram(G,b,1e-99,100,project,1,cache);matlab(j)=toc;
        tic;xr=SSFPG_gram_rust(G,b,1e-99,100,1,cache);rust(j)=toc;
    else
        tic;xr=SSFPG_gram_rust(G,b,1e-99,100,1,cache);rust(j)=toc;
        tic;xm=SSFPG_gram(G,b,1e-99,100,project,1,cache);matlab(j)=toc;
        tic;xf=SSFPG_fast(G,b,1e-99,100,project,1,cache.L);fast(j)=toc;
        tic;xo=SSFPG(G,b,1e-99,100,'X=max(X,0);',1);original(j)=toc;
    end
end
fo=norm(G*xo-b)^2;ff=norm(G*xf-b)^2;fm=norm(G*xm-b)^2;fr=norm(G*xr-b)^2;
pg=@(x)norm(x-max(x-(G'*(G*x-b))/(cache.L/1.01),0),inf)/max(1,norm(x,inf));
fprintf('Original median=%g s\n',median(original));
fprintf('MATLAB fast median=%g s speedup=%g x\n',median(fast),median(original)/median(fast));
fprintf('MATLAB Gram median=%g s speedup=%g x\n',median(matlab),median(original)/median(matlab));
fprintf('Rust MEX median=%g s speedup_vs_Gram=%g x speedup_vs_original=%g x\n', ...
    median(rust),median(matlab)/median(rust),median(original)/median(rust));
fprintf('relative objectives original=%g fast=%g Rust-vs-Gram=%g\n', ...
    (fo-fm)/fm,(ff-fm)/fm,(fr-fm)/fm);
fprintf('Rust-vs-Gram relative solution=%g projected_gradient=%g\n',norm(xr-xm)/norm(xm),pg(xr));
