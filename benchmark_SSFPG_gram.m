rng(20260904);
N=2000;M=1000;
g=randn(N,1);G=convmtx(g,M);G(N+1:end,:)=[];
xtrue=max(sin(linspace(0,8*pi,M)'),0);
b0=G*xtrue;b=b0+0.1*max(abs(b0))*randn(size(b0));
project=@(X)max(X,0);

tic;cache.H=G'*G;cache.L=1.01*eigs(cache.H,1,'largestabs');prepare=toc;
SSFPG_gram(G,b,1e-99,5,project,1,cache);
SSFPG_fast(G,b,1e-99,5,project,1,cache.L);
SSFPG(G,b,1e-99,5,'X=max(X,0);',1);

repeats=9;
fast=zeros(repeats,1);
gram=zeros(repeats,1);
full=zeros(repeats,1);
original=zeros(repeats,1);
for j=1:repeats
    rng(j);tic;xo=SSFPG(G,b,1e-99,100,'X=max(X,0);',1);original(j)=toc;
    tic;xf=SSFPG_fast(G,b,1e-99,100,project,1,cache.L);fast(j)=toc;
    tic;xg=SSFPG_gram(G,b,1e-99,100,project,1,cache);gram(j)=toc;
    tic;xgf=SSFPG_gram(G,b,1e-99,100,project,1);full(j)=toc;
end

Ltrue=eigs(G'*G,1,'largestabs');
rf=norm(G*xf-b)^2;
rg=norm(G*xg-b)^2;
pg=@(x)norm(x-max(x-(G'*(G*x-b))/Ltrue,0),inf)/max(1,norm(x,inf));
fprintf('prepare=%g s cache=%g MB\n',prepare,numel(cache.H)*8/1e6);
fprintf('original median=%g s\n',median(original));
fprintf('fast median=%g s speedup_vs_original=%g x\n',median(fast),median(original)/median(fast));
fprintf('gram cached median=%g s speedup_vs_fast=%g x speedup_vs_original=%g x\n', ...
    median(gram),median(fast)/median(gram),median(original)/median(gram));
fprintf('gram full median=%g s speedup_vs_original=%g x\n',median(full),median(original)/median(full));
fprintf('gram relgap=%g pg=%g relx=%g\n',(rg-rf)/rf,pg(xg),norm(xg-xf)/norm(xf));
fprintf('times fast=');fprintf(' %g',fast);fprintf('\n');
fprintf('times gram=');fprintf(' %g',gram);fprintf('\n');
fprintf('times full=');fprintf(' %g',full);fprintf('\n');
fprintf('times original=');fprintf(' %g',original);fprintf('\n');
