function test_SSFPG_fast
rng(7);
G=randn(80,40);
xtrue=max(randn(40,1),0);
b=G*xtrue+0.01*randn(80,1);

[x,m,~,~,~,~,L]=SSFPG_fast(G,b,1e-10,300,@(X)max(X,0),1);
[xref,rref]=lsqnonneg(G,b);
r=norm(G*x-b)^2;
assert(all(isfinite(x)) && all(isfinite(m)) && all(x>=0));
assert((r-rref)/max(rref,eps)<1e-8);

xcached=SSFPG_fast(G,b,1e-10,300,'X=max(X,0);',1,L);
assert(norm(x-xcached)/max(norm(x),eps)<1e-8);
assert(norm(x-xref)/max(norm(xref),eps)<1e-5);

xneg=SSFPG_fast(eye(3),-ones(3,1),1e-12,10,'X=max(X,0);',1);
xzero=SSFPG_fast(eye(3),zeros(3,1),1e-12,10,'X=max(X,0);',1);
assert(all(isfinite(xneg)) && norm(xneg)==0);
assert(all(isfinite(xzero)) && norm(xzero)==0);
fprintf('test_SSFPG_fast passed\n');
end
