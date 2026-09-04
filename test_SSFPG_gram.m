function test_SSFPG_gram
rng(9);
G=randn(100,50);
project=@(X)max(X,0);

xtrue=max(randn(50,1),0);
b=G*xtrue+0.01*randn(100,1);
[x,m,~,~,~,~,cache]=SSFPG_gram(G,b,1e-10,300,project,1);
[xref,rref]=lsqnonneg(G,b);
r=norm(G*x-b)^2;
assert(all(isfinite(x)) && all(isfinite(m)) && all(x>=0));
assert((r-rref)/max(rref,eps)<1e-8);
assert(norm(x-xref)/max(norm(xref),eps)<1e-5);

b2=b+0.001*randn(size(b));
xcold=SSFPG_gram(G,b2,1e-10,300,project,1,cache);
xwarm=SSFPG_gram(G,b2,1e-10,300,project,1,cache,x);
assert(norm(xcold-xwarm)/max(norm(xcold),eps)<1e-7);

xneg=SSFPG_gram(eye(3),-ones(3,1),1e-12,10,project,1);
xzero=SSFPG_gram(eye(3),zeros(3,1),1e-12,10,project,1);
assert(all(isfinite(xneg)) && norm(xneg)==0);
assert(all(isfinite(xzero)) && norm(xzero)==0);
badCache=cache;badCache.L=1+1i;
try
    SSFPG_gram(G,b,1e-10,2,project,1,badCache);
    error('test_SSFPG_gram:complexAccepted','Complex L was accepted.')
catch ME
    assert(strcmp(ME.identifier,'SSFPG_gram:cache'))
end
try
    SSFPG_gram(G,b,1e-10,2,project,[0 1]);
    error('test_SSFPG_gram:scalingAccepted','Nonscalar isscaling was accepted.')
catch ME
    assert(strcmp(ME.identifier,'SSFPG_gram:scaling'))
end
fprintf('test_SSFPG_gram passed\n');
end
