function test_SSFPG_gram_rust
rng(20260904);
G=randn(120,60);b=randn(120,1);project=@(X)max(X,0);
[xm,mm,~,rm,am,dm,cache]=SSFPG_gram(G,b,1e-12,200,project,1);
[xr,mr,~,rr,ar,dr]=SSFPG_gram_rust(G,b,1e-12,200,1,cache);
assert(abs(norm(G*xr-b)^2-norm(G*xm-b)^2)/norm(G*xm-b)^2<1e-11)
assert(norm(xr-xm)/max(norm(xm),eps)<1e-9)
assert(isequal(size(am),size(ar)) && numel(mm)==numel(mr) && numel(rm)==numel(rr) && numel(dm)==numel(dr))
[xm0,~]=SSFPG_gram(G,b,0,20,project,0,cache);
[xr0,~]=SSFPG_gram_rust(G,b,0,20,0,cache);
assert(norm(xr0-xm0)/max(norm(xm0),eps)<1e-11)
[xz,mz]=SSFPG_gram_rust(eye(3),zeros(3,1),0,2,1,struct('H',eye(3),'L',1.01));
assert(all(xz==0) && all(isfinite(mz)))
badCache=cache;badCache.L=1+1i;
try
    SSFPG_gram_rust(G,b,0,2,1,badCache);
    error('test_SSFPG_gram_rust:complexAccepted','Complex L was accepted.')
catch ME
    assert(strcmp(ME.identifier,'SSFPG_gram_rust:cache'))
end
try
    SSFPG_gram_rust(G,b,0,2,[0 1],cache);
    error('test_SSFPG_gram_rust:scalingAccepted','Nonscalar isscaling was accepted.')
catch ME
    assert(strcmp(ME.identifier,'SSFPG_gram_rust:scaling'))
end
try
    [~,~,~,~,~,~]=SSFPG_gram_rust_mex(cache.H,G'*b,G,b,ones(2,1),[1 2],1,0,1,ones(60,1));
    error('test_SSFPG_gram_rust:vectorScalarAccepted','A nonscalar MEX control input was accepted.')
catch ME
    assert(strcmp(ME.identifier,'SSFPG_gram_rust:scalars'))
end

rng(7);m=160;n=80;[U,~]=qr(randn(m,n),0);[V,~]=qr(randn(n,n),0);xtrue=0.1+abs(randn(n,1));
for kappa=[10 1e3 1e6 1e8]
    A=U*diag(logspace(0,-log10(kappa),n))*V';y=A*xtrue;
    conditionCache.H=A'*A;conditionCache.L=1.01*eigs(conditionCache.H,1,'largestabs');
    xmatlab=SSFPG_gram(A,y,1e-99,1000,project,1,conditionCache);
    xrust=SSFPG_gram_rust(A,y,1e-99,1000,1,conditionCache);
    rmatlab=norm(A*xmatlab-y)^2;rrust=norm(A*xrust-y)^2;
    assert(norm(xrust-xmatlab)/max(norm(xmatlab),eps)<1e-7)
    assert(abs(rrust-rmatlab)/max(rmatlab,eps)<1e-7)
end
fprintf('test_SSFPG_gram_rust passed\n');
end
