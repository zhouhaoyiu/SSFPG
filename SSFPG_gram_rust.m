function [X_out,misfit,t0,t_isrevise,Xall,Xdif,cache]=SSFPG_gram_rust(G,ob,Xtol0,iter,isscaling,cache,Xinit)
% SSFPG_GRAM_RUST  Rust-MEX nonnegative SSFPG with cached H=G'*G.
if exist('SSFPG_gram_rust_mex','file')~=3
    error('SSFPG_gram_rust:notBuilt','Run build_SSFPG_gram_rust first.')
end
if nargin<5,isscaling=0;end
if ~isscalar(isscaling) || ~(isscaling==0 || isscaling==1)
    error('SSFPG_gram_rust:scaling','isscaling must be 0 or 1.')
end
if ~isa(G,'double') || ~isreal(G) || issparse(G)
    error('SSFPG_gram_rust:type','G must be a full real double matrix.')
end
if ~isa(ob,'double') || ~isreal(ob) || ~iscolumn(ob) || size(G,1)~=numel(ob)
    error('SSFPG_gram_rust:dimensions','ob must be a real double column vector with size(G,1) elements.')
end
if ~isscalar(iter) || iter<1 || iter~=fix(iter),error('SSFPG_gram_rust:iterations','iter must be a positive integer.');end
if ~isscalar(Xtol0) || ~isfinite(Xtol0) || Xtol0<0
    error('SSFPG_gram_rust:tolerance','Xtol0 must be a nonnegative finite scalar.')
end
m=size(G,2);
if nargin<6 || isempty(cache)
    cache.H=G'*G;
    cache.L=1.01*eigs(cache.H,1,'largestabs');
end
if ~isstruct(cache) || ~isfield(cache,'H') || ~isfield(cache,'L') || ...
        ~isa(cache.H,'double') || ~isreal(cache.H) || issparse(cache.H) || ...
        ~isequal(size(cache.H),[m,m]) || ~isscalar(cache.L) || ~isfinite(cache.L) || cache.L<=0
    error('SSFPG_gram_rust:cache','cache must contain a full real double H and positive L.')
end
if nargin<7 || isempty(Xinit)
    Xinit=ones(m,1);
elseif ~isa(Xinit,'double') || ~isreal(Xinit) || ~iscolumn(Xinit) || ...
        numel(Xinit)~=m || any(~isfinite(Xinit))
    error('SSFPG_gram_rust:Xinit','Xinit must be a finite real double column vector with size(G,2) elements.')
end
Xinit=max(Xinit,0);

s=load(fullfile(fileparts(mfilename('fullpath')),'t0_1e6_30.mat'));
t0=repmat(s.t0/cache.L,[ceil(iter/numel(s.t0)),1]);
t0=t0(1:iter);
tmax=2/cache.L;
obE=max(ob'*ob,eps);
c=G'*ob;
if nargout>=5
    [X_out,misfit,t0,t_isrevise,Xdif,niter,Xall]=SSFPG_gram_rust_mex( ...
        cache.H,c,G,ob,t0,tmax,obE,double(Xtol0),double(isscaling==1),Xinit);
    Xall=Xall(:,1:niter);
else
    [X_out,misfit,t0,t_isrevise,Xdif,niter]=SSFPG_gram_rust_mex( ...
        cache.H,c,G,ob,t0,tmax,obE,double(Xtol0),double(isscaling==1),Xinit);
    Xall=[];
end
misfit=misfit(1:niter);Xdif=Xdif(1:niter);
end
