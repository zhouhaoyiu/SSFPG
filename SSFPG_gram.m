function [X_out,misfit,t0,t_isrevise,Xall,Xdif,cache]=SSFPG_gram(G,ob,Xtol0,iter,evalchar,isscaling,cache,Xinit)
%==========================================================================
% SSFPG using a cached normal matrix H=G'*G for repeated inversions.
% -------------------------------------------------------------------------
% Input
%        G: the matrix
%       ob: the data
%     Xtol: stopping tolerance of solution change, a small number. it can be a very small value (i.e. 1e-99), and then won't be used
%     iter: maximum iteration number
% evalchar: projection of X on the feasible set in each iteration, either a
%           function handle such as @(X)max(X,0), or a compatible code string
%           and/or upper bounder limitation: 'X(X>1)=1;',
%isscaling: an option, is isscaling==1, then the solution will be scaled to
%           best fit ob, note that it works only when non-negative limiation 'X=max(X,0);' is used.
%    cache: optional struct returned by an earlier call with fields H and L
%    Xinit: optional warm-start solution
% Output
%    X_out: solution with the minimum residual
%   misfit: normalized misfit
%       t0: used step sizes
% t_isrevise: if the misfit increases at the i-th iteration, then search for an optimized step size, and then t_isrevise(i)=1
%     Xall: solutions obtained at all iterations
%     Xdif: difference of the solution compared with the previous one
%     cache: reusable H=G'*G and L=lambda_max(H)
%--------------------------------------------------------------------------
% Reference:
% Zhang Yong. 2026. An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions
% Experimental derivative of ZhangYong-Group/SSFPG, distributed under the
% repository's MIT License.
%==========================================================================
if nargin<6
    isscaling=0;
end
if ~(isnumeric(isscaling) || islogical(isscaling)) || ~isscalar(isscaling) || ...
        ~isreal(isscaling) || ~isfinite(isscaling) || ~(isscaling==0 || isscaling==1)
    error('SSFPG_gram:scaling','isscaling must be 0 or 1.')
end
sg=size(G);
if ~iscolumn(ob) || size(G,1)~=numel(ob)
    error('SSFPG_gram:dimensions','ob must be a column vector with size(G,1) elements.')
end
if iter<1 || iter~=fix(iter)
    error('SSFPG_gram:iterations','iter must be a positive integer.')
end
if ~isscalar(Xtol0) || ~isfinite(Xtol0) || Xtol0<0
    error('SSFPG_gram:tolerance','Xtol0 must be a nonnegative finite scalar.')
end

if nargin<7 || isempty(cache)
    cache.H=G'*G;
    [~,cache.L,flag]=eigs(cache.H,1,'largestabs');
    if flag~=0 || ~isreal(cache.L) || ~isfinite(cache.L) || cache.L<=0
        error('SSFPG_gram:eigs','Unable to estimate a positive finite real largest eigenvalue.')
    end
    cache.L=1.01*cache.L;
end
if ~isstruct(cache) || ~isfield(cache,'H') || ~isfield(cache,'L') || ...
        ~isequal(size(cache.H),[sg(2),sg(2)]) || ~isnumeric(cache.L) || ~isscalar(cache.L) || ...
        ~isreal(cache.L) || ~isfinite(cache.L) || cache.L<=0
    error('SSFPG_gram:cache','cache must contain a compatible H and positive finite real L.')
end
H=cache.H;
egmax=cache.L;
isprojector=isa(evalchar,'function_handle');
if ~isprojector && ~(ischar(evalchar) || (isstring(evalchar) && isscalar(evalchar)))
    error('SSFPG_gram:projection','evalchar must be a function handle or code string.')
end

% prepare of relaxations for iterations
t0=load(fullfile(fileparts(mfilename('fullpath')),'t0_1e6_30.mat'));% pre-calculated step sizes
t0=cast(t0.t0,'like',G)/cast(egmax,'like',G); % scaled by the Lipschitz constant
tmax=2/egmax;
repnum=ceil(iter/numel(t0));
t0=repmat(t0,[repnum,1]); % if the iteration exceeds the number of step size prepared, use them repeatedly

%Gob=G'*ob;
%GobE=Gob'*Gob;

% prepare for iterative calculations
if nargout>=5
    Xall=zeros(sg(2),iter,'like',G);% solutions in all iterations
else
    Xall=[];
end

if nargin<8 || isempty(Xinit)
    X=ones(sg(2),1,'like',G); % initial solution
else
    if ~iscolumn(Xinit) || numel(Xinit)~=sg(2) || any(~isfinite(Xinit))
        error('SSFPG_gram:Xinit','Xinit must be a finite column vector with size(G,2) elements.')
    end
    X=cast(Xinit,'like',G);
    if isprojector
        X=evalchar(X);
    else
        eval(evalchar);
    end
end

misfit=zeros(iter,1); % relative misfit
%misfite=zeros(iter,1); % relative misfit
Xdif=zeros(iter,1);
obE=max(ob'*ob,eps(class(ob))); % energy of data

% calculate the normal-equation right-hand side and gradient
c=G'*ob;
HX=H*X;
Xe=c-HX;
t_isrevise=zeros(iter,1);
X_out=X;
bestmisfit=inf;

for i=1:iter
    X0=X;
    X=X+Xe*t0(i);

    if isprojector
        X=evalchar(X);
    else
        eval(evalchar);% compatibility with the original SSFPG interface
    end

    HX=H*X;
    p=X'*c;
    q=X'*HX;
    if isscaling==1
        if q>0
            fac=max(p/q,0);X=X*fac;HX=HX*fac;p=p*fac;q=q*fac^2;
        else
            X(:)=0;HX(:)=0;p=0;q=0;
        end
    end
    misfit(i)=max(obE-2*p+q,0)/obE;
    if misfit(i)<sqrt(eps(class(ob)))
        misfit(i)=norm(ob-G*X)^2/obE;
    end
    if t0(i)>tmax
        if i>1&&misfit(i)-misfit(i-1)>eps
            t0(i)=1.5/egmax;
            X=X0+Xe*t0(i);

            if isprojector
                X=evalchar(X);
            else
                eval(evalchar);
            end

            HX=H*X;
            p=X'*c;
            q=X'*HX;
            if isscaling==1
                if q>0
                    fac=max(p/q,0);X=X*fac;HX=HX*fac;p=p*fac;q=q*fac^2;
                else
                    X(:)=0;HX(:)=0;p=0;q=0;
                end
            end
            misfit(i)=max(obE-2*p+q,0)/obE;
            if misfit(i)<sqrt(eps(class(ob)))
                misfit(i)=norm(ob-G*X)^2/obE;
            end
            t_isrevise(i)=t_isrevise(i)+1;
        end
    end
    if misfit(i)<bestmisfit
        bestmisfit=misfit(i);
        X_out=X;
    end
    Xe=c-HX;

    %misfite(i)=Xe'*Xe/GobE;%(X-X0)'*(X-X0);

    if nargout>=5
        Xall(:,i)=X;
    end

    % stop the iteration if the deepest misfit reduction (t0>tmax) has been stably small
    Xdif(i)=norm(X-X0)/max(norm(X),eps(class(X)));
    if Xdif(i)<Xtol0
        if nargout>=5
            Xall(:,i+1:end)=[];
        end
        misfit(i+1:end)=[];
        %misfite(i+1:end)=[];
        Xdif(i+1:end)=[];
        break;
    end
end
