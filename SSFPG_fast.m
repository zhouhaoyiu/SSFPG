function [X_out,misfit,t0,t_isrevise,Xall,Xdif,egmax]=SSFPG_fast(G,ob,Xtol0,iter,evalchar,isscaling,egmax,Xinit)
%==========================================================================
% using the SSFPG method to solve G*X=ob
% This version uses a single safeguarded fallback step, a faster conservative
% eigenvalue estimate, optional eigenvalue reuse, and finite scaling at zero.
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
%    egmax: optional cached estimate of lambda_max(G'*G)
%    Xinit: optional warm-start solution
% Output
%    X_out: solution with the minimum residual
%   misfit: normalized misfit
%       t0: used step sizes
% t_isrevise: if the misfit increases at the i-th iteration, then search for an optimized step size, and then t_isrevise(i)=1
%     Xall: solutions obtained at all iterations
%     Xdif: difference of the solution compared with the previous one
%     egmax: estimate used for lambda_max(G'*G), reusable in later calls
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
    error('SSFPG_fast:scaling','isscaling must be 0 or 1.')
end
sg=size(G);
if ~iscolumn(ob) || size(G,1)~=numel(ob)
    error('SSFPG_fast:dimensions','ob must be a column vector with size(G,1) elements.')
end
if iter<1 || iter~=fix(iter)
    error('SSFPG_fast:iterations','iter must be a positive integer.')
end
if ~isscalar(Xtol0) || ~isfinite(Xtol0) || Xtol0<0
    error('SSFPG_fast:tolerance','Xtol0 must be a nonnegative finite scalar.')
end

if nargin<7 || isempty(egmax)
    egmax=egmax_esti(G);
end
if ~isnumeric(egmax) || ~isscalar(egmax) || ~isreal(egmax) || ~isfinite(egmax) || egmax<=0
    error('SSFPG_fast:egmax','egmax must be a positive finite real scalar.')
end
isprojector=isa(evalchar,'function_handle');
if ~isprojector && ~(ischar(evalchar) || (isstring(evalchar) && isscalar(evalchar)))
    error('SSFPG_fast:projection','evalchar must be a function handle or code string.')
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
        error('SSFPG_fast:Xinit','Xinit must be a finite column vector with size(G,2) elements.')
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

% calculate the residual and gradient
ex=ob-G*X;% sg(1)*sg(2)
Xe=G'*ex;% sg(1)*sg(2)
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

    syn=G*X;
    if isscaling==1
        den=syn'*syn;
        if den>0
            fac=max((ob'*syn)/den,0);syn=syn*fac;X=X*fac;
        else
            X(:)=0;syn(:)=0;
        end
    end
    ex=ob-syn;

    misfit(i)=ex'*ex/obE;
    if t0(i)>tmax
        if i>1&&misfit(i)-misfit(i-1)>eps
            t0(i)=1.5/egmax;
            X=X0+Xe*t0(i);

            if isprojector
                X=evalchar(X);
            else
                eval(evalchar);
            end

            syn=G*X;
            if isscaling==1
                den=syn'*syn;
                if den>0
                    fac=max((ob'*syn)/den,0);syn=syn*fac;X=X*fac;
                else
                    X(:)=0;syn(:)=0;
                end
            end
            ex=ob-syn;
            misfit(i)=ex'*ex/obE;
            t_isrevise(i)=t_isrevise(i)+1;
        end
    end
    if misfit(i)<bestmisfit
        bestmisfit=misfit(i);
        X_out=X;
    end
    Xe=G'*ex;% sg(1)*sg(2)

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

function egmax=egmax_esti(G)

m=size(G,2);
if m<8
    egmax=norm(G)^2;
    return
end

Bmul=@(x)G'*(G*x);
opts.issym=true;
opts.isreal=true;
opts.tol=1e-2;
opts.maxit=12;
opts.p=8;
[~,egmax,flag]=eigs(Bmul,m,1,'largestabs',opts);
if flag~=0
    opts.tol=1e-8;
    opts.maxit=100;
    opts.p=min(20,m);
    [~,egmax,flag]=eigs(Bmul,m,1,'largestabs',opts);
    if flag~=0
        error('SSFPG_fast:eigs','Unable to estimate the largest eigenvalue.')
    end
end
egmax=1.01*egmax;
