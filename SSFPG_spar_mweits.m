function [X,misfit,t0,t_isrevise,Xall]=SSFPG_spar_mweits(G,D,weits,ob,iter,evalchar,isscaling)
%==========================================================================
% using the SSFPG method to solve [G;D*weits]*X=ob, where G is a
% dense matrix and D is a sparse matrix, with multiple weightings applying
% on the D (usually smoothing constraint)
% -------------------------------------------------------------------------
% Input
%        G: the dense matrix
%        D: the sparse matrix
%    weits: a vector containing multiple values
%       ob: the data
%     Xtol: stopping tolerance of solution change, a small number. it can be a very small value (i.e. 1e-99), and then won't be used
%     iter: maximum iteration number
% evalchar: projection of X on the feasible set in each iteration, such as lower bounder limitation: 'X(X<0)=0;' or 'X=max(X,0);' 
%           and/or upper bounder limitation: 'X(X>1)=1;',
%isscaling: an option, is isscaling==1, then the solution will be scaled to
%           best fit ob, note that it works only when non-negative limiation 'X=max(X,0);' is used.
% Output
%    X_out: solution with the minimum residual
%   misfit: normalized misfit
%       t0: used step sizes
% t_isrevise: if the misfit increases at the i-th iteration, then search for an optimized step size, and then t_isrevise(i)=1
%     Xall: solutions obtained at all iterations
%--------------------------------------------------------------------------
% Reference:
% Zhang Yong. 2026. An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions
%==========================================================================
if nargin<7
    isscaling=0;
end
weits=weits(:);

%% get the maximum eigenvalue of G'*G+D'*D*weits.^2 using the power iteration method (simple but converges slowly)
%egmax=egmax_esti_mweits(G,D,weits,100);% using the power iteration method (simple but converges slowly)
egmax=GD_mweits(G,D,weits,20,1e-10);% using the lanczos method (fast)

% prepare of relaxations for iterations
t0=load('t0_1e6_30.mat');t0=t0.t0;t0(t0==0)=[];
t0=repmat(t0,[1,numel(weits)]);
t0=t0./egmax(:)';

%tmax=2./egmax;
repnum=ceil(iter/size(t0,1));
t0=repmat(t0,[repnum,1]);

%Gob=G'*ob(1:sg(1))+D'*ob(1+sg(1):end);
%GobE=Gob'*Gob;

misfit=zeros(iter,numel(weits)); % relative misfit
obE=ob'*ob; % energy of data
%numstop=0;

% prepare for iterative calculations
sg=size(G);
Xall=zeros(sg(2),numel(weits),iter);% solutions in all iterations
% calculate the residual and gradient
X=zeros(sg(2),numel(weits)); % initial solution

%ex=ob-[G*X;D*X];% sg(1)*sg(2)
ex=ob-[G*X;D*(X.*weits(:)')];% sg(1)*sg(2)
Xe=G'*ex(1:sg(1),:)+(D'*ex(sg(1)+1:end,:)).*weits(:)';% sg(1)*sg(2)
t_isrevise=zeros(iter,numel(weits));

for i=1:iter
    X0=X;
    X=X+Xe.*t0(i,:);    
    
    eval(evalchar);% projection, e.g., non-negativity, upper bound limitation, smoothing
        
    % calculate the residual and gradient
    %syn=[G*X;D*X];fac=bestfac(ob,syn);syn=syn*fac;X=X*fac;ex=ob-syn;
    syn=[G*X;D*(X.*weits(:)')];
    if isscaling==1
        fac=sum(ob.*syn)./sum(syn.*syn);syn=syn.*fac;X=X.*fac; % with only non-negativity, the solution can be scaled for rapid converging
    end
    ex=ob-syn;
    
    misfit(i,:)=sum(ex.^2)/obE;
    
    if i>1
        misfit_delta=misfit(i,:)-misfit(i-1,:);
        dex=find(misfit_delta>eps);
        if numel(dex)>0
            t0_dex=1.5/egmax(dex);            
            X_dex=X0(:,dex)+Xe(:,dex).*t0_dex;
            t0(i,dex)=t0_dex;
            
            X(:,dex)=X_dex;
            eval(evalchar);% projection, e.g., non-negativity, upper bound limitation, smoothing
            X_dex=X(:,dex);
            
            syn_dex=[G*X_dex;(D*X_dex).*weits(dex)'];
            if isscaling==1
                fac=sum(ob.*syn_dex)./sum(syn_dex.*syn_dex);syn_dex=syn_dex.*fac;X(:,dex)=X_dex.*fac; % with only non-negativity, the solution can be scaled for rapid converging
            end
            ex_dex=ob-syn_dex;
            ex(:,dex)=ex_dex;
            
            misfit_dex=diag(ex_dex'*ex_dex)/obE;
            misfit(i,dex)=misfit_dex;
            t_isrevise(i,dex)=1;
        end
    end
        
    Xe=G'*ex(1:sg(1),:)+(D'*ex(sg(1)+1:end,:)).*weits(:)';% sg(1)*sg(2)  
       
    Xall(:,:,i)=X;
end

% get the solution with the minimum misfit
for i=1:numel(weits)
    [~,n]=min(misfit(:,i));
    X(:,i)=Xall(:,i,n);
end


% function egmax=egmax_esti_mweits(G,D,weits,kk)
% 
% m=size(G,2);
% 
% numweits=numel(weits);
% x=randn(m,numweits);
% x=x./sqrt(sum(x.^2));
% 
% for i=1:kk
%     x = G'*(G*x)+D'*(D*x).*weits(:)'.^2;
%     x=x./sqrt(sum(x.^2));
% end
% egmax = sum([G*x;(D*x).*weits(:)'].^2);

function [L,info] = GD_mweits(G,D,weights,maxit,tol)
% Estimate lambda_max of [G; w*D]'*[G; w*D]
% simultaneously for multiple weights using batched Lanczos.
%
% INPUT
%   G       : dense N-by-M matrix
%   D       : sparse P-by-M matrix
%   weights : vector [w1,w2,...,wK]
%   maxit   : maximum Lanczos iterations, default 12
%   tol     : relative Ritz residual tolerance, default 1e-3
%
% OUTPUT
%   L       : lambda_max(G'*G + w^2*D'*D) for all weights
%             = sigma_max([G;w*D])^2
%   info.iter
%   info.relres
%
% The expensive G'*G and D'*D matrices are never formed.

if nargin < 4 || isempty(maxit)
    maxit = 12;
end
if nargin < 5 || isempty(tol)
    tol = 1e-3;
end

M = size(G,2);

if size(D,2) ~= M
    error('G and D must have the same number of columns.');
end

w = weights(:).';                 % 1-by-K
K = numel(w);
w2 = w.^2;

%% Initial vectors
Q = randn(M,K);
Q = Q ./ sqrt(sum(Q.^2,1));

Qold = zeros(M,K);
beta_old = zeros(1,K);

alpha = zeros(maxit,K);
beta  = zeros(maxit,K);

L      = zeros(1,K);
relres = inf(1,K);

for j = 1:maxit

    %% ---------------------------------------------------------
    % Batched application of:
    %
    % B_k q_k = G'G q_k + w_k^2 D'D q_k
    %
    % No loop over weights in the expensive operations.
    % ----------------------------------------------------------
    GQ = G*Q;
    DQ = D*Q;

    Z = G'*GQ + D'*(DQ .* w2);

    %% Lanczos three-term recurrence
    if j > 1
        Z = Z - Qold .* beta_old;
    end

    aj = sum(Q .* Z,1);
    Z  = Z - Q .* aj;

    % Small local reorthogonalization
    if j > 1
        c = sum(Qold .* Z,1);
        Z = Z - Qold .* c;
    end

    bj = sqrt(sum(Z.^2,1));

    alpha(j,:) = aj;
    beta(j,:)  = bj;

    %% Ritz values
    % Only tiny j-by-j matrices are handled in this loop.
    % Its cost is negligible compared with G*Q and G'*GQ.
    for k = 1:K

        T = diag(alpha(1:j,k));

        if j > 1
            b = beta(1:j-1,k);
            T = T + diag(b,1) + diag(b,-1);
        end

        [V,E] = eig(T,'vector');
        [L(k),id] = max(E);

        y = V(:,id);

        relres(k) = bj(k)*abs(y(end)) / max(abs(L(k)),eps);
    end

    %% Stop when all weights have converged
    if j >= 3 && all(relres < tol)
        break
    end

    if max(bj) < 100*eps
        break
    end

    Qold = Q;
    beta_old = bj;

    Q = Z ./ max(bj,eps);

end

info.iter   = j;
info.relres = relres;

% Preserve input shape
L = reshape(L,size(weights));
info.relres = reshape(relres,size(weights));
