function [X_out,misfit,t0,t_isrevise,Xall,Xdif]=SSFPG(G,ob,Xtol0,iter,evalchar,isscaling)
%==========================================================================
% using the SSFPG method to solve G*X=ob 
% -------------------------------------------------------------------------
% Input
%        G: the matrix
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
%     Xdif: difference of the solution compared with the previous one
%--------------------------------------------------------------------------
% Reference:
% Zhang Yong. 2026. An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions
%==========================================================================
if nargin<6
    isscaling=0;
end
sg=size(G);

egmax=egmax_esti(G);% more efficient

% prepare of relaxations for iterations
t0=load('t0_1e6_30.mat');% pre-calculated step sizes
t0=t0.t0/egmax; % scaled by the Lipschitz constant
tmax=2/egmax;
repnum=ceil(iter/numel(t0));
t0=repmat(t0,[repnum,1]); % if the iteration exceeds the number of step size prepared, use them repeatedly

%Gob=G'*ob;
%GobE=Gob'*Gob;

% prepare for iterative calculations
Xall=zeros(sg(2),iter);% solutions in all iterations

X=ones(sg(2),1); % initial solution

misfit=zeros(iter,1); % relative misfit
%misfite=zeros(iter,1); % relative misfit
Xdif=zeros(iter,1);
obE=ob'*ob; % energy of data

% calculate the residual and gradient
ex=ob-G*X;% sg(1)*sg(2)
Xe=G'*ex;% sg(1)*sg(2)
t_isrevise=zeros(iter,1);

for i=1:iter
    X0=X;    
    X=X+Xe*t0(i);
    
    eval(evalchar);% projection, e.g., non-negativity, upper bound limitation, smoothing
    
    syn=G*X;
    if isscaling==1
        fac=ob'*syn./(syn'*syn);syn=syn*fac;X=X*fac; % with only non-negativity, the solution can be scaled for rapid converging
    end
    ex=ob-syn;
    
    misfit(i)=ex'*ex/obE;
    if t0(i)>tmax
        if i>1&&misfit(i)-misfit(i-1)>eps
            t0i=logspace(log10(tmax/2),log10(t0(i)),40);
            X=X0+Xe*t0i;
            
            eval(evalchar);% projection, e.g., non-negativity, upper bound limitation, smoothing
            
            syn=G*X;
            if isscaling==1
                fac=sum(ob.*syn)./sum(syn.*syn);
                syn=syn.*fac;X=X.*fac;% with only non-negativity, the solution can be scaled for rapid converging
            end
            ex=ob-syn;
            
            misfitj=diag(ex'*ex)/obE;
            
            [~,nj]=min(misfitj);
                        
            t0(i)=t0i(nj);misfit(i)=misfitj(nj);X=X(:,nj);ex=ex(:,nj);
            t_isrevise(i)=t_isrevise(i)+1;
        end
    end
    Xe=G'*ex;% sg(1)*sg(2)  
    
    %misfite(i)=Xe'*Xe/GobE;%(X-X0)'*(X-X0);
     
    Xall(:,i)=X;
    
    % stop the iteration if the deepest misfit reduction (t0>tmax) has been stably small
    Xdif(i)=norm(X-X0)/norm(X);
    if Xdif(i)<Xtol0
        Xall(:,i+1:end)=[];
        misfit(i+1:end)=[];
        %misfite(i+1:end)=[];
        Xdif(i+1:end)=[];
        break;
    end
end

% get the solution with the minimum misfit
[~,n]=min(misfit);
X_out=Xall(:,n);


function [egmax,egmax_rough]=egmax_esti(G,D)

m=size(G,2);

k=2;
x=randn(m,1);
x=x./norm(x);

if nargin==1
    Bmul = @(x) G'*(G*x);
    
    for i=1:k
        x = G'*(G*x);
        x = x/norm(x);
    end
    egmax_rough = norm(G*x)^2;
else
    Bmul = @(x) G'*(G*x)+D'*(D*x);
    
    for i=1:k
        x = G'*(G*x)+D'*(D*x);
        x = x/norm(x);        
    end
    egmax_rough = norm([G*x;D*x])^2;
end

opts.issym = true;
opts.isreal = true;
opts.tol   = egmax_rough*1e-8;
opts.maxit = 100;

[~, egmax] = eigs(Bmul, m, 1, 'largestabs', opts);