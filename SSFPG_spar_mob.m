function [X_out,misfit,t0,t_isrevise,Xall]=SSFPG_spar_mob(G,D,ob,iter,evalchar,isscaling)
%==========================================================================
% using the SSFPG method to solving [G;D]*X=ob, where G is a dense matrix and D is a sparse matrix, 
% and ob can be a matrix consisting of multiple observed data 
% -------------------------------------------------------------------------
% Input
%        G: the dense matrix
%        D: the sparse matrix
%       ob: the data, which can be either a vector or a matrix (multiple vectors)
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
if nargin<6
    isscaling=0;
end


%% get the maximum eigenvalue of G'*G+D'*D
egmax=egmax_esti(G,D);% more efficient estimation

% prepare of relaxations for iterations
t0=load('t0_1e6_30.mat');
t0=t0.t0;t0(t0==0)=[];
t0=t0/egmax;

tmax=2/egmax;
repnum=ceil(iter/numel(t0));
t0=repmat(t0,[repnum,1]);

%Gob=G'*ob(1:sg(1),:)+D'*ob(1+sg(1):end,:);
%GobE=sum(Gob.^2);

misfit=zeros(iter,size(ob,2)); % relative misfit
obE=sum(ob.^2); % energy of data

% prepare for iterative calculations
sg=size(G);
Xall=zeros(sg(2),size(ob,2),iter);% solutions in all iterations
% calculate the residual and gradient

X=zeros(sg(2),size(ob,2)); % initial solution

ex=ob-[G*X;D*X];% sg(1)*sg(2)
Xe=G'*ex(1:sg(1),:)+D'*ex(sg(1)+1:end,:);% sg(1)*sg(2)
t_isrevise=zeros(iter,1);

for i=1:iter
    X0=X;    
    X=X+Xe*t0(i);    
    eval(evalchar);% Projection, e.g., positivity
        
    syn=[G*X;D*X];
    
    if isscaling==1
        fac=sum(ob.*syn)./sum(syn.*syn);
        fac=sparse(1:numel(fac),1:numel(fac),fac);syn=syn*fac;X=X*fac;
    end
    
    ex=ob-syn;
    
    misfit(i,:)=sum(ex.^2)./obE;

    % if t0(i) exceeds tmax, misfit(i) needs t0 be smaller than misfit(i-1)
    if t0(i)>tmax
        safeguard=(i>1&misfit(i,:)-misfit(i-1,:)>eps);
        index=find(safeguard==1);
        if numel(index)>0
            X0_re=X0(:,index);
            Xe_re=Xe(:,index);
            ob_re=ob(:,index);
            %X_rea=zeros([size(X,1),20,size(X_re,2)]);
            
            t0i=1.5/egmax;
            X_re=X0_re+Xe_re*t0i;
            
            X(:,index)=X_re;
            eval(evalchar);
            X_re=X(:,index);
            
            syn_re=[G*X_re;D*X_re];
            if isscaling==1
                fac=sum(ob_re.*syn_re)./sum(syn_re.*syn_re);
                fac=fac(:)';syn_re=syn_re.*fac;X_re=X_re.*fac;
            end
            ex_re=ob_re-syn_re;
            misfit_re=sum(ex_re.^2)./obE(index);
            misfit(i,index)=misfit_re;X(:,index)=X_re;ex(:,index)=ex_re;
            
%             t0i=logspace(log10(tmax/2),log10(t0(i)),20);
%             for kk=1:size(X0_re,2)
%                 X_rea0=X0_re(:,kk)+Xe_re(:,kk)*t0i;
%                 X_rea0(X_rea0<0)=0;
%                 syn=[G*X_rea0;D*X_rea0];
%                 if isscaling==1
%                     fac=sum(ob(:,index(kk)).*syn)./sum(syn.*syn);
%                     fac=fac(:)';syn=syn.*fac;X_rea0=X_rea0.*fac;
%                 end
%                 ex0=ob(:,index(kk))-syn;
%                 misfitj=sum(ex0.^2)./obE(index(kk));
%                 [~,nj]=min(misfitj);
%                 misfit(i,index(kk))=misfitj(nj);X(:,index(kk))=X_rea0(:,nj);ex(:,index(kk))=ex0(:,nj);
%             end
        end
    end
        
    Xe=G'*ex(1:sg(1),:)+D'*ex(sg(1)+1:end,:);% sg(1)*sg(2)  
       
    Xall(:,:,i)=X;
    
end

% get the solution with the minimum misfit
X_out=zeros(size(G,2),size(ob,2));
for i=1:size(ob,2)
    [~,n]=min(misfit(:,i));
    X_out(:,i)=Xall(:,i,n);
end

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