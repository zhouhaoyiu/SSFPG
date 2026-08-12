clearvars
close all

N=2e3;M=1e3;

g=randn(N,1);% random noise as green's functions
G=convmtx(g,M);% convolution matrix
G(N+1:end,:)=[];% 

X=sin(linspace(0,pi*8,M)');X(X<0)=0; % positive true solution

ob0=G*X; % true data
ob=ob0+max(abs(ob0)).*randn(size(ob0)).*0.1; % noised data

maxiter=100;

% regular deconvolution
tic;[X1,misfit1]=SSFPG(G,ob,1e-99,maxiter,'X(X<0)=0;',1);toc
figure
subplot(121)
plot([X,X1])
xlabel('Time');ylabel('Amplitude')
subplot(122)
semilogy(misfit1-min(misfit1)+eps)
xlabel('Iteration');ylabel('Misfit gap')

% deconvolution with minimum energy constraint
D=sparse(eye(M));
lambda=eigs(G'*G,1)*3e-3;
tic;[X2,misfit2]=SSFPG_spar(G,D*lambda,[ob;zeros(M,1)],1e-99,maxiter,'X(X<0)=0;',1);toc
figure
subplot(121)
plot([X,X2])
xlabel('Time');ylabel('Amplitude')
subplot(122)
semilogy(misfit2-min(misfit2)+eps)
xlabel('Iteration');ylabel('Misfit gap')

% deconvolution with multiple weights of constraint
mlambda=eigs(G'*G,1)*(1e-4:1e-4:1e-2);
tic;[X3,misfit3]=SSFPG_spar_mweits(G,D,mlambda,[ob;zeros(M,1)],maxiter,'X(X<0)=0;',1);toc
figure
subplot(121)
plot([X3])
xlabel('Time');ylabel('Amplitude')
subplot(122)
plot(misfit3)
xlabel('Iteration');ylabel('Misfit')

% deconvolution for multiple obsevations with minimum energy constraint
num=100;% number of observation
mob=ob0+max(abs(ob0)).*randn(size(ob0,1),num).*0.1;
tic;[X4,misfit4]=SSFPG_spar_mob(G,D*lambda,[mob;zeros(M,num)],maxiter,'X(X<0)=0;',1);toc
figure
subplot(121)
plot([X4])
xlabel('Time');ylabel('Amplitude')
subplot(122)
plot(misfit4)
xlabel('Iteration');ylabel('Misfit')