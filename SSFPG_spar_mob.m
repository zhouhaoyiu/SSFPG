function [X_out,misfit,t0,t_isrevise,Xall]=SSFPG_spar_mob(G,D,ob,iter,evalchar,isscaling)  % 函数声明：多组观测数据同时反演（批量）的 SSFPG 求解器
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
% ------------------- 中文说明（以上英文注释的逐条翻译） -------------------
% 功能：求解 [G;D]*X = ob，其中 G 为稠密矩阵、D 为稀疏矩阵，
%       而 ob 可以是“多列”的观测数据矩阵（即同时反演多个数据集）
% 输入：
%        G：稠密矩阵
%        D：稀疏矩阵（约束矩阵）
%       ob：观测数据，可以是向量，也可以是矩阵（每一列代表一组独立观测）
%     iter：最大迭代次数
% evalchar：每次迭代把解 X 投影到可行域的指令字符串，例如 'X(X<0)=0;' / 'X=max(X,0);' / 'X(X>1)=1;'
% isscaling：选项开关；若 isscaling==1，则把解整体缩放以最佳拟合 ob（仅在使用 'X=max(X,0);' 时有效）
% 输出：
%    X_out：残差最小的解（矩阵形式，每列对应一组观测）
%   misfit：归一化残差（矩阵形式，每列对应一组观测）
%       t0：实际使用的步长序列
% t_isrevise：若第 i 次迭代残差上升并触发步长重搜索，则 t_isrevise(i)=1
%     Xall：各次迭代得到的解（三维数组：解维度 × 数据组数 × 迭代次数）
% 参考文献：
% Zhang Yong. 2026. An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions
%==========================================================================
if nargin<6                                       % 判断调用时是否省略了第 6 个输入参数 isscaling
    isscaling=0;                                  % 省略时默认关闭“解缩放”功能
end                                               % 输入参数默认值处理结束
%% get the maximum eigenvalue of G'*G+D'*D        % 原文小节标题：求 G'*G+D'*D 的最大特征值
egmax=egmax_esti(G,D);% more efficient estimation           % 高效估计 Lipschitz 常数（调用本文件末尾子函数）
% 下面准备迭代所用的松弛（步长）序列
t0=load('t0_1e6_30.mat');                                   % 载入预先计算好的步长序列
t0=t0.t0;t0(t0==0)=[];                                      % 取出步长向量，并剔除其中为 0 的元素
t0=t0/egmax;                                                % 用 Lipschitz 常数对步长归一化缩放
% 计算步长上界并拼接步长序列
tmax=2/egmax;                                               % 保证收敛的步长上界 2/egmax
repnum=ceil(iter/numel(t0));                                % 计算步长序列需要重复多少遍
t0=repmat(t0,[repnum,1]);                                   % 迭代超长时把步长序列循环拼接使用
% 下面预分配与记录数组（注意：多组数据的量都放在“列”方向）
%Gob=G'*ob(1:sg(1),:)+D'*ob(1+sg(1):end,:);                  % （保留原注释）投影梯度相关量，此处未使用
%GobE=sum(Gob.^2);                                           % （保留原注释）G'*ob 的能量，此处未使用
misfit=zeros(iter,size(ob,2)); % relative misfit            % 归一化残差记录：行=迭代次数，列=数据组
obE=sum(ob.^2); % energy of data                            % 各组观测数据的能量（按列求和），用于归一化残差
% 初始化解存储与尺寸信息
sg=size(G);                                                 % 取 G 的尺寸 [数据行数 模型参数个数]
Xall=zeros(sg(2),size(ob,2),iter);% solutions in all iterations      % 三维数组存放所有解：解维度 × 数据组数 × 迭代次数
% 下面计算初始残差与初始梯度
X=zeros(sg(2),size(ob,2)); % initial solution                % 初始解取全 0 矩阵（列数=数据组数）
ex=ob-[G*X;D*X];% sg(1)*sg(2)                                % 初始残差：ob 与堆叠正演结果 [G*X;D*X] 之差
Xe=G'*ex(1:sg(1),:)+D'*ex(sg(1)+1:end,:);% sg(1)*sg(2)      % 初始梯度：两块残差分别经 G'、D' 作用后按列相加
t_isrevise=zeros(iter,1);                                    % 标记每次迭代是否触发过步长重搜索
% 主迭代循环开始
for i=1:iter                                                 % 逐次迭代，共 iter 次
    X0=X;                                                    % 保存上一轮解（整批数据），用于回溯
    X=X+Xe*t0(i);                                            % 投影梯度更新：步长 t0(i) 为标量，对整批数据同时生效
    eval(evalchar);% Projection, e.g., positivity            % 对整批解执行可行域投影（如非负投影）
    % 计算合成数据
    syn=[G*X;D*X];                                           % 正演合成数据：把 G*X 与 D*X 按行堆叠
    % 可选的解缩放
    if isscaling==1                                          % 若开启了“解缩放”选项
        fac=sum(ob.*syn)./sum(syn.*syn);                     % 逐列计算缩放因子（每列一组数据）
        fac=sparse(1:numel(fac),1:numel(fac),fac);syn=syn*fac;X=X*fac;   % 构造成对角稀疏矩阵，一次性完成整批缩放
    end                                                      % 解缩放分支结束
    % 计算残差并记录
    ex=ob-syn;                                               % 更新残差
    misfit(i,:)=sum(ex.^2)./obE;                             % 记录本轮各组数据的归一化残差
    % if t0(i) exceeds tmax, misfit(i) needs t0 be smaller than misfit(i-1)   % 原文注释：步长过大时残差必须下降
    if t0(i)>tmax                                            % 若当前步长超过保证收敛的上界
        safeguard=(i>1&misfit(i,:)-misfit(i-1,:)>eps);       % 逐列判断：非首轮且该列残差变大（需要重搜步长的列）
        index=find(safeguard==1);                            % 找出所有需要重搜步长的数据组序号
        if numel(index)>0                                    % 若存在需要重搜的数据组
            X0_re=X0(:,index);                               % 取出这些组上一轮的解
            Xe_re=Xe(:,index);                               % 取出这些组当前的下降方向
            ob_re=ob(:,index);                               % 取出这些组的观测数据
            %X_rea=zeros([size(X,1),20,size(X_re,2)]);       % （保留原注释）批量试探解缓存，未使用
            % 用一个缩小后的固定步长重新试算
            t0i=1.5/egmax;                                   % 重搜时取的收缩步长 1.5/egmax
            X_re=X0_re+Xe_re*t0i;                            % 由上一轮解沿下降方向生成新的试探解
            % 把试探解写回整体并投影
            X(:,index)=X_re;                                 % 先把试探解写回整批解矩阵
            eval(evalchar);                                  % 对整批解做投影（只有 index 列被改动）
            X_re=X(:,index);                                 % 取回投影后的试探解
            % 对试探解正演并计算残差
            syn_re=[G*X_re;D*X_re];                          % 正演合成数据
            if isscaling==1                                  % 若开启了“解缩放”选项
                fac=sum(ob_re.*syn_re)./sum(syn_re.*syn_re); % 逐列计算缩放因子
                fac=fac(:)';syn_re=syn_re.*fac;X_re=X_re.*fac;   % 按行向量逐列缩放合成数据与解
            end                                              % 缩放分支结束
            ex_re=ob_re-syn_re;                              % 计算试探解的残差
            misfit_re=sum(ex_re.^2)./obE(index);             % 计算试探解的归一化残差
            misfit(i,index)=misfit_re;X(:,index)=X_re;ex(:,index)=ex_re;   % 用试探结果覆盖这些组的解与残差
%             t0i=logspace(log10(tmax/2),log10(t0(i)),20);            % （保留原注释）另一种做法：对每组数据各试 20 个步长
%             for kk=1:size(X0_re,2)                                  % （保留原注释）逐组循环
%                 X_rea0=X0_re(:,kk)+Xe_re(:,kk)*t0i;                 % （保留原注释）生成该组的候选解
%                 X_rea0(X_rea0<0)=0;                                 % （保留原注释）非负投影
%                 syn=[G*X_rea0;D*X_rea0];                            % （保留原注释）正演
%                 if isscaling==1                                     % （保留原注释）若开启缩放
%                     fac=sum(ob(:,index(kk)).*syn)./sum(syn.*syn);   % （保留原注释）计算缩放因子
%                     fac=fac(:)';syn=syn.*fac;X_rea0=X_rea0.*fac;    % （保留原注释）缩放合成数据与解
%                 end                                                 % （保留原注释）缩放分支结束
%                 ex0=ob(:,index(kk))-syn;                            % （保留原注释）计算残差
%                 misfitj=sum(ex0.^2)./obE(index(kk));                % （保留原注释）计算归一化残差
%                 [~,nj]=min(misfitj);                                % （保留原注释）取残差最小的候选
%                 misfit(i,index(kk))=misfitj(nj);X(:,index(kk))=X_rea0(:,nj);ex(:,index(kk))=ex0(:,nj);   % （保留原注释）覆盖结果
%             end                                                     % （保留原注释）循环结束
        end                                                  % 重搜数据组判断结束
    end                                                      % 步长上界判断结束
    % 更新梯度并保存本轮解
    Xe=G'*ex(1:sg(1),:)+D'*ex(sg(1)+1:end,:);% sg(1)*sg(2)  % 用新残差更新整批数据的下降方向
    Xall(:,:,i)=X;                                           % 把本轮解存入 Xall 的第 i 层（同一批全部数据组）
end                                                          % 主迭代循环结束
% get the solution with the minimum misfit                   % 原文注释：取残差最小的解作为结果
X_out=zeros(size(G,2),size(ob,2));                           % 预分配输出解矩阵
for i=1:size(ob,2)                                           % 逐组（逐列）挑出残差最小的迭代结果
    [~,n]=min(misfit(:,i));                                  % 第 i 组数据中残差最小的迭代序号 n
    X_out(:,i)=Xall(:,i,n);                                  % 取该组在最优迭代上的解
end                                                          % 逐组挑选结束
% 以下为本文件内的局部子函数：估计最大特征值（Lipschitz 常数）
function [egmax,egmax_rough]=egmax_esti(G,D)                 % 子函数声明：估计 G'*G 或 G'*G+D'*D 的最大特征值
% 取模型参数个数
m=size(G,2);                                                 % G 的列数即为模型参数个数 m
% 幂迭代所需的初始设置
k=2;                                                         % 粗略估计（幂迭代）的次数
x=randn(m,1);                                                % 随机初始向量
x=x./norm(x);                                                % 归一化为单位向量
% 分支一：只传入 G 时，估计 G'*G 的最大特征值
if nargin==1                                                 % 判断是否只给了一个输入参数 G
    Bmul = @(x) G'*(G*x);                                    % 定义算子句柄 Bx = G'*G*x
    % 幂迭代若干次
    for i=1:k                                                % 幂迭代循环
        x = G'*(G*x);                                        % 作用一次算子 B
        x = x/norm(x);                                       % 归一化
    end                                                      % 幂迭代结束
    egmax_rough = norm(G*x)^2;                               % 最大特征值的粗略估计 ‖Gx‖²
else                                                         % 分支二：同时传入 G 与 D 时，估计 G'*G+D'*D 的最大特征值
    Bmul = @(x) G'*(G*x)+D'*(D*x);                           % 定义算子句柄 Bx = G'*G*x + D'*D*x
    % 幂迭代若干次
    for i=1:k                                                % 幂迭代循环
        x = G'*(G*x)+D'*(D*x);                               % 作用一次算子 B
        x = x/norm(x);                                       % 归一化
    end                                                      % 幂迭代结束
    egmax_rough = norm([G*x;D*x])^2;                         % 最大特征值的粗略估计 ‖[Gx;Dx]‖²
end                                                          % 分支结束
% 设置 eigs（Lanczos）求解选项
opts.issym = true;                                           % 声明算子为对称矩阵，可显著提速
opts.isreal = true;                                          % 声明算子在实数域内
opts.tol   = egmax_rough*1e-8;                               % 收敛容差，以粗略估计值为基准缩放
opts.maxit = 100;                                            % 最大迭代次数
% 用 eigs 求绝对值最大的特征值作为 Lipschitz 常数估计
[~, egmax] = eigs(Bmul, m, 1, 'largestabs', opts);           % 取最大特征值（个数为 1）
