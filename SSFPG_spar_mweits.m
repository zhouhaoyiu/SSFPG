function [X,misfit,t0,t_isrevise,Xall]=SSFPG_spar_mweits(G,D,weits,ob,iter,evalchar,isscaling)  % 函数声明：对稀疏约束矩阵施加“多重权重”并一次性同时求解的 SSFPG 求解器
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
% ------------------- 中文说明（以上英文注释的逐条翻译） -------------------
% 功能：求解 [G; D*weits]*X = ob，其中 G 为稠密矩阵、D 为稀疏矩阵，
%       并且对 D（通常是平滑约束）施加“多个不同的权重”。
%       一次反演即可同时得到不同权重所对应的全部解，避免逐权重重复计算。
% 输入：
%        G：稠密矩阵
%        D：稀疏矩阵
%    weits：包含多个权重值的向量（每个权重对应一列解）
%       ob：观测数据
%     iter：最大迭代次数
% evalchar：每次迭代把解 X 投影到可行域的指令字符串，例如 'X(X<0)=0;' / 'X=max(X,0);' / 'X(X>1)=1;'
% isscaling：选项开关；若 isscaling==1，则把解整体缩放以最佳拟合 ob（仅在使用 'X=max(X,0);' 时有效）
% 输出：
%    X：残差最小的解（矩阵，每列对应一个权重）
%   misfit：归一化残差（矩阵，每列对应一个权重）
%       t0：实际使用的步长（矩阵，每列对应一个权重）
% t_isrevise：若某步长在该次迭代因残差上升而重搜，则对应位置置 1
%     Xall：各次迭代得到的解（三维数组：解维度 × 权重个数 × 迭代次数）
% 参考文献：
% Zhang Yong. 2026. An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions
%==========================================================================
if nargin<7                                       % 判断调用时是否省略了第 7 个输入参数 isscaling
    isscaling=0;                                  % 省略时默认关闭“解缩放”功能
end                                               % 输入参数默认值处理结束
weits=weits(:);                                   % 把权重数组强制拉成列向量（1 列 K 行），便于按列广播
%% get the maximum eigenvalue of G'*G+D'*D*weits.^2 using the power iteration method (simple but converges slowly)   % 原文小节标题：求 G'*G + D'*D*weits² 的最大特征值（原用幂迭代法，收敛慢）
%egmax=egmax_esti_mweits(G,D,weits,100);% using the power iteration method (simple but converges slowly)   % （保留原注释）幂迭代法：实现简单但收敛慢，未启用
egmax=GD_mweits(G,D,weits,20,1e-10);% using the lanczos method (fast)                                       % 用批量 Lanczos 法（快）一次算出所有权重对应的最大特征值
% 下面准备迭代所用的松弛（步长）序列（按权重逐列构造）
t0=load('t0_1e6_30.mat');t0=t0.t0;t0(t0==0)=[];                       % 载入预设步长序列，并剔除其中为 0 的元素
t0=repmat(t0,[1,numel(weits)]);                                       % 把步长序列横向复制，使每一列对应一个权重
t0=t0./egmax(:)';                                                     % 逐列用各自的最大特征值归一化（按行广播）
%tmax=2./egmax;                                    % （保留原注释）保证收敛的步长上界，此处未使用
repnum=ceil(iter/size(t0,1));                                         % 计算步长序列需要重复多少遍
t0=repmat(t0,[repnum,1]);                                             % 迭代超长时把步长序列循环拼接使用
% 下面预分配记录数组
%Gob=G'*ob(1:sg(1))+D'*ob(1+sg(1):end);            % （保留原注释）投影梯度相关量，未使用
%GobE=Gob'*Gob;                                    % （保留原注释）G'*ob 的能量，未使用
misfit=zeros(iter,numel(weits)); % relative misfit                    % 归一化残差记录：行=迭代次数，列=权重
obE=ob'*ob; % energy of data                                          % 观测数据能量，用于归一化残差
%numstop=0;                                        % （保留原注释）用于提前停止的计数器，未启用
% 初始化解存储与尺寸信息
sg=size(G);                                                           % 取 G 的尺寸 [数据行数 模型参数个数]
Xall=zeros(sg(2),numel(weits),iter);% solutions in all iterations      % 三维数组存放所有解：解维度 × 权重个数 × 迭代次数
% 下面计算初始残差与初始梯度
X=zeros(sg(2),numel(weits)); % initial solution                       % 初始解取全 0 矩阵（列数=权重个数）
%ex=ob-[G*X;D*X];% sg(1)*sg(2)                    % （保留原注释）未加权时的残差写法
ex=ob-[G*X;D*(X.*weits(:)')];% sg(1)*sg(2)                            % 初始残差：约束块按列乘上各自的权重
Xe=G'*ex(1:sg(1),:)+(D'*ex(sg(1)+1:end,:)).*weits(:)';% sg(1)*sg(2)  % 初始梯度：D' 作用后再按列乘权重（与加权残差配套）
t_isrevise=zeros(iter,numel(weits));                                  % 标记各权重在该次迭代是否触发步长重搜索
% 主迭代循环开始
for i=1:iter                                                          % 逐次迭代，共 iter 次
    X0=X;                                                             % 保存上一轮解（所有权重），用于回溯
    X=X+Xe.*t0(i,:);                                                  % 投影梯度更新：逐列使用各自步长 t0(i,:)（按行广播）
    eval(evalchar);% projection, e.g., non-negativity, upper bound limitation, smoothing   % 对全体权重的解执行投影
    % 计算合成数据
    %syn=[G*X;D*X];fac=bestfac(ob,syn);syn=syn*fac;X=X*fac;ex=ob-syn; % （保留原注释）未加权写法
    syn=[G*X;D*(X.*weits(:)')];                                       % 正演合成数据：约束块按列乘权重
    % 可选的解缩放
    if isscaling==1                                                   % 若开启了“解缩放”选项
        fac=sum(ob.*syn)./sum(syn.*syn);syn=syn.*fac;X=X.*fac; % scaled for rapid converging   % 逐列缩放因子，同步缩放合成数据与解
    end                                                               % 解缩放分支结束
    ex=ob-syn;                                                        % 更新残差
    misfit(i,:)=sum(ex.^2)/obE;                                       % 记录本轮各权重的归一化残差
    % 残差上升的列需要重新搜索步长
    if i>1                                                            % 首次迭代无历史残差，跳过
        misfit_delta=misfit(i,:)-misfit(i-1,:);                       % 本轮与上一轮残差之差
        dex=find(misfit_delta>eps);                                   % 找出残差变大的权重序号（步长过大）
        if numel(dex)>0                                               % 若存在需要重搜的权重
            t0_dex=1.5/egmax(dex);                                    % 对每个权重取收缩步长 1.5/λmax
            X_dex=X0(:,dex)+Xe(:,dex).*t0_dex;                        % 从上一轮解沿下降方向生成候选解
            t0(i,dex)=t0_dex;                                         % 把收缩步长登记到步长矩阵
            % 把候选解写回并投影
            X(:,dex)=X_dex;                                           % 候选解写回整批解矩阵
            eval(evalchar);% projection, e.g., non-negativity, upper bound limitation, smoothing   % 对整批解投影
            X_dex=X(:,dex);                                           % 取回投影后的候选解
            % 对候选解正演并计算残差
            syn_dex=[G*X_dex;(D*X_dex).*weits(dex)'];                 % 正演合成数据（按权重乘约束块）
            if isscaling==1                                           % 若开启了“解缩放”选项
                fac=sum(ob.*syn_dex)./sum(syn_dex.*syn_dex);syn_dex=syn_dex.*fac;X(:,dex)=X_dex.*fac; % scaled for rapid converging   % 缩放后写回
            end                                                       % 缩放分支结束
            ex_dex=ob-syn_dex;                                        % 计算候选解残差
            ex(:,dex)=ex_dex;                                         % 覆盖对应权重的残差
            misfit_dex=diag(ex_dex'*ex_dex)/obE;                      % 取对角元得到候选解各权重的归一化残差
            misfit(i,dex)=misfit_dex;                                 % 覆盖对应权重的残差值
            t_isrevise(i,dex)=1;                                      % 标记这些权重在本轮触发过重搜索
        end                                                           % 重搜权重判断结束
    end                                                               % 残差上升判断结束
    Xe=G'*ex(1:sg(1),:)+(D'*ex(sg(1)+1:end,:)).*weits(:)';% sg(1)*sg(2)   % 用新残差更新下降方向（约束块按权重加权）
    Xall(:,:,i)=X;                                                    % 把本轮解存入 Xall 的第 i 层（同一迭代的所有权重）
end                                                                   % 主迭代循环结束
% 从全部迭代结果中按权重分别选出残差最小的解
for i=1:numel(weits)                                                  % 逐个权重挑选
    [~,n]=min(misfit(:,i));                                           % 第 i 个权重中残差最小的迭代序号 n
    X(:,i)=Xall(:,i,n);                                               % 取该权重在最优迭代上的解
end                                                                   % 逐权重挑选结束
% 以下为被注释保留的另一种最大特征值估计（幂迭代版），未启用
% function egmax=egmax_esti_mweits(G,D,weits,kk)                      % （保留原注释）幂迭代估计多权重最大特征值
%                                                                     % （保留原注释）空行
% m=size(G,2);                                                        % （保留原注释）模型参数个数
%                                                                     % （保留原注释）空行
% numweits=numel(weits);                                              % （保留原注释）权重个数
% x=randn(m,numweits);                                                % （保留原注释）随机初始矩阵
% x=x./sqrt(sum(x.^2));                                               % （保留原注释）逐列归一化
%                                                                     % （保留原注释）空行
% for i=1:kk                                                          % （保留原注释）幂迭代循环
%     x = G'*(G*x)+D'*(D*x).*weits(:)'.^2;                            % （保留原注释）作用一次加权算子
%     x=x./sqrt(sum(x.^2));                                           % （保留原注释）逐列归一化
% end                                                                 % （保留原注释）幂迭代结束
% egmax = sum([G*x;(D*x).*weits(:)'].^2);                             % （保留原注释）最大特征值估计
% 以下为本文件内的局部子函数：批量 Lanczos 法估计多权重最大特征值
function [L,info] = GD_mweits(G,D,weights,maxit,tol)                  % 子函数声明：同时估计多个权重下 [G; w*D]'*[G; w*D] 的最大特征值
% Estimate lambda_max of [G; w*D]'*[G; w*D]                           % 原文注释：估计 [G; w*D]'*[G; w*D] 的最大特征值
% simultaneously for multiple weights using batched Lanczos.          % 原文注释：用批量 Lanczos 法对多个权重同时求解
%                                                                     % 原文注释：空行
% INPUT                                                               % 原文注释：输入
%   G       : dense N-by-M matrix                                     % 原文注释：稠密矩阵 G，N×M
%   D       : sparse P-by-M matrix                                    % 原文注释：稀疏矩阵 D，P×M
%   weights : vector [w1,w2,...,wK]                                   % 原文注释：权重向量
%   maxit   : maximum Lanczos iterations, default 12                  % 原文注释：最大 Lanczos 迭代次数，默认 12
%   tol     : relative Ritz residual tolerance, default 1e-3          % 原文注释：相对 Ritz 残差容差，默认 1e-3
%                                                                     % 原文注释：空行
% OUTPUT                                                              % 原文注释：输出
%   L       : lambda_max(G'*G + w^2*D'*D) for all weights             % 原文注释：各权重的最大特征值
%             = sigma_max([G;w*D])^2                                  % 原文注释：亦等于 [G;w*D] 最大奇异值的平方
%   info.iter                                                         % 原文注释：实际迭代次数
%   info.relres                                                       % 原文注释：相对残差
%                                                                     % 原文注释：空行
% The expensive G'*G and D'*D matrices are never formed.              % 原文注释：全程不显式构造昂贵的 G'*G 与 D'*D 矩阵
% 参数默认值处理
if nargin < 4 || isempty(maxit)                                       % 若未给定最大迭代次数
    maxit = 12;                                                       % 默认 12
end                                                                   % 默认值分支结束
if nargin < 5 || isempty(tol)                                         % 若未给定收敛容差
    tol = 1e-3;                                                       % 默认 1e-3
end                                                                   % 默认值分支结束
% 维度检查
M = size(G,2);                                                        % 模型参数个数 M（G 的列数）
if size(D,2) ~= M                                                     % 检查 D 与 G 的列数是否一致
    error('G and D must have the same number of columns.');           % 不一致则报错退出
end                                                                   % 维度检查结束
% 权重整理
w = weights(:).';                 % 1-by-K                           % 权重整理为行向量 1×K
K = numel(w);                                                         % 权重个数 K
w2 = w.^2;                                                            % 预先算好权重的平方，供算子中使用
%% Initial vectors                                                     % 原文小节标题：初始化向量
Q = randn(M,K);                                                       % 随机初始化 K 个列向量（每个权重一个）
Q = Q ./ sqrt(sum(Q.^2,1));                                           % 逐列归一化为单位向量
% Lanczos 递推所需的上一轮量
Qold = zeros(M,K);                                                    % 上一轮的 Q（首次迭代置 0）
beta_old = zeros(1,K);                                                % 上一轮的 beta（首次迭代置 0）
% 预分配三对角矩阵的对角/次对角元素
alpha = zeros(maxit,K);                                               % 三对角矩阵的对角元
beta  = zeros(maxit,K);                                               % 三对角矩阵的次对角元
% 预分配输出
L      = zeros(1,K);                                                  % 最大特征值估计（每个权重一个）
relres = inf(1,K);                                                    % 相对 Ritz 残差（初值置无穷大）
% Lanczos 主循环
for j = 1:maxit                                                       % 逐个 Lanczos 步迭代
    %% ---------------------------------------------------------       % 原文小节标题：批量算子作用说明
    % Batched application of:                                         % 原文注释：批量作用下述算子
    %                                                                 % 原文注释：空行
    % B_k q_k = G'G q_k + w_k^2 D'D q_k                               % 原文注释：算子定义（每个权重一个）
    %                                                                 % 原文注释：空行
    % No loop over weights in the expensive operations.               % 原文注释：昂贵的矩阵运算中不做权重循环（批量化）
    % ----------------------------------------------------------      % 原文注释：说明结束
    GQ = G*Q;                                                         % 预先计算 G*Q，供 G'*GQ 复用
    DQ = D*Q;                                                         % 预先计算 D*Q，供 D'*(DQ.*w2) 复用
    Z = G'*GQ + D'*(DQ .* w2);                                        % 批量作用算子 B：G'GQ + w²(D'DQ)，一次算完所有权重
    %% Lanczos three-term recurrence                                  % 原文小节标题：Lanczos 三项递推
    if j > 1                                                          % 首轮没有上一轮分量，跳过
        Z = Z - Qold .* beta_old;                                     % 减去上一轮方向的分量（三项递推的第三步）
    end                                                               % 递推分支结束
    aj = sum(Q .* Z,1);                                               % 计算当前方向上的投影（三对角对角元 alpha_j）
    Z  = Z - Q .* aj;                                                 % 减去当前方向的分量，得到新正交方向
    % Small local reorthogonalization                                 % 原文注释：小的局部再正交化，抑制数值正交性丢失
    if j > 1                                                          % 首轮无需再正交化
        c = sum(Qold .* Z,1);                                         % 与上一轮方向的内积
        Z = Z - Qold .* c;                                            % 扣除该分量
    end                                                               % 再正交化分支结束
    bj = sqrt(sum(Z.^2,1));                                           % 新方向的模长（三对角次对角元 beta_j）
    alpha(j,:) = aj;                                                  % 记录对角元
    beta(j,:)  = bj;                                                  % 记录次对角元
    %% Ritz values                                                    % 原文小节标题：求 Ritz 值
    % Only tiny j-by-j matrices are handled in this loop.             % 原文注释：此循环只处理极小的 j×j 矩阵
    % Its cost is negligible compared with G*Q and G'*GQ.             % 原文注释：其开销相比 G*Q、G'*GQ 可忽略
    for k = 1:K                                                       % 逐个权重构造小规模三对角矩阵并求特征值
        T = diag(alpha(1:j,k));                                       % 用前 j 个对角元构造三对角矩阵
        if j > 1                                                      % 若已有多于一行
            b = beta(1:j-1,k);                                        % 取前 j-1 个次对角元
            T = T + diag(b,1) + diag(b,-1);                           % 叠加次对角线（上下各一条）
        end                                                           % 三对角矩阵构造结束
        [V,E] = eig(T,'vector');                                      % 求该小矩阵的特征值与特征向量
        [L(k),id] = max(E);                                           % 取最大特征值（即当前 Ritz 值近似）
        y = V(:,id);                                                  % 对应的特征向量
        relres(k) = bj(k)*abs(y(end)) / max(abs(L(k)),eps);           % 相对 Ritz 残差，用于判断是否收敛
    end                                                               % 逐权重循环结束
    %% Stop when all weights have converged                           % 原文小节标题：所有权重都收敛后停止
    if j >= 3 && all(relres < tol)                                    % 至少迭代 3 步且全部权重残差达标
        break                                                         % 提前结束 Lanczos 迭代
    end                                                               % 收敛分支结束
    if max(bj) < 100*eps                                              % 若新方向模长已趋 0（子空间已穷尽）
        break                                                         % 结束迭代，避免除零
    end                                                               % 退化分支结束
    % 保存当前方向并推进
    Qold = Q;                                                         % 当前方向变为“上一轮方向”
    beta_old = bj;                                                    % 记录本次的 beta
    Q = Z ./ max(bj,eps);                                             % 归一化得到下一轮方向（用 max 防止除 0）
end                                                                   % Lanczos 主循环结束
% 汇总输出信息
info.iter   = j;                                                      % 实际执行的 Lanczos 迭代次数
info.relres = relres;                                                 % 各权重的相对残差
% Preserve input shape                                                % 原文注释：保持输出形状与输入权重形状一致
L = reshape(L,size(weights));                                         % 把 L 还原为 weights 的形状
info.relres = reshape(relres,size(weights));                          % 把 relres 还原为 weights 的形状
