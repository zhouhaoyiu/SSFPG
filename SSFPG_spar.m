function [X_out, misfit, t0, t_isrevise, Xall, Xdif] = SSFPG_spar(G, D, ob, Xtol0, iter, evalchar, isscaling)  % 稠密+稀疏矩阵版
%==========================================================================
% using the SSFPG method to solve [G;D]*X=ob, where G is dense and D is sparse
% -------------------------------------------------------------------------
% Input
%        G: the dense matrix
%        D: the sparse matrix
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
% ------------------- 中文说明（以上英文注释的逐条翻译） -------------------
% 功能：用 SSFPG 方法求解 [G;D]*X = ob，其中 G 为稠密矩阵、D 为稀疏矩阵
%       （把“数据拟合项”与“约束项”按行堆叠成一个整体方程，避免各自单独处理）
% 输入：
%        G：稠密矩阵（通常对应数据方程）
%        D：稀疏矩阵（通常对应平滑/最小能量等正则化约束）
%       ob：观测数据（按 [数据; 约束右端项] 的顺序堆叠）
%    Xtol0：解变化的停止容差，很小；可取极小值（如 1e-99），此时该判据实际失效
%     iter：最大迭代次数
% evalchar：每次迭代把解 X 投影到可行域的指令字符串，例如下界限制 'X(X<0)=0;' 或 'X=max(X,0);'
%           以及/或者上界限制 'X(X>1)=1;'
% isscaling：选项开关；若 isscaling==1，则把解整体缩放以最佳拟合 ob；
%           注意该开关仅在使用非负限制 'X=max(X,0);' 时有效
% 输出：
%    X_out：残差最小的解
%   misfit：归一化残差
%       t0：实际使用的步长序列
% t_isrevise：若第 i 次迭代残差上升，则重新搜索最优步长，此时 t_isrevise(i)=1
%     Xall：各次迭代得到的解（按列存放）
%     Xdif：本轮解相对上一轮解的变化量
% 参考文献：
% Zhang Yong. 2026. An Accelerated Projected-Gradient Method for Large-Scale Nonnegative Earthquake Source Inversions
%==========================================================================
if nargin < 7                                  % 判断调用时是否省略了第 7 个输入参数 isscaling
    isscaling = 0;                             % 省略时默认关闭“解缩放”功能
end                                            % 输入参数默认值处理结束
% 取矩阵尺寸
sg = size(G);                                  % G 的尺寸 [数据行数 模型参数个数]；D 与 G 的列数必须相同
% 下面估计 Lipschitz 常数（[G;D]'*[G;D] 的最大特征值）
egmax = egmax_esti(G, D);                      % 高效估计 G'*G+D'*D 的最大特征值（调用本文件末尾子函数）
% 下面准备迭代所用的松弛（步长）序列
t0 = load('t0_1e6_30.mat');                    % 载入预先计算好的步长序列
t0 = t0.t0 / egmax;                            % 用 Lipschitz 常数对步长归一化缩放
tmax = 2 / egmax;                              % 保证收敛的步长上界 2/egmax
repnum = ceil(iter / numel(t0));               % 计算步长序列需要重复多少遍
t0 = repmat(t0, [repnum, 1]);                  % 迭代超长时把步长序列循环拼接使用
% 下面预分配迭代过程中需要记录的各数组
%Gob = G' * ob(1:sg(1)) + D' * ob(1 + sg(1):end);  % 【原注】投影梯度相关量，此处未使用
%GobE = Gob' * Gob;                            % 【原注】G'*ob 的能量，此处未使用
Xall = zeros(sg(2), iter);                     % 存放所有迭代的解：每列对应一次迭代
% 初始化解向量
X = zeros(sg(2), 1);                           % 初始解取全 0 向量（长度等于模型参数个数）
% 初始化残差与变化量记录数组
misfit = zeros(iter, 1);                       % 归一化残差记录数组
%misfite = zeros(iter, 1);                     % 【原注】另一种残差指标，未启用
Xdif = zeros(iter, 1);                         % 相邻两轮解相对变化量记录数组
obE = ob' * ob;                                % 观测数据（含约束右端项）的能量，用于归一化残差
% 下面计算初始残差与初始梯度
ex = ob - [G * X; D * X];                      % 初始残差：把两块的残差按行堆叠 ex = ob - [G*X; D*X]【原文：sg(1)*sg(2)】
Xe = G' * ex(1:sg(1)) + D' * ex(sg(1) + 1:end);  % 初始梯度：分别取两块残差做 G'、D' 作用后相加【原文：sg(1)*sg(2)】
t_isrevise = zeros(iter, 1);                   % 标记每次迭代是否触发了步长重搜索，初值全 0
% 主迭代循环开始
for i = 1:iter                                 % 逐次迭代，共 iter 次
    X0 = X;                                    % 先保存上一轮的解，用于回溯与计算解的变化量
    X = X + Xe * t0(i);                        % 投影梯度更新：沿下降方向走一步，步长取 t0(i)
    % 执行可行域投影
    eval(evalchar);                            % 按用户指令投影（非负、上界、平滑等）
    % 计算合成数据与新残差
    syn = [G * X; D * X];                      % 正演合成数据：把 G*X 与 D*X 按行堆叠
    if isscaling == 1                          % 若开启了“解缩放”选项
        % 缩放因子 fac=(ob'syn)/(syn'syn)，同步缩放以加速收敛
        fac = ob' * syn ./ (syn' * syn); syn = syn * fac; X = X * fac;
    end                                        % 解缩放分支结束
    ex = ob - syn;                             % 更新残差
    % 记录本轮的归一化残差
    misfit(i) = ex' * ex / obE;                % 归一化残差 = 残差能量 / 数据能量
    if t0(i) > tmax                            % 若当前步长超过了保证收敛的上界 2/egmax
        if i > 1 && misfit(i) - misfit(i - 1) > eps  % 且非首次迭代、且残差反而变大（说明步长过大）
            t0i = logspace(log10(tmax / 2), log10(t0(i)), 40);  % 在 [tmax/2, t0(i)] 上对数等分生成 40 个候选步长
            X = X0 + Xe * t0i;                 % 用 40 个候选步长批量生成 40 个试探解
            % 对批量试探解做同样的投影
            eval(evalchar);                    % 对全部试探解执行相同的可行域投影
            % 对批量试探解做正演
            syn = [G * X; D * X];              % 批量正演合成数据
            if isscaling == 1                  % 若开启了“解缩放”选项
                fac = sum(ob .* syn) ./ sum(syn .* syn);  % 逐列计算缩放因子
                syn = syn .* fac; X = X .* fac;  % 对批量合成数据与批量解同步缩放
            end                                % 批量缩放分支结束
            ex = ob - syn;                     % 批量更新残差
            % 计算每个候选步长对应的残差
            misfitj = diag(ex' * ex) / obE;    % 取对角元得到 40 个候选解各自的归一化残差
            % 挑出残差最小的候选
            [~, nj] = min(misfitj);            % nj 为最优候选步长的序号
            % 用最优候选覆盖本轮结果
            t0(i) = t0i(nj); misfit(i) = misfitj(nj); X = X(:, nj); ex = ex(:, nj);  % 覆盖步长、残差、解、残差向量
            t_isrevise(i) = t_isrevise(i) + 1;  % 标记本轮触发过一次步长重搜索
        end                                    % 步长重搜索分支结束
    end                                        % 步长上界判断结束
    Xe = G' * ex(1:sg(1)) + D' * ex(sg(1) + 1:end);  % 用新残差更新梯度（两块分别作用后相加）
    % 下面一行是另一种停止判据，未启用
    %misfite(i) = Xe' * Xe / GobE;%(X-X0)'*(X-X0);  % 【原注】基于梯度的残差指标
    Xall(:, i) = X;                            % 把本轮解存入 Xall 的第 i 列
    % 收敛判据：当解基本不再变化时停止迭代
    % stop the iteration if the deepest misfit reduction (t0>tmax) has been stably small
    Xdif(i) = norm(X - X0) / norm(X);          % 本轮解相对上一轮解的相对变化量
    if Xdif(i) < Xtol0                         % 若小于给定容差，认为已收敛
        Xall(:, i + 1:end) = [];               % 裁掉 Xall 中未使用的列
        misfit(i + 1:end) = [];                % 裁掉 misfit 中未使用的元素
        Xdif(i + 1:end) = [];                  % 裁掉 Xdif 中未使用的元素
        break;                                 % 跳出主迭代循环
    end                                        % 收敛判断结束
end                                            % 主迭代循环结束
% 从全部迭代结果中选出残差最小的解
[~, n] = min(misfit);                          % n 为残差最小的迭代序号
X_out = Xall(:, n);                            % 输出该次迭代对应的解
% 以下为本文件内的局部子函数：估计最大特征值（Lipschitz 常数）
function [egmax, egmax_rough] = egmax_esti(G, D)  % 子函数声明：估计 G'*G 或 G'*G+D'*D 的最大特征值
% 取模型参数个数
m = size(G, 2);                                % G 的列数即为模型参数个数 m
% 幂迭代所需的初始设置
k = 2;                                         % 粗略估计（幂迭代）的次数
x = randn(m, 1);                               % 随机初始向量
x = x ./ norm(x);                              % 归一化为单位向量
% 分支一：只传入 G 时，估计 G'*G 的最大特征值
if nargin == 1                                 % 判断是否只给了一个输入参数 G
    Bmul = @(x) G' * (G * x);                  % 定义算子句柄 Bx = G'*G*x
    % 幂迭代若干次
    for i = 1:k                                % 幂迭代循环
        x = G' * (G * x);                      % 作用一次算子 B
        x = x / norm(x);                       % 归一化
    end                                        % 幂迭代结束
    egmax_rough = norm(G * x) ^ 2;             % 最大特征值的粗略估计 ‖Gx‖²
else                                           % 分支二：同时传入 G 与 D 时，估计 G'*G+D'*D 的最大特征值
    Bmul = @(x) G' * (G * x) + D' * (D * x);   % 定义算子句柄 Bx = G'*G*x + D'*D*x
    % 幂迭代若干次
    for i = 1:k                                % 幂迭代循环
        x = G' * (G * x) + D' * (D * x);       % 作用一次算子 B
        x = x / norm(x);                       % 归一化
    end                                        % 幂迭代结束
    egmax_rough = norm([G * x; D * x]) ^ 2;    % 最大特征值的粗略估计 ‖[Gx;Dx]‖²
end                                            % 分支结束
% 设置 eigs（Lanczos）求解选项
opts.issym = true;                             % 声明算子为对称矩阵，可显著提速
opts.isreal = true;                            % 声明算子在实数域内
opts.tol = egmax_rough * 1e-8;                 % 收敛容差，以粗略估计值为基准缩放
opts.maxit = 100;                              % 最大迭代次数
% 用 eigs 求绝对值最大的特征值作为 Lipschitz 常数估计
[~, egmax] = eigs(Bmul, m, 1, 'largestabs', opts);  % 取最大特征值（个数为 1）
