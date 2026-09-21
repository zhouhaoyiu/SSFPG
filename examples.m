clearvars                                      % 清空工作区中的所有变量，保证示例可重复运行
close all                                      % 关闭所有已打开的图形窗口
% 设置问题的规模
% N=2000 为数据点数，M=1000 为待反演的模型参数个数
N = 2e3;
M = 1e3;
% 构造格林函数与卷积矩阵
g = randn(N, 1);                               % 用随机噪声充当“格林函数”（子波）
G = convmtx(g, M);                             % 由子波构造卷积矩阵（稠密矩阵），尺寸为 N×M
G(N + 1:end, :) = [];                          % 截断多余的行，使卷积矩阵与数据长度一致
% 构造正的“真实解”
% 正弦波并置负值为 0，得到一个非负的真实模型
X = sin(linspace(0, pi * 8, M)');
X(X < 0) = 0;
% 由真实模型正演合成观测数据
ob0 = G * X;                                   % 无噪声的理论数据（正演结果）
ob = ob0 + max(abs(ob0)) .* randn(size(ob0)) .* 0.1;  % 加入 10% 峰值的随机噪声，得到“观测数据”
% 统一设置最大迭代次数
maxiter = 100;                                 % 所有算例的最大迭代次数均取 100
% ---------------------------------------------------------------- （下块开始：普通反卷积）
% regular deconvolution                        % 普通（无约束正则化）反卷积
% 用 SSFPG 求解并计时；核函数为非负投影 + 开启解缩放
tic;
[X1, misfit1] = SSFPG(G, ob, 1e-99, maxiter, 'X(X<0)=0;', 1);
toc
figure                                         % 新建图窗
subplot(121)                                   % 左子图
plot([X, X1])                                  % 对比真实解 X 与反演解 X1
% 横轴为时间，纵轴为幅值
xlabel('Time');
ylabel('Amplitude')
subplot(122)                                   % 右子图
semilogy(misfit1 - min(misfit1) + eps)         % 半对数显示残差随迭代的下降（整体平移到最小残差处）
% 横轴为迭代次数，纵轴为残差差距
xlabel('Iteration');
ylabel('Misfit gap')
% ---------------------------------------------------------------- （下块开始：带最小能量约束的反卷积）
% deconvolution with minimum energy constraint  % 带最小能量约束的反卷积
D = sparse(eye(M));                            % 约束矩阵取单位阵的稀疏形式（对应最小能量约束）
lambda = eigs(G' * G, 1) * 3e-3;               % 由 G'*G 的最大特征值确定约束权重 lambda
% 反演；数据按 [观测; 约束右端项] 堆叠，并计时
tic;
[X2, misfit2] = SSFPG_spar(G, D * lambda, [ob; zeros(M, 1)], 1e-99, maxiter, 'X(X<0)=0;', 1);
toc
figure                                         % 新建图窗
subplot(121)                                   % 左子图
plot([X, X2])                                  % 对比真实解与带约束的反演解
% 横轴为时间，纵轴为幅值
xlabel('Time');
ylabel('Amplitude')
subplot(122)                                   % 右子图
semilogy(misfit2 - min(misfit2) + eps)         % 半对数显示残差下降曲线
% 横轴为迭代次数，纵轴为残差差距
xlabel('Iteration');
ylabel('Misfit gap')
% ---------------------------------------------------------------- （下块开始：多权重约束反卷积）
% deconvolution with multiple weights of constraint  % 对约束施加多重权重并一次反演
mlambda = eigs(G' * G, 1) * (1e-4:1e-4:1e-2);  % 生成一组（100 个）等差权重，均以最大特征值为尺度
% 一次反演同时得到所有权重对应的解，并计时
tic;
[X3, misfit3] = SSFPG_spar_mweits(G, D, mlambda, [ob; zeros(M, 1)], maxiter, 'X(X<0)=0;', 1);
toc
figure                                         % 新建图窗
subplot(121)                                   % 左子图
plot([X3])                                     % 绘制不同权重下的反演解（每列一个权重）
% 横轴为时间，纵轴为幅值
xlabel('Time');
ylabel('Amplitude')
subplot(122)                                   % 右子图
plot(misfit3)                                  % 绘制各权重对应的残差随迭代变化
% 横轴为迭代次数，纵轴为残差
xlabel('Iteration');
ylabel('Misfit')
% ---------------------------------------------------------------- （下块开始：多组观测同时反演）
% deconvolution for multiple obsevations with minimum energy constraint  % 多组观测数据同时反演（含最小能量约束）
num = 100;                                     % 设定观测数据的组数（100 组）
mob = ob0 + max(abs(ob0)) .* randn(size(ob0, 1), num) .* 0.1;  % 为每组观测分别加入 10% 随机噪声，组成 N×num 的数据矩阵
% 批量反演 100 组数据，并计时
tic;
[X4, misfit4] = SSFPG_spar_mob(G, D * lambda, [mob; zeros(M, num)], maxiter, 'X(X<0)=0;', 1);
toc
figure                                         % 新建图窗
subplot(121)                                   % 左子图
plot([X4])                                     % 绘制 100 组反演解（每列一组）
% 横轴为时间，纵轴为幅值
xlabel('Time');
ylabel('Amplitude')
subplot(122)                                   % 右子图
plot(misfit4)                                  % 绘制各组残差随迭代变化
% 横轴为迭代次数，纵轴为残差
xlabel('Iteration');
ylabel('Misfit')
