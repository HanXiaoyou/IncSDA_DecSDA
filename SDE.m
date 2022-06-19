function [V, time] = SDE(data,untrainSet ,options,k)
% 局部判别分析(semi-supervised Discriminant Embedding, SDE)
% 求 A*x = a*B*x 特征值问题
%     Input:
%          data         ----训练样本集，每行表示一副人脸，在SDE中作为标记的数据集;
%          testdata     ----测试样本集，每行表示一副人脸，在SDE中作为未标记的数据集;
%          options      ----MATLAB中的结构体,它的参数包括
%              'gnd'        ----训练数据所属类别标签，必须提供;
%     Output:
%          V    ----特征向量;
%
%%
t0=clock;
[m,~]=size(data);
wholeSet=[data;untrainSet];%SDE中要求训练集为一部分标记(data)一部分未标记(untrainSet)
wholeSet=wholeSet/norm(wholeSet,'fro');%对数据进行初始化
data=wholeSet(1:m,:);

%设定参数
r=10^(-6);
p=10^(-3);
%-----------------------------构造矩阵S,Sw,Sb--------------------------%
[S,Sw, Sb] = Construct_S3(wholeSet,data, options);
D = diag(sum(S));   L = D - S;
Dw = diag(sum(Sw)); Lw = Dw - Sw; 
Db = diag(sum(Sb)); Lb = Db - Sb; 

% Y v = eXv 特征值问题
X = data' * Lw * data; X = max(X, X'); % 对称矩阵
Y = data' * Lb * data; Y = max(Y, Y');      
X1 = wholeSet' * L * wholeSet; X1 = max(X1, X1');
A = Y; 
n = size(X,1);
B = X + r*X1 + p*speye(n);

%----------------------------RSDE的优化问题-------------------------%
[V ,E] = eig(A,B);
[~, idx] = sort( diag(E), 'descend');%取最大的k-1个特征值
V = V(:,idx);

%%%%

V= V(:,1:k-1);
V = V * diag(1./(sum(V.^2).^0.5));%对特征向量进行归一化
V=orth(V);
time=etime(clock,t0);

end

