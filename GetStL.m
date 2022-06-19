% Copy of StGetStbar_eig.m
% Difference: Change the formation of the laplacian matrix L to the SRDA method
function [meanVector, N, eigenVectors, eigenValues, index] = GetStL(dataset,Label, options,eigenThreshold)
% 使用eig代替svd求解来求主特征对
% Input:
%   dataset: DxN matrix
%   Label: 1xN vector
%   eigenThreshold
%   options:
%       beta1
%       beta2
% Output:
%   meanVector: Dx1 vector - the mean vector of the raw data
%   N:(numSample) 1x1 value - the number of sample in the raw data
%   eigenVectors: DxR matrix - each column is a principal component, R is the reduced dimension
%   eigenValues: RxR matrix - diagonal matrix storing the eigenvalues of the total scatter matrix
%   index: r_t*1 vextor - dimensionality reduction of St_bar 
% -----------------------------------------------------------------------
beta1 = options.beta1; beta2=options.beta2; 
[D,N] = size(dataset);%N
meanVector = mean(dataset, 2);
data = dataset - repmat(meanVector, 1, N); %数据中心化处理

% 使用knn与高斯核结合计算laplacian矩阵
% method = 'exp'; option.k = 5;
% L = build_laplacianD(data,Label,method,option);
% L = sparse(L);%稀疏存储
% % 使用cos计算laplacian矩阵
% method = 'cos'; option.k = 5;
% L = build_laplacian(featureVectorInCol,method,option);

W = constructW(data',options);%data:N*D

Dd = full(sum(W,2));
sizeW = length(Dd);
if isfield(options,'LaplacianNorm') && options.LaplacianNorm
    Dd=sqrt(1./Dd);
    Dd=spdiags(Dd,0,sizeW,sizeW);
    W=Dd*W*Dd;
    L=speye(sizeW)-W;
else
    L = spdiags(Dd,0,sizeW,sizeW)-W;
end

% Component selection
if D>=N
    % 使用QR分解计算
    [Q,R] = qr(data,0);
    normCovMatrix = (1-beta1)*(R*R')+R*L*R'+beta2*speye(N);
    normCovMatrix = max(normCovMatrix,normCovMatrix');%保对称
    [V,E] = eig(normCovMatrix);
else
    % 直接使用eig计算
    normCovMatrix = (1-beta1)*(data*data')+beta1*data*L*data'+beta2*eye(D);
    normCovMatrix = max(normCovMatrix,normCovMatrix');%保对称
    [V, E] = eig(normCovMatrix);
end

% 获取特征值和特征向量
[~, idx] = sort( diag(E), 'descend');%取最大的d个特征值，按降序排列
V = V(:,idx);
E = E(idx,idx);

if (~exist('options.k','var'))
    testRow = diag(E);
    testIdx = find(testRow>eigenThreshold);
    k = size(testIdx,1);
else
    k = options.k;
end
V = V(:,1:k);
E = diag(E(1:k,1:k));
index = idx(1:k);%r_t*1
if D>=N
    eigenVectors = Q*V;
else
    eigenVectors = V;
end
eigenValues = diag(E);

end

