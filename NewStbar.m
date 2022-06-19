% Copy of GetStbar_eig.m
% Update the corresponding elements of the Laplacian matrix when
% decrementing (only for D>>N).
function [meanVector, N, eigenVectors, eigenValues] = NewStbar(dataset,Label, options,eigenThreshold)
% 使用eig代替svd求解来求主特征对
% Input:
%   dataset: DxN matrix
%   Label: 1xN vector
%   eigenThreshold
%   options:
%       beta1
%       beta2
%       decr_ind    ----wrong label index
%       L           ----initial compute L(sparse)
%       normCovMatrix ----(1-beta1)*(R*R') + beta2*speye(N)(only D>>N)
%       recomputeL   ----wether to recompute the Laplacian matrix
% Output:
%   meanVector: Dx1 vector - the mean vector of the raw data
%   N:(numSample) 1x1 value - the number of sample in the raw data
%   eigenVectors: DxR matrix - each column is a principal component, R is the reduced dimension
%   eigenValues: RxR matrix - diagonal matrix storing the eigenvalues of the total scatter matrix
% -----------------------------------------------------------------------
[D,N] = size(dataset);

% computed the new Laplacian matrix L_new
if options.recomputeL
    method = 'exp'; options.k = 5;
    data = options.data;
    [L,~] = build_laplacianD(data,Label,method,options);
    L = sparse(L);
else
    % computed the updated Laplacian matrix L_new
    S = options.S;
    S(:,options.decr_ind) = 0;
    S(options.decr_ind,:) = 0; % decremental data distance set to 0
    method = 'dec_exp'; option.k = 5; option.decr_ind = options.decr_ind;
    Label = zeros(size(dataset,2),1);
    [~,S_dec] = build_laplacianD(dataset,Label,method,option);
    S = S + S_dec;
    L = diag(sum(S,2)) - S;%计算Laplacian矩阵,D取行和
    L = sparse(L);%稀疏存储
end

% Component selection
% 使用QR分解计算
Q = options.Q; R = options.R; 
normCovMatrix = options.normCovMatrix+R*L*R';
normCovMatrix = max(normCovMatrix,normCovMatrix');%保对称
[V,E] = eig(normCovMatrix);

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
eigenVectors = Q*V;
eigenValues = diag(E);
meanVector = options.meanVector;
end

