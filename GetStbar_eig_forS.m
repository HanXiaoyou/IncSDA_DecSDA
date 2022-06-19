function [meanVector, N, eigenVectors, eigenValues, dec_element] = GetStbar_eig_forS(dataset,Label, options,eigenThreshold)
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
%   dec_element method:
%         index: r_t*1 vextor - dimensionality reduction of St_bar 
%         S:similarity matrix NxN  -the number of whole data set
%         normCovMatrix_singular: ->  (1-beta1)*(R*R') + beta2*speye(N)(only D>>N)
%         Q: qr decomposition of X
%         R: qr decomposition of X
%         meanVector: mean vector of X
% -----------------------------------------------------------------------
beta1 = options.beta1; beta2=options.beta2; 
[D,N] = size(dataset);%N
meanVector = mean(dataset, 2);
data = dataset - repmat(meanVector, 1, N); 

% 
method = 'knn'; option.k = 5;
[L,S] = build_laplacianD(data,Label,method,option);
L = sparse(L);
% % 
% method = 'cos'; option.k = 5;
% L = build_laplacian(featureVectorInCol,method,option);

% Component selection
if D>=N
    % 使用QR分解计算
    [Q,R] = qr(data,0);
%     normCovMatrix_singular = (1-beta1)*(R*R') + beta2*speye(N);
    normCovMatrix_singular = R*R' + beta2*speye(N);
    normCovMatrix = normCovMatrix_singular + R*L*R';
    normCovMatrix = max(normCovMatrix,normCovMatrix');%保对�?
    [V,E] = eig(normCovMatrix);
    
    dec_element.normCovMatrix_singular = normCovMatrix_singular;
    dec_element.Q = Q;
    dec_element.R = R;
else
    % 直接使用eig计算
%     normCovMatrix = (1-beta1)*(data*data')+beta1*data*L*data'+beta2*eye(D);
    normCovMatrix = data*data'+beta1*data*L*data'+beta2*eye(D);
    normCovMatrix = max(normCovMatrix,normCovMatrix');%保对�?
    [V, E] = eig(normCovMatrix);
    
     dec_element.normCovMatrix_singular = normCovMatrix;
end

% 获取特征值和特征向量
[~, idx] = sort( abs(diag(E)), 'descend');%取最大的d个特征�?�，按降序排�?
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

% for decremental method 
dec_element.S = S;
dec_element.meanVector = meanVector;
dec_element.data = data;
end

