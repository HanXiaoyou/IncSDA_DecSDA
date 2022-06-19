function [L]=build_laplacian(Xs,method,option)
%   Input:
%       data:
%           Xs     ------样本数矩阵D×N
%       method:      
%           'exp'       ------使用基础的指数计算
%           'exp slow'  ------使用快速的指数计算
%           'cos'       ------使用cos值计算

%       option:      
%           k    ------k近邻参数
%   Output:
%      	L            ----laplacian矩阵N×N
[D,N]=size(Xs);
S = zeros(N,N);
switch method
    case 'exp'%p-nearest neighbor graph
        k = option.k;
        [neighborIds,neighborDistances,~] = kNearestNeighbors(Xs',Xs',k+1);
        % sigma = mean(mean(Dist));%依据CaiDeng选取核函数的sigma
        sigma = mean(mean(neighborDistances(:,2:k+1)));%只用选择k近邻后的距离求sigma
        neighborDistances = exp(-neighborDistances/(2*sigma^2));
        for i=1:N
            S(i,neighborIds(i,2:k+1)) = neighborDistances(i,2:k+1);%从第二列开始取，去掉自身的距离
        end
    case 'exp slow'%p-nearest neighbor graph
        k = option.k;
        for i=1:N
            [sortval,sortpos] = pdist2(Xs',Xs(:,i)','euclidean','Smallest',k);
            sigma = mean(mean(sortval));%取k近邻的距离近似求sigma
            S(sortpos,i) = exp(-sortval.^2/(2*sigma^2));
        end
    case 'cos'
        k = option.k;
        [neighborIds,neighborDistances,Dist] = kNearestNeighbors(Xs',Xs',k+1);
        neighborDistances = squareform(1-pdist(Xs','cosine'));
        for i=1:N
            S(i,neighborIds(i,2:k+1)) = neighborDistances(i,2:k+1);%从第二列开始取，去掉自身的距离
        end  
end
S = (S+S')/2;%保证矩阵的对称性；
L = diag(sum(S,2)) - S;%计算Laplacian矩阵,D取行和
