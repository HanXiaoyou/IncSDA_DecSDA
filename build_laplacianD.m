function [L,S]=build_laplacianD(Xs,featureLabel,method,option)
%   Input:
%       data:
%           Xs     ------样本数矩阵D×N
%           featureLabel -----样本标签
%       method:
%           'exp'       ------使用基础的指数计算
%           'exp slow'  ------使用快速的指数计算
%           'cos'       ------使用cos值计算
%           'dec_exp'   ------update similarity matrix for decremental method
%       option:
%           k    ------k近邻参数
%           decr_ind    ------index of decremental data(only in 'dec_exp') 1xN_dec
%   Output:
%      	L            ----laplacian矩阵NxN
%       S            ----similarity matrix NxN
% 2020 PWR
[~,N]=size(Xs);
S = zeros(N,N);
switch method
    case 'exp'%p-nearest neighbor graph
        %有标记样本Xs1
        labelindice = find(featureLabel~=0);
        S1 = zeros(N,N);
        Xs1 = Xs(:,labelindice);
        Label = featureLabel(labelindice);
        m = size(Xs1,2);
        for i = 1:m
            for j = i:m
                if Label(j) == Label(i)
                    S1(i,j) = 1;
                end
            end
        end
        S1 = max(S1,S1');   %时刻保持对称
        S1 = S1 - diag(diag(S1)); % 近邻不包括自己
        %无标记样本Xs2
        k = option.k;
        Xs2 = Xs(:,featureLabel==0);
        [neighborIds,neighborDistances,~] = kNearestNeighbors(Xs',Xs2',k+1);
        % sigma = mean(mean(Dist));%依据CaiDeng选取核函数的sigma
        sigma = mean(mean(neighborDistances(:,2:k+1)));%只用选择k近邻后的距离求sigma
        neighborDistances = exp(-neighborDistances/(2*sigma^2));
        for i= m+1:N
            S(i,neighborIds(i-m,2:k+1)) = neighborDistances(i-m,2:k+1);%从第二列开始取，去掉自身的距离
        end
        S = S+S';
        S = S1+S;
    case 'exp slow'%p-nearest neighbor graph
        k = option.k;
        for i=1:N
            [sortval,sortpos] = pdist2(Xs',Xs(:,i)','euclidean','Smallest',k);
            sigma = mean(mean(sortval));%取k近邻的距离近似求sigma
            S(sortpos,i) = exp(-sortval.^2/(2*sigma^2));
        end
    case 'cos'
        k = option.k;
        [neighborIds,~,~] = kNearestNeighbors(Xs',Xs',k+1);
        neighborDistances = squareform(1-pdist(Xs','cosine'));
        for i=1:N
            S(i,neighborIds(i,2:k+1)) = neighborDistances(i,2:k+1);%从第二列开始取，去掉自身的距离
        end
    case 'dec_exp' %update similarity matrix for decremental method
        Xs2 = Xs(:,option.decr_ind); % 1xN_dec
        N_dec = size(option.decr_ind,2);
        k = option.k;
        [neighborIds,neighborDistances,~] = kNearestNeighbors(Xs',Xs2',k+1);
        % sigma = mean(mean(Dist));%依据CaiDeng选取核函数的sigma
        sigma = mean(mean(neighborDistances(:,2:k+1)));%只用选择k近邻后的距离求sigma
        neighborDistances = exp(-neighborDistances/(2*sigma^2));
        for i= 1:N_dec
            S(option.decr_ind(i,:),neighborIds(i,2:k+1)) = neighborDistances(i,2:k+1);%从第二列开始取，去掉自身的距离
        end
        S = S+S';
        %         S = max(S,S');
    case 'knn'%p-nearest neighbor graph
        %all data Xs
        k = option.k;
        [neighborIds,neighborDistances,~] = kNearestNeighbors(Xs',Xs',k+1);
        % sigma = mean(mean(Dist));%依据CaiDeng选取核函数的sigma
        sigma = mean(mean(neighborDistances(:,2:k+1)));%只用选择k近邻后的距离求sigma
        neighborDistances = exp(-neighborDistances/(2*sigma^2));
        for i= 1:N
            S(i,neighborIds(i,2:k+1)) = neighborDistances(i,2:k+1);%从第二列开始取，去掉自身的距离
        end
        S = S+S';
    case 'exp_knn'%p-nearest neighbor graph
        %有标记样本Xs1
        labelindice = find(featureLabel~=0);
        S1 = zeros(N,N);
        Xs1 = Xs(:,labelindice);
        Label = featureLabel(labelindice);
        m = size(Xs1,2);
        for i = 1:m
            for j = i:m
                if Label(j) == Label(i)
                    S1(i,j) = 1;
                end
            end
        end
        S1 = max(S1,S1');   %时刻保持对称
        S1 = S1 - diag(diag(S1)); % 近邻不包括自己
        %无标记样本Xs2
        k = option.k;
        [neighborIds,neighborDistances,~] = kNearestNeighbors(Xs',Xs',k+1);
        % sigma = mean(mean(Dist));%依据CaiDeng选取核函数的sigma
        sigma = mean(mean(neighborDistances(:,2:k+1)));%只用选择k近邻后的距离求sigma
        neighborDistances = exp(-neighborDistances/(2*sigma^2));
        for i= 1:N
            S(i,neighborIds(i,2:k+1)) = neighborDistances(i,2:k+1);%从第二列开始取，去掉自身的距离
        end
        S = S+S';
        S = S1+S;
end
S = (S+S')/2;%保证矩阵的对称性；
L = diag(sum(S,2)) - S;%计算Laplacian矩阵,D取行和
