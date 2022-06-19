function [result,t] = IPCA(data,option)
% Input:
%     data:
%           X_init     ------初始训练样本集矩阵D×N
%           X_incr     ------增量训练样本集矩阵D×(n×Ni)
%     option:
%             n                   ------增量次数
%             p                   ------每次增量样本个数
% Output:
%       result:
%               V            ----降维向量D×M(M为降低后的维数)
%               index        ----原始和增量部分降维向量的索引
%               accuracy     ----精确度
%        t      ----运算时间(n+1)×1
%incremental
X_init = data.X_init; X_incr = data.X_incr; 
for nupdate=0:option.n
    % init proc
    if(nupdate==0)
        %-------------------------------PCA & Test---------------------------------
        X1 = X_init;%DxN
        tic;
        %中心化数据
        Xm = mean(X1'); %1xD                                                           %X1转置后每行代表一张图片，按列求平均,返回一个行向量1*10304
        for j = 1:size(X1,2)
            Xcter(:,j) = X1(:,j)-Xm';
        end
        X1=Xcter;                                                                  %此时，X1为中心化的样本数据
        % The principal component coefficients are the eigenvectors of
        % S = X1'*X1./(n-1), but computed using SVD.
        [U0,namda,V] = svd(X1); % put in 1/sqrt(n-1) later
        dnamda = diag(namda);
        %以下选择90%的能量--------------
        dsum = 0;
        for i=1:size(X1,2)
            dsum = dsum + dnamda(i)^2;
        end
        dsum_extract=0;
        k=0;
        while(dsum_extract/dsum<0.95)
            dsum_extract = 0;
            k=k+1;
            for i=1:k
                dsum_extract = dsum_extract + dnamda(i)^2;
            end
        end
        %选择k个主分量------------------
        for i=1:k
            U(:,i) = U0(:,i);
        end
        Y = U'*X1;%Y:kxN  U':kxD  X1:DxN
        t(nupdate+1,1)=toc;%记录运算时间
        result.V = U;
        result.index(nupdate+1,:) = [1,size(U,2)];
        %-------------------------------PCA---------------------------------
    else
        %-------------------------------IPCA---------------------------------------
        % for new data
        p = option.p;
        X2 = X_incr(:,(nupdate-1)*p+1:nupdate*p);
        %中心化数据------------------------
        tic        
        Xm = mean(X2');                                                            %X2转置后每行代表一张图片，按列求平均,返回一个行向量1*10304
        for j = 1:size(X2,2)
            Xcter(:,j) = X2(:,j)-Xm';
        end
        X2 = Xcter;
        %此时，X2为中心化的样本数据
        rnamda = zeros(k,k);
        for i=1:k
            rnamda(i,i) = dnamda(i).^(-0.5);
        end
        H1 = U*rnamda;%原文为P^T                                                             %size(H1)=[n k]
        %----------------------------------                                                       %size(S2bar)=[k k]
        A = H1'*X2;
        B = A*X2';
        S2bar = B*H1;
        %----------------------------------
        [P2,namda2] = eig(S2bar); %等同于求解原文的Q3   
%         [P2,namda2,~] = svd(S2bar);
        dnamda2 = diag(namda2);
        P = H1*P2;  
%         P = orth(H1)*P2;
%         P = orth(P);%size(P)=[n k]  找到了S1+S2的特征向量P，即更新后的特征空间。
        X = [X1';X2']';
        Y = P'*X;
        t(nupdate+1,1)=toc;%记录运算时间
        result.V = [result.V,P];
        result.index(nupdate+1,:) = [result.index(nupdate,2)+1,result.index(nupdate,2)+size(P,2)];
               % update variables
        dnamda = dnamda2;
        namda = namda2;
        U = P;
        X1 = [X1,X2];
    end
end


