function [L] = Construct_S5(wholeSet,indata,L)
%%%更新L
S2 = squareform(1-pdist(indata,'cosine')); %%增加样本的距离矩阵
S1 = 1-pdist2(wholeSet,indata,'cosine'); %%求增加的样本与已知样本的距离矩阵（cos） 
D1 = diag(sum(S1')); 
D2 = diag(sum([S1',S2]'));
L =[L+D1,-S1;-S1',D2-S2];
end

