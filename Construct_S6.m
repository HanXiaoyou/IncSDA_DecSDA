function [L] = Construct_S6(data,untrainSet,indata,L)
%%%更新L
[m1,~]=size(data);
[m2,~]=size(untrainSet);
S22 = squareform(1-pdist(indata,'cosine')); %%增加样本的距离矩阵
S12 = 1-pdist2(data,indata,'cosine'); %%求增加的样本与已知样本的距离（cos） 
S23 = 1-pdist2(indata,untrainSet,'cosine'); %%求增加的样本与已知样本的距离（cos） 

D12 = diag(sum(S12')); 
D22 = diag(sum([S12',S22,S23]'));
D23 = diag(sum(S23)); 

a=L(1:m1,1:m1);
b=L(1:m1,m1+1:m1+m2);
d=L(m1+1:m1+m2,m1+1:m1+m2);

L =[D12+a,-S12,b;-S12',D22-S22,-S23;b',-S23',D23+d];

end
