function [L] = Construct_S5(wholeSet,indata,L)
%%%����L
S2 = squareform(1-pdist(indata,'cosine')); %%���������ľ������
S1 = 1-pdist2(wholeSet,indata,'cosine'); %%�����ӵ���������֪�����ľ������cos�� 
D1 = diag(sum(S1')); 
D2 = diag(sum([S1',S2]'));
L =[L+D1,-S1;-S1',D2-S2];
end

