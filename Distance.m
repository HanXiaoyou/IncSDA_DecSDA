function D = Distance(fea_a,fea_b)
%%%%%函数功能：求两个矩阵fea_a,fea_b的欧式距离
%%%%这里求fea_a的每行到fea_b的每行的距离，即人脸的距离
%%%%D:为距离矩阵，维数fea_a的行*fea_b的行
%%%%距离求错了！！！！
% D=[];
% [aSmp aFea]=size(fea_a);
% [bSmp bFea]=size(fea_b);
% aa=fea_a*fea_a;
% bb=fea_b*fea_b;
% ab=fea_a*fea_b';
% for i=1:aSmp
%     for j=1:bSmp
%        ssum=0;
%        for k=1:aFea
%            for t=1:bSmp;
%               ssum=ssum+(fea_a(j,k)-fea_b(t,k))^2;
%            end
%        end
%        D(i,j)=ssum;
%     end   
% end
%%%%%%%%%%%%%%%%这样求距离不对！！！
%%%%%%%%%%%%%%还要再添一个for才可以！！！
%% 原来用的程序, use L2 norm
[aSmp aFea]=size(fea_a);
[bSmp bFea]=size(fea_b);
aa = sum(fea_a.*fea_a,2);%%%行相加
bb = sum(fea_b.*fea_b,2);%%%行相加
ab = fea_a*fea_b';
aa = full(aa);
bb = full(bb);
ab = full(ab);
D = sqrt(repmat(aa, 1, bSmp) + repmat(bb', aSmp, 1) - 2*ab);
D = abs(D);
%% Use L1-norm
% aSmp=size(fea_a,1);
% bSmp=size(fea_b,1);
% D=zeros(aSmp,bSmp);
% for i=1:aSmp
%     for j=1:bSmp
%         %D(i,j)=norm(fea_a(i,:)-fea_b(j,:),2); % Use L2 norm
%         %
%         %D(i,j)=norm(fea_a(i,:)-fea_b(j,:),1); % Use L1 norm
%         D(i,j)=sum(abs(fea_a(i,:)-fea_b(j,:))); % Use L1 norm
%     end
% end
end