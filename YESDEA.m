function [Q,R,Lw,Lb,L,V,time] = YESDEA(data,untrainSet,options,k)

 t0=clock;
 [m,~]=size(data);
 
% wholeSet=[data;untrainSet];  %SDE中要求训练集为一部分标记(data)一部分未标记(testdata)，wholeSet作为训练集
% wholeSet=wholeSet/norm(wholeSet,'fro');%对数据进行初始化
% data=wholeSet(1:m,:);

[m1,~]=size(untrainSet);


wholeSet=[data;untrainSet];
%-----------------------------构造矩阵S,Sw,Sb--------------------------%
[S,Sw, Sb] = Construct_S3(wholeSet,data, options);
D = diag(sum(S));   L = D - S;
Dw = diag(sum(Sw)); Lw = Dw - Sw; 
Db = diag(sum(Sb)); Lb = Db - Sb; 
[Q,R]=qr(wholeSet',0);



%设定参数
r=10^(-6);
yA=1;
yB=10;

%-----------------------------构造矩阵S,Sw,Sb--------------------------%
% [S,Sw, Sb] = Construct_S3(wholeSet,data, options);
% D = diag(sum(S));   L = D - S;
% Dw = diag(sum(Sw)); Lw = Dw - Sw; 
% Db = diag(sum(Sb)); Lb = Db - Sb; 

% 对SDE进行加速改进 求 (I+R*WA*R')*(Q'x)=a*(I+R*WB*R')*(Q'x) 特征值问题
E=[eye(m);zeros(m1,m)];
TA =yA*( E * Lb * E');
TB =yB*( E * Lw * E' + r * L );
[Q,R]=qr(wholeSet',0);
TA=TA/norm(R*TA*R','fro');
TB=TB/norm(R*TB*R','fro');
wholeSet1 = R' * R;
z1=TA * wholeSet1;
z2=TB * wholeSet1;
zA=phipade(z1 ,1);
zB=phipade(z2 ,1);
n1=size(R,1);
A = speye(n1) + R*(zA*TA)*R';
A=max(A,A');
B = speye(n1) + R*(zB*TB)*R';
B=max(B,B');   

%----------------------------ESDE的优化问题-------------------------%
[V ,E] = eig(A,B);
V = Q*V;
[~, idx] = sort( diag(E), 'descend');%取最大的d个特征值，按降序排列
V = V(:,idx);

%%%%%%

V= V(:,1:k-1);
V = V * diag(1./(sum(V.^2).^0.5));%对特征向量进行归一化
V=orth(V);
time=etime(clock,t0);

end