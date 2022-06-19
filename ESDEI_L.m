function [V, time,Q,R,Lw,Lb,L] = ESDEI_L(data,untrainSet,indata,options,k,Q,R,Lw,Lb,L)
% ESDE的快速增量算法
% 对ESDE进行加速改进 求 (I+RAR')*(Q'x)=a*(I+RBR')*(Q'x) 特征值问题
%     Input:
%          data         ----训练样本集，每行表示一副人脸，在SDE中作为标记的数据集;
%          testdata     ----测试样本集，每行表示一副人脸，在SDE中作为未标记的数据集;
%          options      ----MATLAB中的结构体,它的参数包括;
%              'gnd'        ----训练数据所属类别标签，必须提供;
%                               当大于1时，表示保留的维数，最大不超过原始特征值维数 
%              'k1'         ----同类近邻个数,默认为 5
%              'k2'         ----异类近邻个数,默认为 10
%     Output:
%          V    ----特征向量;
%%
t0=clock;
[m,~]=size(indata);
[m1,~]=size(data);
[m2,~]=size(untrainSet);
options.gnd=options.gndL;

%设定参数
r=10^(-6);
yA=1;
yB=10;

%%
% 对SDE进行加速改进 求 (I+R*WA*R')*(Q'x)=a*(I+R*WB*R')*(Q'x) 特征值问题
%%已知 Q,R,Lw，Lb,L对增量进行运算处理,更新Lw,Lb,L,Q,R
[Lw,Lb] = Construct_SIN(data,indata,Lw,Lb,options); %%更新Lw,Lb
[L] = Construct_S6(data,untrainSet,indata,L);       %%更新L
Q=Q(:,1:m1);R=R(1:m1,1:m1);
q=indata'-Q*(Q'*indata');
[Q1,R1]=qr(q,0);
[~,nr]=size(R);
[mr1,~]=size(R1);
R=[R,Q'*indata';zeros(mr1,nr),R1];
Q=[Q,Q1];  
q=untrainSet'-Q*(Q'*untrainSet');
[Q1,R1]=qr(q,0);
[~,nr]=size(R);
[mr1,~]=size(R1);
R=[R,Q'*untrainSet';zeros(mr1,nr),R1];
Q=[Q,Q1];  
%%%%%%更新Lw,Lb,L,Q,R--------end---------

E=[eye(m+m1);zeros(m2,m+m1)];
TA =yA*( E * Lb * E');
TB =yB*( E * Lw * E' + r * L );
a=norm(R*TA*R','fro');
b=norm(R*TB*R','fro');
TA=TA/a;
TB=TB/b;
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

%----------------------------SDE的优化问题-------------------------%
[V ,E] = eig(A,B);
V = Q*V;
[~, idx] = sort( diag(E), 'descend');  %取最大的d个特征值，按降序排列
V = V(:,idx);

%%%%%%%

V= V(:,1:k-1);
V = V * diag(1./(sum(V.^2).^0.5));%对特征向量进行归一化
V = orth(V);
time=etime(clock,t0);
end
