function [V, time,Q,R,Lw,Lb,L] = ESDEI_L(data,untrainSet,indata,options,k,Q,R,Lw,Lb,L)
% ESDE�Ŀ��������㷨
% ��ESDE���м��ٸĽ� �� (I+RAR')*(Q'x)=a*(I+RBR')*(Q'x) ����ֵ����
%     Input:
%          data         ----ѵ����������ÿ�б�ʾһ����������SDE����Ϊ��ǵ����ݼ�;
%          testdata     ----������������ÿ�б�ʾһ����������SDE����Ϊδ��ǵ����ݼ�;
%          options      ----MATLAB�еĽṹ��,���Ĳ�������;
%              'gnd'        ----ѵ��������������ǩ�������ṩ;
%                               ������1ʱ����ʾ������ά������󲻳���ԭʼ����ֵά�� 
%              'k1'         ----ͬ����ڸ���,Ĭ��Ϊ 5
%              'k2'         ----������ڸ���,Ĭ��Ϊ 10
%     Output:
%          V    ----��������;
%%
t0=clock;
[m,~]=size(indata);
[m1,~]=size(data);
[m2,~]=size(untrainSet);
options.gnd=options.gndL;

%�趨����
r=10^(-6);
yA=1;
yB=10;

%%
% ��SDE���м��ٸĽ� �� (I+R*WA*R')*(Q'x)=a*(I+R*WB*R')*(Q'x) ����ֵ����
%%��֪ Q,R,Lw��Lb,L�������������㴦��,����Lw,Lb,L,Q,R
[Lw,Lb] = Construct_SIN(data,indata,Lw,Lb,options); %%����Lw,Lb
[L] = Construct_S6(data,untrainSet,indata,L);       %%����L
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
%%%%%%����Lw,Lb,L,Q,R--------end---------

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

%----------------------------SDE���Ż�����-------------------------%
[V ,E] = eig(A,B);
V = Q*V;
[~, idx] = sort( diag(E), 'descend');  %ȡ����d������ֵ������������
V = V(:,idx);

%%%%%%%

V= V(:,1:k-1);
V = V * diag(1./(sum(V.^2).^0.5));%�������������й�һ��
V = orth(V);
time=etime(clock,t0);
end
