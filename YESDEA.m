function [Q,R,Lw,Lb,L,V,time] = YESDEA(data,untrainSet,options,k)

 t0=clock;
 [m,~]=size(data);
 
% wholeSet=[data;untrainSet];  %SDE��Ҫ��ѵ����Ϊһ���ֱ��(data)һ����δ���(testdata)��wholeSet��Ϊѵ����
% wholeSet=wholeSet/norm(wholeSet,'fro');%�����ݽ��г�ʼ��
% data=wholeSet(1:m,:);

[m1,~]=size(untrainSet);


wholeSet=[data;untrainSet];
%-----------------------------�������S,Sw,Sb--------------------------%
[S,Sw, Sb] = Construct_S3(wholeSet,data, options);
D = diag(sum(S));   L = D - S;
Dw = diag(sum(Sw)); Lw = Dw - Sw; 
Db = diag(sum(Sb)); Lb = Db - Sb; 
[Q,R]=qr(wholeSet',0);



%�趨����
r=10^(-6);
yA=1;
yB=10;

%-----------------------------�������S,Sw,Sb--------------------------%
% [S,Sw, Sb] = Construct_S3(wholeSet,data, options);
% D = diag(sum(S));   L = D - S;
% Dw = diag(sum(Sw)); Lw = Dw - Sw; 
% Db = diag(sum(Sb)); Lb = Db - Sb; 

% ��SDE���м��ٸĽ� �� (I+R*WA*R')*(Q'x)=a*(I+R*WB*R')*(Q'x) ����ֵ����
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

%----------------------------ESDE���Ż�����-------------------------%
[V ,E] = eig(A,B);
V = Q*V;
[~, idx] = sort( diag(E), 'descend');%ȡ����d������ֵ������������
V = V(:,idx);

%%%%%%

V= V(:,1:k-1);
V = V * diag(1./(sum(V.^2).^0.5));%�������������й�һ��
V=orth(V);
time=etime(clock,t0);

end