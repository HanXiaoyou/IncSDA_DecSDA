function [V, time] = SDE(data,untrainSet ,options,k)
% �ֲ��б����(semi-supervised Discriminant Embedding, SDE)
% �� A*x = a*B*x ����ֵ����
%     Input:
%          data         ----ѵ����������ÿ�б�ʾһ����������SDE����Ϊ��ǵ����ݼ�;
%          testdata     ----������������ÿ�б�ʾһ����������SDE����Ϊδ��ǵ����ݼ�;
%          options      ----MATLAB�еĽṹ��,���Ĳ�������
%              'gnd'        ----ѵ��������������ǩ�������ṩ;
%     Output:
%          V    ----��������;
%
%%
t0=clock;
[m,~]=size(data);
wholeSet=[data;untrainSet];%SDE��Ҫ��ѵ����Ϊһ���ֱ��(data)һ����δ���(untrainSet)
wholeSet=wholeSet/norm(wholeSet,'fro');%�����ݽ��г�ʼ��
data=wholeSet(1:m,:);

%�趨����
r=10^(-6);
p=10^(-3);
%-----------------------------�������S,Sw,Sb--------------------------%
[S,Sw, Sb] = Construct_S3(wholeSet,data, options);
D = diag(sum(S));   L = D - S;
Dw = diag(sum(Sw)); Lw = Dw - Sw; 
Db = diag(sum(Sb)); Lb = Db - Sb; 

% Y v = eXv ����ֵ����
X = data' * Lw * data; X = max(X, X'); % �Գƾ���
Y = data' * Lb * data; Y = max(Y, Y');      
X1 = wholeSet' * L * wholeSet; X1 = max(X1, X1');
A = Y; 
n = size(X,1);
B = X + r*X1 + p*speye(n);

%----------------------------RSDE���Ż�����-------------------------%
[V ,E] = eig(A,B);
[~, idx] = sort( diag(E), 'descend');%ȡ����k-1������ֵ
V = V(:,idx);

%%%%

V= V(:,1:k-1);
V = V * diag(1./(sum(V.^2).^0.5));%�������������й�һ��
V=orth(V);
time=etime(clock,t0);

end

