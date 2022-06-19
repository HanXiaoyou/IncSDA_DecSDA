function [Lw,Lb] = Construct_SIN(data,indata,Lw,Lb, options)
%%%����Lw,Lb,Q,R;
[nSmp, ~] = size(indata);
[m,~]=size(data);
Label = options.gndt;
Labelt= [options.gnd;options.gndt];

S1 = zeros(nSmp);
S2 = zeros(nSmp);
S3 = zeros(m,nSmp);
S4 = zeros(m,nSmp);
% �������Data�� "��" ֮���cos����

Dist = squareform(1-pdist(indata,'cosine'));   %   
   %-------------------����S1,S2---------------------------%
            for i = 1:nSmp
                for j = i:nSmp
                    if Label(j) == Label(i)
                        S1(i,j) = Dist(i,j);
                    else
                        S2(i,j) = 1;
                    end
                end
            end
            S1 = max(S1,S1');   S2 = max(S2,S2'); %ʱ�̱��ֶԳ�
            S1 = S1 - diag(diag(S1)); % ���ڲ������Լ�
            S2 = S2 - diag(diag(S2));   
        %-------------------����S3,S4---------------------------%     
   Dist1 =1-pdist2(data,indata,'cosine');%%�����ӵ���������֪�����ľ��루cos��
             for i = 1:m
                for j = 1+m:nSmp+m
                    if Labelt(j) == Labelt(i)
                        S3(i,j-m) = Dist1(i,j-m);
                    else
                        S4(i,j-m) =1;                       
                    end
                end
             end
%%%          
D1 = diag(sum(S3,2)); 
D2 = diag(sum(S4,2)); 
D3 = diag(sum([S3',S1],2)); 
D4 = diag(sum([S4',S2],2));  

Lw=[Lw+D1,-S3;-S3',D3-S1];
Lb=[Lb+D2,-S4;-S4',D4-S2];  

end
