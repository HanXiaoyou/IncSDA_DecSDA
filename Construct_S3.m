function [S, Sw, Sb] = Construct_S3(wholeSet,data,options)
% Construct_S3  ����ͼG��G',ͬ���������ڽӾ���Sw,Sb�Լ����ƾ���S
% Input:
%       data        ------ѵ��������,һ�д���һ������
%       options     ------����ѵ�����������Ϣoptions.gnd = trainLabel
%          gnd      ------����ǩ
%          lag      ------�ڽӾ���Ԫ�����ͣ� 1---EXP���ͣ�0----0,1����(Ĭ��EXP����)

%          k1       ------ͬ����ڸ���,����ÿһ���б�������ĸ���                        
%          k2       ------������ڸ���,�������б������������ÿһ��ı�����������Ĳ�   
%       type        ------���ڵ�ѡ������
%             'KNN'   ----K����,Ĭ������Ϊ'KNN'
%             'ENN'   ----E����,Բ����
% Output:
%       S            ------���������ƾ���
%       Sw           ------ͬ��ͼG���ڽӾ���
%       Sb           ------����ͼG'���ڽӾ���

% 2017/7/11 ���� 4 

if ~isfield(options,'type')
    options.type = 'KNN';
end

if ~isfield(options,'lag')
    options.lag = 1;
end

[m, ~] = size(data);
Label = options.gnd;
% �������Data�� "��" ֮������Ҿ���
S = squareform(1-pdist(wholeSet,'cosine')); 
Sw = zeros(m);
Sb = zeros(m);
if options.lag %EXP����
    switch lower(options.type) 
        case 'knn'
        %-------------------����K���ڹ���G,G',��S��Sp----------------------%
        % K����,k1ͬ��,k2����,Ĭ��Ϊ5,10
            if ~isfield(options,'k1')
               options.k1 = options.train_No;  %����ÿһ���б�������ĸ���
            end 
            
            if ~isfield(options,'k2')
                options.k2 = m-options.train_No;  %�������б������������ÿһ��ı�����������Ĳ�
            end
            k1 = options.k1;  k2 = options.k2;
        % ��ֵ
            for i = 1:m
                for j = i:m
                    if Label(j) == Label(i)
                        Sw(i,j) = S(i,j);
                    else
                        Sb(i,j) = 1;
                    end
                end
            end
        % S,Sp�Գ�
            Sw = max(Sw,Sw');   Sb = max(Sb,Sb'); %ʱ�̱��ֶԳ�
            Sw = Sw - diag(diag(Sw)); % ���ڲ������Լ�
            Sb = Sb - diag(diag(Sb));
        %-------------------KNN----G,G'�������---------------------------%
        

        case 'enn' 
            %--------------------����Բ���򹹽�G,G',��Sw��Sb--------------------%
            % E����,e1ͬ��,e2����
            e1 = 100; e2 = 90; % e1��e2��ѡȡ�ܴ�̶ȵ�Ӱ������ʶ����,��Ҫ������֤�ˡ�
            Adj_Sw = zeros(m); % G���ڽӾ���
            Adj_Sb = zeros(m);% G'���ڽӾ���
            for i = 1:m
                for j = 1:m
                    if Label(j) == Label(i) % ͬ�����ݣ����������ݣ�ͬ��֮��Ľ���
                        if S(i,j) < e1
                            Adj_Sw(i,j) = 1;
                            Sw(i,j) = S(i,j);
                        end
                    else % ��ͬ�����ݡ���������ݣ�����֮��Ľ���        
                        if S(i,j) < e2
                            Adj_Sb(i,j) = 1;
                            Sb(i,j) = S(i,j);
                        end
                    end % if
                end % for
            end % for

        %-------------------ENN---����G,G'����----------------------------%
        otherwise;
            error('��ѡȡ��ȷ�����ݼ��ط�ʽ��');
    end % switch
    
    
    
else % ����������0~1
    switch lower(options.type)
       case 'knn'
        %-------------------����K���ڹ���G,G',��S��Sp----------------------%
        % K����,k1ͬ��,k2����,Ĭ��Ϊ4,4
             if ~isfield(options,'k1')
                %options.k1 = 5;
                  options.k1 = options.train_No;  %����ÿһ���б�������ĸ���
            end
            if ~isfield(options,'k2')
               % options.k2 = 10;
               options.k2 = m-options.train_No;  %�������б������������ÿһ��ı�����������Ĳ�
            end
            k1 = options.k1;  k2 = options.k2;
        % ��ʱ�ȸ�ֵΪ����
            for i = 1:m
                for j = i:m
                    if Label(j) == Label(i)
                        Sw(i,j) = 1;
                    else
                        Sb(i,j) = 1;
                    end
                end
            end
        % S,Sp�Գ�
            Sw = min(Sw, Sw'); % �����ȡ��Сֵ
            Sb = min(Sb, Sb');
            % K����ѡ��
            [s, idx] = sort(Sw);
            for i=1:m
                Sw(i, idx((2 + k1):end, i)) = inf;
            end
            [s, idx] = sort(Sb);
            for i = 1:m
                Sb(i, idx((2 + k2):end, i)) = inf;
            end
            index = find(Sw>1); Sw(index)=0; clear index;
            index = find(Sb>1); Sb(index)=0; clear index;
  
            Sw = max(Sw,Sw');   Sb = max(Sb,Sb'); %ʱ�̱��ֶԳ�
            Sw = Sw - diag(diag(Sw)); % ���ڲ������Լ�
            Sb = Sb - diag(diag(Sb));
        %-------------------KNN----G,G'�������---------------------------%
        otherwise
            error('��ѡȡ��ȷ�����ݼ��ط�ʽ��');
    end % switch
            
 end % if
end

