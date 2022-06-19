function [S, Sw, Sb] = Construct_S3(wholeSet,data,options)
% Construct_S3  构造图G和G',同类和异类的邻接矩阵Sw,Sb以及相似矩阵S
% Input:
%       data        ------训练集数据,一行代表一张人脸
%       options     ------包含训练集的类别信息options.gnd = trainLabel
%          gnd      ------类别标签
%          lag      ------邻接矩阵元素类型； 1---EXP类型；0----0,1类型(默认EXP类型)

%          k1       ------同类近邻个数,等于每一类中标记样本的个数                        
%          k2       ------异类近邻个数,等于所有标记样本个数和每一类的标记样本个数的差   
%       type        ------近邻的选择类型
%             'KNN'   ----K近邻,默认类型为'KNN'
%             'ENN'   ----E近邻,圆邻域
% Output:
%       S            ------样本的相似矩阵
%       Sw           ------同类图G的邻接矩阵
%       Sb           ------异类图G'的邻接矩阵

% 2017/7/11 星期 4 

if ~isfield(options,'type')
    options.type = 'KNN';
end

if ~isfield(options,'lag')
    options.lag = 1;
end

[m, ~] = size(data);
Label = options.gnd;
% 计算矩阵Data各 "行" 之间的余弦距离
S = squareform(1-pdist(wholeSet,'cosine')); 
Sw = zeros(m);
Sb = zeros(m);
if options.lag %EXP类型
    switch lower(options.type) 
        case 'knn'
        %-------------------利用K近邻构建G,G',即S，Sp----------------------%
        % K参数,k1同类,k2异类,默认为5,10
            if ~isfield(options,'k1')
               options.k1 = options.train_No;  %等于每一类中标记样本的个数
            end 
            
            if ~isfield(options,'k2')
                options.k2 = m-options.train_No;  %等于所有标记样本个数和每一类的标记样本个数的差
            end
            k1 = options.k1;  k2 = options.k2;
        % 赋值
            for i = 1:m
                for j = i:m
                    if Label(j) == Label(i)
                        Sw(i,j) = S(i,j);
                    else
                        Sb(i,j) = 1;
                    end
                end
            end
        % S,Sp对称
            Sw = max(Sw,Sw');   Sb = max(Sb,Sb'); %时刻保持对称
            Sw = Sw - diag(diag(Sw)); % 近邻不包括自己
            Sb = Sb - diag(diag(Sb));
        %-------------------KNN----G,G'构造结束---------------------------%
        

        case 'enn' 
            %--------------------利用圆邻域构建G,G',即Sw，Sb--------------------%
            % E参数,e1同类,e2异类
            e1 = 100; e2 = 90; % e1和e2的选取很大程度的影响最后的识别率,需要交叉验证了。
            Adj_Sw = zeros(m); % G的邻接矩阵
            Adj_Sb = zeros(m);% G'的邻接矩阵
            for i = 1:m
                for j = 1:m
                    if Label(j) == Label(i) % 同类数据，即类内数据，同类之间的近邻
                        if S(i,j) < e1
                            Adj_Sw(i,j) = 1;
                            Sw(i,j) = S(i,j);
                        end
                    else % 不同类数据。即类间数据，异类之间的近邻        
                        if S(i,j) < e2
                            Adj_Sb(i,j) = 1;
                            Sb(i,j) = S(i,j);
                        end
                    end % if
                end % for
            end % for

        %-------------------ENN---构建G,G'结束----------------------------%
        otherwise;
            error('请选取正确的数据加载方式！');
    end % switch
    
    
    
else % 二进制类型0~1
    switch lower(options.type)
       case 'knn'
        %-------------------利用K近邻构建G,G',即S，Sp----------------------%
        % K参数,k1同类,k2异类,默认为4,4
             if ~isfield(options,'k1')
                %options.k1 = 5;
                  options.k1 = options.train_No;  %等于每一类中标记样本的个数
            end
            if ~isfield(options,'k2')
               % options.k2 = 10;
               options.k2 = m-options.train_No;  %等于所有标记样本个数和每一类的标记样本个数的差
            end
            k1 = options.k1;  k2 = options.k2;
        % 暂时先赋值为距离
            for i = 1:m
                for j = i:m
                    if Label(j) == Label(i)
                        Sw(i,j) = 1;
                    else
                        Sb(i,j) = 1;
                    end
                end
            end
        % S,Sp对称
            Sw = min(Sw, Sw'); % 无穷大取最小值
            Sb = min(Sb, Sb');
            % K近邻选择
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
  
            Sw = max(Sw,Sw');   Sb = max(Sb,Sb'); %时刻保持对称
            Sw = Sw - diag(diag(Sw)); % 近邻不包括自己
            Sb = Sb - diag(diag(Sb));
        %-------------------KNN----G,G'构造结束---------------------------%
        otherwise
            error('请选取正确的数据加载方式！');
    end % switch
            
 end % if
end

