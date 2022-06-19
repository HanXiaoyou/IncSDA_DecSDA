%% %----------------------------参数的设�?-------------------------------%
% DataBase = 'VIS128x128'; train_num = 2; DataBaseIndex = 'VIS';
% DataBase = 'NIR128x128'; train_num = 16; DataBaseIndex = 'NIR';
% DataBase = 'YaleB100x100'; train_num = 51; DataBaseIndex = 'YaleB';
DataBase = 'YouTubeFace_320x320'; train_num = 72; DataBaseIndex = 'YouTubeFace';
% DataBase = 'YaleB32x32'; train_num = 51; DataBaseIndex = 'YaleB'; %只用于调�?
type = 'Normalize';
incunlabel = 0.6;%增量过程无标记样本所占比�?
%% %-----------------------------导入数据--------------------------------%
[fea,gnd]=DataProcess(DataBase,type);
for group = 1:1
   fprintf(1,strcat('The',num2str(group),' time of experiment ','\n'));
%-----------------------------导出索引数据--------------------------------%
%     % --只有YouTubeFace数据集才�?�?%***%代码
%     DataBase = 'YouTubeFace'; train_num = 72;
%     j = num2str(group);
%     databasegroup = [DataBase,j];
%     type = 'Normalize';
%     incunlabel = 0.6;
%     [fea,gnd]=DataProcess(databasegroup,type);
%     % --
    eval(['load '  'DataBase_Index\',DataBaseIndex '\' int2str(train_num) 'Train\'  int2str(group) '.mat']);   %
    X_train = (fea(trainIdx,:))';  %D×N 
    Y_train = (gnd(trainIdx,:))';  %D×1
    X_test = (fea(testIdx,:))';    %D×N
    Y_test = (gnd(testIdx,:))';    %D×1
   %% % ISSDA
   %训练部分
    X_Unlable = X_train(:,init_unlabel);
    X_Train_L = X_train(:,init_ind);
    Y_Train_L = Y_train(:,init_ind);
    Ninit = size(init_ind,1);
    photo = size(incr_ind,2);
    incr_ind(:,1:floor(photo*incunlabel))=[];
    Nincr = size(incr_ind,2);
    nupdate = size(incr_ind,1);
    for i = 1:nupdate
        X_Train_L = [X_Train_L,X_train(:,incr_ind(i,:))];
        Y_Train_L = [Y_Train_L,Y_train(:,incr_ind(i,:))];
    end
    option.n = nupdate; option.DataBase = DataBase;
    [result,t_ISSDA] = ISSDA(X_Unlable,X_Train_L,Y_Train_L,Ninit,Nincr,option);
    %测试部分(NNCD)
    accuracy_ISSDA = zeros(nupdate+1,1);
    method = 'NNCD';
    model.X_Test = X_test;model.Y_Test = Y_test;
    model.ClassCenter = result.ClassCenter;
    model.ClassLabel = union([],Y_train);
    for i = 1:nupdate+1
        W = result.V(:,result.index(i,1):result.index(i,2));
        option.nupdate = i-1;
        accuracy_ISSDA(i,1) = TestEval(W,model,method);
    end
    %储存结果
    alg = 'ISSDA_LU';
    file = strcat(DataBase,alg,'_',num2str(group),'.mat');
    save(['RepeatResult\', file],'accuracy_ISSDA','t_ISSDA','-v7.3');
end