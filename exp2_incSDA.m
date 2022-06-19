%% %----------------------------dataset proposal-------------------------------%
% DataBase = 'ExtendedYaleB_crop_169x192'; train_num = 225; DataBaseIndex = 'ExtendedYaleB_crop_169x192';
% DataBase = 'ExtendedYaleB_crop_169x192'; train_num = 22597; DataBaseIndex = 'ExtendedYaleB_crop_169x192';
% DataBase = 'ExtendedYaleB_crop_100x100'; train_num = 22597; DataBaseIndex = 'ExtendedYaleB_crop_169x192';
% DataBase = 'ExtendedYaleB_crop_100x100'; train_num = 300125; DataBaseIndex = 'ExtendedYaleB_crop_169x192';
% DataBase = 'ExtendedYaleB_crop_169x192'; train_num = 300125; DataBaseIndex = 'ExtendedYaleB_crop_169x192';
% DataBase = 'ExtendedYaleB_crop_100x100'; train_num = 262112; DataBaseIndex = 'ExtendedYaleB_crop_169x192';
DataBase = 'NIR128x128'; train_num = 16; DataBaseIndex = 'NIR';
% DataBase = 'YaleB100x100'; train_num = 51; DataBaseIndex = 'YaleB';
type = 'Normalize';
incunlabel = 1;
%% %-----------------------------load dataset--------------------------------%
[fea,gnd]=DataProcess(DataBase,type);
%%
% DataBase made from MainCreatTrainTest.m
for group = 1:1
    fprintf(1,strcat('第',num2str(group),'次实验','\n'));
    %-----------------------------get data--------------------------------%
    eval(['load '  'DataBase_Index\',DataBaseIndex '\' int2str(train_num) 'Train\'  int2str(group) '.mat']);   %
    X_train = (fea(trainIdx,:))';  %DxN
    Y_train = (gnd(trainIdx,:))';  %Dx1
    X_test = (fea(testIdx,:))';    %DxN
    Y_test = (gnd(testIdx,:))';    %Dx1
    %     clear fea gnd
    %-----------------------------incSDA train-----------------------------%
    Y_LU_init = zeros(1,size([init_ind;init_unlabel],1));
    data2.X_init = [X_train(:,init_ind),X_train(:,init_unlabel)];%[XL,XU]
    Y_LU_init(1,1:size(init_ind,1)) = Y_train(init_ind);
    data2.Y_LU_init = Y_LU_init;
    
    nupdate = size(incr_ind,1);  photo = size(incr_ind,2);
    data2.X_incr = [];
    data2.Y_LU_incr = zeros(nupdate,size(incr_ind,2));
    for n = 1:nupdate
        data2.X_incr = [data2.X_incr,X_train(:,incr_ind(n,:))];
        data2.Y_LU_incr(n,:) = Y_train(1,incr_ind(n,:));
        data2.Y_LU_incr(n,1:floor(photo*incunlabel)) = 0;
    end
    option2.n = nupdate; option2.p = photo; option2.beta1=1e-6; option2.beta2=1e-3;
    eigenThreshold = 0.01;
    [result,t_incSDAIU] = incSDAIncUnlabel_eig(data2,option2,eigenThreshold);
    %-----------------------------incSDA test-----------------------------%
    %use method of Cai Deng's NNCD
    accuracy_incSDAIU = zeros(nupdate+1,1);
    method = 'NNCD';
    model.X_Test = X_test;model.Y_Test = Y_test;
%     model.ClassCenter = result.ClassCenter;
    X_init = X_train(:,init_ind);
    Y_init = Y_train(:,init_ind);
    Class_Number = size(union([],gnd),1);
    for i = 1:Class_Number
        ClassCenter(:,i) = mean(X_init(:,find(Y_init==i)),2);
    end
    model.ClassCenter = ClassCenter;
    model.ClassLabel = union([],Y_train);
    
    for i = 1:nupdate+1
        W = result.V(:,result.index(i,1):result.index(i,2));
        option.nupdate = i-1;
        [accuracy_incSDAIU(i,1),~,predictlabel(i,:)] = TestEval(W,model,method);
    end
    %save results
    alg = 'incSDA';
    file = strcat(DataBase,alg,'_',num2str(group),'.mat');
    save(['RepeatResult\', file],'accuracy_incSDAIU','t_incSDAIU','predictlabel','-v7.3');
    clearvars -except fea gnd type DataBaseName train_num DataBaseIndex DataBase incunlabel
end
