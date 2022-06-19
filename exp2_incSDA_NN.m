%% %----------------------------dataset proposal-------------------------------%
% DataBase = 'ExtendedYaleB_crop_169x192'; train_num = 225; DataBaseIndex = 'ExtendedYaleB_crop_169x192';
% DataBase = 'ExtendedYaleB_crop_169x192'; train_num = 22597; DataBaseIndex = 'ExtendedYaleB_crop_169x192';
% DataBase = 'ExtendedYaleB_crop_100x100'; train_num = 22597; DataBaseIndex = 'ExtendedYaleB_crop_169x192';
% DataBase = 'ExtendedYaleB_crop_100x100'; train_num = 300125; DataBaseIndex = 'ExtendedYaleB_crop_169x192';
DataBase = 'ExtendedYaleB_crop_100x100'; train_num = 262112; DataBaseIndex = 'ExtendedYaleB_crop_169x192';
%  behave bad
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
%     eigenThreshold = 0.01;
    eigenThreshold = 0.001;
    [result,t_incSDAIU] = incSDAIncUnlabel_eig(data2,option2,eigenThreshold);
    %-----------------------------incSDA test-----------------------------%
    %use method of NN
    accuracy_incSDAIU = zeros(nupdate+1,1)
    trainLabel = Y_train(:,init_ind)';
    testLabel = Y_test';
    for i = 1:nupdate+1      
        eigvector = result.V(:,result.index(i,1):result.index(i,2));
        low=1;  high = size(eigvector,2);
        accuracy = zeros(1, high); %
        testLabel = Y_test';
        for j = low:high
            train = X_train(:,init_ind)' * eigvector(:,1:j);
            test = X_test' * eigvector(:,1:j);
            [accuracy(j),~,sub_predictlabel(j,:)] = NN(train,test,trainLabel,testLabel);%nearest neighbour鍒嗙被鍣紝寮犵珛瀹跺笀鍏勭敤鐨凪atlab鑷甫鍒嗙被鍣╟laasify
            fprintf(1,strcat('第',num2str(i),'次实验的','accuracy_12 = ',num2str(accuracy(j)),'\n'));
        end
        [accuracy_incSDAIU(1,n+1),idx_predict] = max(accuracy);
        predictlabel(1,:) = sub_predictlabel(idx_predict,:);
    end
    
    %save results
    alg = 'incSDA_NN';
    file = strcat(DataBase,alg,'_',num2str(group),'.mat');
    save(['RepeatResult\', file],'accuracy_incSDAIU','t_incSDAIU','predictlabel','-v7.3');
    clearvars -except fea gnd type DataBaseName train_num DataBaseIndex DataBase incunlabel
end
