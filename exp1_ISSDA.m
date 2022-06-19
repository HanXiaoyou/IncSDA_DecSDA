% DataBaseName = 'VIS128x128';
% DataBaseName = 'VIS32x32';
DataBaseName = 'NIR128x128';
type = 'Normalize';
[fea,gnd]=DataProcess(DataBaseName,type);
for nexp = 1:10
    Nexp = num2str(nexp);
    fprintf(1,strcat('The',Nexp,'times of experiment','\n'));
    eval(['load ' 'DataBase_Index\',num2str(DataBaseName),'\',Nexp '.mat'])
    X_test = (fea(pro_idx(:,6)~=0,:))';    %D×N
    Y_test = (gnd(pro_idx(:,6)~=0,:))';    %1×N
   %% % ISSDA
   %训练部分
    X_Unlable = (fea(pro_idx(:,1)==0,:))'; %D×N
    X_Train_L = (fea(pro_idx(:,2)~=0,:))'; %D×N
    Y_Train_L = (gnd(pro_idx(:,2)~=0,:))'; %1×N
    Ninit = size(Y_Train_L,2);
    Nincr = Ninit;
    nupdate = size(pro_idx,2) - 3;
    Class_Number = size(union([],gnd),1);
    for i = 1:nupdate
        X_Train_L = [X_Train_L,(fea(pro_idx(:,i+1)==1,:))'];
        Y_Train_L = [Y_Train_L,(gnd(pro_idx(:,i+1)==1,:))'];
    end
    option.n = nupdate; option.DataBase = DataBaseName;
    [result,t_ISSDA_O] = ISSDA(X_Unlable,X_Train_L,Y_Train_L,Ninit,Nincr,option);
    %测试部分(NNCD)
    accuracy_ISSDA_O = zeros(nupdate+1,1);
    predictlabel = zeros(nupdate+1,size(X_test,2));
    method = 'NNCD';
    model.X_Test = X_test;model.Y_Test = Y_test;
    model.ClassCenter = result.ClassCenter;
    model.ClassLabel = union([],Y_test);
    for i = 1:nupdate+1
        W = result.V(:,result.index(i,1):result.index(i,2));
        option.nupdate = i-1;
        [accuracy_ISSDA_O(i,1),~,predictlabel(i,:)] = TestEval(W,model,method);
    end
    %储存结果
    alg = 'ISSDA_L';
    file = strcat(num2str(DataBaseName),alg,'_',Nexp,'.mat');
    save(['RepeatResult\', file],'accuracy_ISSDA_O','t_ISSDA_O','predictlabel','-v7.3');
    clearvars -except fea gnd type DataBaseName 
end