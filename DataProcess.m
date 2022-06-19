function [fea,gnd]=DataProcess(DataBase,type)
% DataProcess.m�Ƕ�ȫ������Ԥ����
% �����LoadFace.m�൱�����ظ�ʵ����ֻ������Ԥ������һ�Σ�ʡʱ��
%��������
%     Input:
%          DataBase     -----ѡ�����ݿ�����ʵ�飬����ѡ��Ϊ'ORL','Yale','YaleB' or 'PIE';
%          train_num    -----ÿ����ѡ������ѵ����ͼƬ����;
%          group        -----���ɵ������ǩ�������ѡȡ��ͬ����������ѵ�����ܹ���50��;
%          type         -----���ݼ��صķ�ʽ������������'Original','Scale','Normalize'.
%              'Original'    ----����ԭʼͼ��Ҷ�ֵ
%              'Scale'       ----���Ҷ�ֵӳ�䵽[0,1]��
%              'Normalize'   ----��ÿ���������й�һ��
%     Output:
%          fea   -----������ȫ���ݼ�������ÿ�д���1������;
%          gnd   -----������ȫ���ݼ���ǩ;
%    Written By PWR
switch DataBase
    case 'CAS_PEAL'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YaleB100x100'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YaleB_32x32'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YaleB32x32'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgn
%     case 'YaleB64x64'
%         eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgn
    case 'YaleB_100x100'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'ExtendedYaleB_crop_169x192'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'ExtendedYaleB_crop_100x100'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'VIS32x32'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'VIS128x128'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'NIR32x32'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'NIR128x128'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'PIE_32x32'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'PIE_486x640_800'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace_320x320'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace1'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace2'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace3'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace4'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace5'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace6'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace7'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace8'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace9'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace10'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
    case 'YouTubeFace_320x320_crop_160x160'
        eval(['load ' 'DataBase\' DataBase '.mat']);%�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
        
end
fea = double(fea);
[nSmp,nFea] = size(fea);    %nSmp:������������ nFea:������ά������ORL_32x32:nSmp=400, nFea=32x32=1024
if (~exist('type','var'))
    type = 'original';
    %     type='normalize';
end
switch lower(type)
    case 'scale'
        maxValue = max(max(fea));                                 %�������ֵ(������������)��������ֵӳ�䵽[0,1]��
        fea = fea/maxValue;
    case 'normalize'
        for i=1:nSmp
            fea(i,:) = fea(i,:)./ max(1e-12,norm(fea(i,:)));     %��ֹ����0������������һ������
        end
    case 'original'
    otherwise
        error('��ѡȡ��ȷ�����ݼ��ط�ʽ��');
end

