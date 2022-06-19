function  [V,time,QS2,RS2,LwS2,LbS2,LS2]= ESDEI_LU(t,trainSetL1,trainSetU1,updataSet,options,k,QS2,RS2,LwS2,LbS2,LS2);
if (mod(t,2)==1)
    [V,time,QS2,RS2,LwS2,LbS2,LS2]= ESDEI_L(trainSetL1,trainSetU1,updataSet,options,k,QS2,RS2,LwS2,LbS2,LS2);
else
    [V,time,QS2,RS2,LwS2,LbS2,LS2]= ESDEI_U(trainSetL1,trainSetU1,updataSet,options,k,QS2,RS2,LwS2,LbS2,LS2);
    
end
end