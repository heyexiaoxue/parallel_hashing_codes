
clc;
clear all;
addpath(genpath('toolbox'));
parpool local;
exp_data = construct_dataset(1000);
dim = size(exp_data.train_data,1);
bits = 128;
ntrain = exp_data.train_num;

%L多个
L = 100;
%L = 150;
maxIter = 300;
beta = 1;
lambda = 0.01;
alpha = 0.1;

%划分空间
npart = 16;
sbits = bits/npart;
sub_Dim = dim/npart;
params = cell(1,npart);
dataMean = cell(1,npart);
R = cell(1,npart);
sub_W=cell(1,npart);
sub_C = cell(1,npart);
sub_codebook=cell(1,npart);
sub_C_ori = cell(1,npart);
tic

parfor m=1:npart
    m
    %平行处理
    temp =1;
    X_sub = exp_data.train_data((m-1)*sub_Dim+1:m*sub_Dim,:)';
    %中心化
    dataMean{m}=mean(X_sub);
    X_sub = bsxfun(@minus,X_sub,dataMean{m});
    %PCA
    [pc,~,latent,~] = princomp(X_sub);
    R{m}= pc(:,1:sbits);
    %k-means
    [I,C] = litekmeans(X_sub,L,'maxIter',50);
    C = C';
    %找权重
    wn = zeros(1,L);
    for i =1:L
        wn(i)=numel(find(I==i))/ntrain;
    end
    %求出相似性
    Diff = EuDist2(C');
    [~,SimMatrix] = ObtainAffinityMatrix(Diff);
    %
    Original_C = C;
    C = C'*R{m};
    dims = size(C,2);
    W = randn(dims,sbits);
    iter=0;
    temp =1;
    while(iter<maxIter)
        iter = iter+1;
        i = mod(iter,L);
        if mod(iter ,L)==0
            i=L;
        end
    count =0;
    %i = unidrnd(L);
    list = SimMatrix(i,:);
    [~,ind] = sort(list,'ascend');
    gradW = zeros(size(W));
    t_land = C(ind,:);
    
    H = tanh(C*W);
    for j = 2:floor(L/2)
        for k = j+1:L
            Hxi = H(1,:);Hxj=H(j,:);Hxk = H(k,:);
            tmp1 = t_land(1,:)'*((1-Hxi.^2).*(Hxj))+t_land(j,:)'*((1-Hxj.^2).*Hxi);
            tmp2 = t_land(1,:)'*((1-Hxi.^2).*(Hxk))+t_land(k,:)'*((1-Hxk.^2).*Hxi);
            Tij = Hxi*(Hxj');
            Tik = Hxi*(Hxk');
            ytmp = 1/(1+exp(-(beta+Tij-Tik)));
            y = ytmp*(1-ytmp);
            if beta+Tij>Tik
                gradW = gradW-0.5*y*(tmp1-tmp2);
                count = count + 1;
            end
        end
    end
    I = eye(size(W,1),size(W,1));
    Wt = (W*W'-I)*W;
    W  = W - alpha*(lambda)*Wt -(1/count)*gradW;
    
    end
    sub_W{m} = W;
        %重新定位中心点
    LC = compactbit(C*W>0);
    C = C';
    codebook = unique(LC,'rows');
    new_C = zeros(sbits,size(codebook,1));
    new_C_ori = zeros(sub_Dim,size(codebook,1));
    for i=1:size(codebook,1)
            %找出相同的code
         [x,~]=find(LC ==codebook(i));
         new_C(:,i) = wn(x)*C(:,x)'/sum(wn(x));
         new_C_ori(:,i) = wn(x)*Original_C (:,x)'/sum(wn(x));
    end
   
    C = single(new_C); 
    codebook = compactbit(C'*W>0);
    sub_C{m} = C;
    sub_C_ori{m} = single(new_C_ori);
    sub_codebook{m} = codebook;
end
for m=1:npart
    param.R = R{m};
    param.C = sub_C{m};
    param.W=sub_W{m};
    param.sbits = sbits;
    param.sub_Dim = sub_Dim;
    param.dataMean = dataMean{m};
    param.codebook = sub_codebook{m};
    param.sub_C_ori = sub_C_ori{m};
    params{m}= param;
end
clear R sub_c sub_W dataMean codebook sub_C_ori
toc
%测试
[B_db]=compressAOH(exp_data.db_data,params,npart);
[B_tst] = compressAOH(exp_data.test_data,params,npart);
D_dist = hammingDist(B_tst,B_db);
[r,p] = recall_precision(exp_data.groundtruth,D_dist);
mAP = area_RP(r,p)
%end
delete(gcp('nocreate'));%close parpool