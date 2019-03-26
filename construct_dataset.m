function exp_data = construct_dataset(topNum)
%data constrcution

load 'LabelMe_gist';
db_data = gist; clear gist;

%%
%%db_data里面存放着一万个样本
[ndata, D] = size(db_data);
%randperm随机打乱一个数字序列
R = randperm(ndata);
num_test = 2000;
%取500个样本当测试
test_data = db_data(R(1:num_test), :);
% test_ID = R(1:num_test);
R(1: num_test) = [];
train_data = db_data(R, :);%剩下的9500当训练样本
% train_ID = R;
num_training = size(train_data, 1);
%%
if topNum == 0
    topNum = 0.02 * size(train_data,1);
end
% topNum = 500;
%DtrueTestTraining存放的是test_data每个样本到train_data每个样本的欧式距离的平方
DtrueTestTraining = distMat(test_data, train_data);
%按行排序，把每一行里的样本按从小到大排
[~,ind] = sort(DtrueTestTraining,2);
%找到前topNum个
ind = ind(:,1:topNum);
WtrueTestTraining = zeros(num_test,num_training);

for i=1:num_test
     WtrueTestTraining(i,ind(i,:)) = 1;
end
%WtrueTestTraining存放的是一个0,1矩阵，1表示这个位置的数据在前1000个里面
clear DtrueTestTraining;
%%
% generate training ans test split and the data matrix
XX = [train_data; test_data];

% % center the data, VERY IMPORTANT
% sampleMean = mean(XX,1);
% XX = (double(XX)-repmat(sampleMean,size(XX,1),1));
% 
% normalize the data
% XX = normalize1(XX);


exp_data.db_data = XX(1:num_training, :)';
exp_data.test_data = XX(num_training+1:end, :)';
clear XX;
train_num = 20019;
rp = randperm(num_training);
%打乱训练数据
exp_data.train_data = train_data(rp(1:train_num),:)';

exp_data.dim = size(test_data,1);
exp_data.ndim = 1;
exp_data.train_num = train_num;
exp_data.groundtruth = WtrueTestTraining;
  
  