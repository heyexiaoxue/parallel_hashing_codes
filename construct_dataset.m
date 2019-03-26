function exp_data = construct_dataset(topNum)
%data constrcution

load 'LabelMe_gist';
db_data = gist; clear gist;

%%
%%db_data��������һ�������
[ndata, D] = size(db_data);
%randperm�������һ����������
R = randperm(ndata);
num_test = 2000;
%ȡ500������������
test_data = db_data(R(1:num_test), :);
% test_ID = R(1:num_test);
R(1: num_test) = [];
train_data = db_data(R, :);%ʣ�µ�9500��ѵ������
% train_ID = R;
num_training = size(train_data, 1);
%%
if topNum == 0
    topNum = 0.02 * size(train_data,1);
end
% topNum = 500;
%DtrueTestTraining��ŵ���test_dataÿ��������train_dataÿ��������ŷʽ�����ƽ��
DtrueTestTraining = distMat(test_data, train_data);
%�������򣬰�ÿһ�������������С������
[~,ind] = sort(DtrueTestTraining,2);
%�ҵ�ǰtopNum��
ind = ind(:,1:topNum);
WtrueTestTraining = zeros(num_test,num_training);

for i=1:num_test
     WtrueTestTraining(i,ind(i,:)) = 1;
end
%WtrueTestTraining��ŵ���һ��0,1����1��ʾ���λ�õ�������ǰ1000������
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
%����ѵ������
exp_data.train_data = train_data(rp(1:train_num),:)';

exp_data.dim = size(test_data,1);
exp_data.ndim = 1;
exp_data.train_num = train_num;
exp_data.groundtruth = WtrueTestTraining;
  
  