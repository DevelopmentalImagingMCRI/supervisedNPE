% Script to demonstrate supervised NPE
clearvars
addpath(genpath('NPE'));

% number of subjects?
n = 500;
train = 0.8;

% create age and sex data
idx = randperm(n); age = linspace(1,25, n)';
age = age(idx);
sex = randi(2, [n,1]);
site = randi(5, [n,1]);

% random high-dim data for demonstration
data = randn(n,10000);

% introduce known covariance plus noise
% more noise=worse embedding
demographicData = zscore([age, sex, randn(n,3)]);
covMat=cov(demographicData');
covMat = nearestSPD(covMat);
data = (data'*chol(covMat))';

% set options
options.NeighborMode = 'Supervised';
options.k = 10;             % number of neighbours in graph
options.ReducedDim = 3;     % final number of embedding dimensions

% split into train and test
train_data = data(1:round(train*n), :);

% set subject attributes and class
options.attributes = (age(1:round(train*n)));
options.class = sex(1:round(train*n));

[ embedding, embedding_vectors, pc_vectors, metrics] = calculateEmbedding(train_data, options);

% rotate to maximise separation in first two dimensions
targetMat = zscore([options.attributes, options.class, zeros(size(embedding,1), size(embedding,2)-2)]);
[ rotated_embedding, rotations] = rotatefactors(embedding, 'Method', 'procrustes', 'Type', 'orthogonal', 'Target', targetMat);

% to project new data
test_data = data(round(train*n)+1:end,:);
test_data = bsxfun(@minus, test_data, metrics.mean);

newEmbedding = test_data * pc_vectors * embedding_vectors * rotations;


subplot(3,2,1)
[~,i]=sort(sex);
imagesc(cov(data(i,:)'))
title('covariance matrix (sorted by group)')

subplot(3,2,1)
[~,i]=sort(age);
imagesc(cov(data(i,:)'))
title('covariance matrix (sorted by age)')
            
subplot(3,2,3)
scatter(rotated_embedding(:,1), rotated_embedding(:,2), 10*age(1:round(train*n)), sex(1:round(train*n)), 'filled');
title('training data')

subplot(3,2,4)
scatter(rotated_embedding(:,1), rotated_embedding(:,2), 10*age(1:round(train*n)), age(1:round(train*n)), 'filled');
title('training data - by age')

subplot(3,2,5)
scatter(newEmbedding(:,1), newEmbedding(:,2), 10*age(round(train*n)+1:end), sex(round(train*n)+1:end), 'filled');
title('test data')

subplot(3,2,6)
scatter(newEmbedding(:,1), newEmbedding(:,2), 10*age(round(train*n)+1:end), age(round(train*n)+1:end), 'filled');
title('test data - by age')