function [ embedding, embedding_vectors, pc_vectors, metrics ] = calculateEmbedding(data, options)
% calculateEmbedding: For supervised NPE embedding of data into a
% low-dimension manifold

%   Data is n-subject x m-feature matrix
%   Need to supply *at least* an attributes file (for supervised nearest
%   neighbours) and class labels (for defining within/without class
%   subjects)
%
%        e.g: options.attributes='age'
%        e.g: options.class = 'sex'        
%           
%       can also supply site labels if required with options.gnd
%
%       can supply other options for number of neighbours, required dimensions etc 
%           see supervisedNPE.m and LGE.m for details (or use defaults)
%
%   To embed new data:  demean columns
%                       data*pc_vectors
%                       standardise columns to std=1
%                       pc_data*embedding_vectors

addpath(genpath('/Users/Gareth/PROJECTS/MCRI/EMBEDDING/NPE'));

if (~exist('options','var'))
   error('need options...');
end
if ~isfield(options,'class')
    error('need subject class labels...');
end
if ~isfield(options,'attributes')
    error('need subject attributes...');
end

% default NPE opts
if ~isfield(options,'NeighborMode')
    options.NeighborMode = 'Supervised';
end
if ~isfield(options,'k') 
    options.k = 10;
end
if ~isfield(options,'ReducedDim') 
    options.ReducedDim = 15;
end


%% preprocess data with PCA (automatically estimate dimensionality or set explicitly)
% demean first
mn = mean(data); 
metrics.mean = mn;
dat = double(bsxfun(@minus, data, mn)); % ensure data is double precision for later

% PCA
[u,s,v] = svd(dat, 'econ');
s = diag(s);

% estimate dimensionality via Minka's approach
% k = laplace_pca(dat);  
% otherwise retain up to 95% variance to save time...
k = sum((cumsum(s.^2)/sum(s.^2))<.95);

%project to whitened PCs
% X_new = X * V / S * sqrt(n_samples)  (equiv. to UxS)
pc_vectors =  (v(:,1:k)/diag(s(1:k))*sqrt(size(dat,1)));
dat = dat*pc_vectors;
% dat is now n components with unit var

%% Run embeddings
[eigvector, eigvalue] = supervisedNPE2(options,dat);

% embed data
embedding_vectors = eigvector;

embedding = dat*eigvector;




