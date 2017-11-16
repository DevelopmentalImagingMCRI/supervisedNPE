function [eigvector, eigvalue] = LGE(W, k, data)
% LGE: Linear Graph Embedding
% for original Matlab code: http://www.cad.zju.edu.cn/home/dengcai/Data/DimensionReduction.html
% Please cite: 
% Xiaofei He, Deng Cai, Shuicheng Yan and Hong-Jiang Zhang, "Neighborhood Preserving Embedding" ICCV 2005. 
% Sam Roweis & Lawrence Saul. "Nonlinear dimensionality reduction by locally linear embedding" Science, 2000, 290:5500, 2323-2326.
%  Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression for Efficient Regularized Subspace Learning" ICCV 2007.

%       [eigvector, eigvalue] = LGE(W, D, options, data)
% 
%             Input:
%               W       - Affinity graph matrix. 
%               k        - The dimensionality of the reduced subspace. 
%               data    - data matrix. Each row vector of data is a
%                         sample vector. 
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           sample vector (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of the eigen-problem.
%                           



%======================================
% setup
%======================================
ReducedDim = k;

[nSmp,nFea] = size(data);

if size(W,1) ~= nSmp
    error('W and data mismatch!');
end

%======================================
% SVD
%======================================
% not neccesary if data already projected to PCs (uncorrelated)
DPrime = data'*data;
DPrime = full(DPrime);
DPrime = max(DPrime,DPrime');
[R,p] = chol(DPrime);
if p==1
    [U, S, V] = svd(data, 'econ');
    data = U;
    eigvalue_PCA = diag(S);
    eigvector_PCA = V*spdiags(eigvalue_PCA.^-1,0,length(eigvalue_PCA),length(eigvalue_PCA));
end

WPrime = data'*W*data;
WPrime = max(WPrime,WPrime');

%======================================
% Generalized Eigen
%======================================

dimMatrix = size(WPrime,2);

if ReducedDim > dimMatrix
    ReducedDim = dimMatrix; 
end

[eigvector, eigvalue] = eig(WPrime,DPrime);

eigvalue = diag(eigvalue);
[junk, index] = sort(-eigvalue);
eigvalue = eigvalue(index);
eigvector = eigvector(:,index);

if ReducedDim < size(eigvector,2)
    eigvector = eigvector(:, 1:ReducedDim);
    eigvalue = eigvalue(1:ReducedDim);
end

if p==1
    eigvector = eigvector_PCA*eigvector;
end

for i = 1:size(eigvector,2)
    eigvector(:,i) = eigvector(:,i)./norm(eigvector(:,i));
end

end

    