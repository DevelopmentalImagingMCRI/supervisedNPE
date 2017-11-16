function [eigvector, eigvalue] = supervisedNPE2(options, data)
% NPE: Neighborhood Preserving Embedding
% for original code: http://www.cad.zju.edu.cn/home/dengcai/Data/DimensionReduction.html
% Please cite: 
% Xiaofei He, Deng Cai, Shuicheng Yan and Hong-Jiang Zhang, "Neighborhood Preserving Embedding," ICCV 2005. 

% edited to support supervised NPE with node attributes and classes
% Gareth Ball, Chris Adamson, Richard Beare, Marc L Seal, "Modelling Neuroanatomical Variation Due To Age And Sex During Childhood And Adolescence"
% http://biorxiv.org/content/early/2017/04/11/126441


%  [eigvector, eigvalue] = NPE(options, data)
% 
%             Input:
%               data    - Data matrix. Each row vector of data is a data point.
%
%                                                                            
%               k           -   The number of neighbors.
%                               Default k = 5;
%               class       -   class labels for separation (neighbours are drawn from within the same class)
%               gnd         -   additional/alternative class labels (neighbours are drawn from different classes - optional)
%               attributes  -   Attribute for each subject to inform nearest neighbour search
%
%               Please see LGE.m for other options.
%
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The eigvalue of LPP eigen-problem. sorted from
%                           smallest to largest. 
%
%    Examples:
    
%       fea = rand(50,70);
%       [eigvector, eigvalue] = NPE(options, fea);
%       Y = fea*eigvector;


%% check options
[nSmp,nFea] = size(data);

if ~isfield(options,'k') 
    options.k = 5;
end
if options.k >= nSmp
    error('k is too large!');
end
if(options.k > nFea)
    tol=1e-3; % regulariser in case constrained fits are ill conditioned
else
    tol=1e-12;
end
if ~isfield(options,'class')
    error('class labels should be provided!');
end
if ~isfield(options,'attributes')
    error('attribute labels should be provided!');
end
if ~isfield(options,'gnd')
    options.gnd = ones(size(options.class));
end
if length(options.class) ~= nSmp
    error('class label and data mismatch!');
end

%% start
Label = unique(options.gnd);
nLabel = length(Label);
classLabel = unique(options.class);
nClassLabel = length(classLabel);
neighborhood = zeros(nSmp,options.k);

for idx=1:nLabel
    for cidx=1:nClassLabel
        % get subjects from same site and same class
        classIdx = find(options.gnd==Label(idx) & options.class==classLabel(cidx));
        if ~isempty(classIdx)
            % get subjects from other sites but same class
            altClassIdx = find(options.gnd~=Label(idx) & options.class==classLabel(cidx));

            % if gnd is not specified, all subject in class considered
            if length(altClassIdx)==0
                altClassIdx=classIdx;
            end

            %% check this matches use of EuDist
            % calculate distance matrix based on subject attribute (e.g age)
            dd1 = pdist2(options.attributes(classIdx), options.attributes(altClassIdx), 'euclidean');

            % calculate distance matrix based on image information
            dd2 = pdist2(data(classIdx,:),data(altClassIdx,:),'euclidean');

            % combine distance matrices - neighbours will be both similar in image and age   
            dd1 = 1-(dd1./max(dd1(:))); 
            dd2 = 1-(dd2./max(dd2(:)));

            % final distance matrix for nearest neighbours
            Distance = 1-(dd1.*dd2);

            [sorted,index] = sort(Distance,2);

            %% Get k nearest neighbours
            if nLabel>1
                neighborhood(classIdx,:) = altClassIdx(index(:,1:options.k)); % subject not included in distance matrix
            else    
                neighborhood(classIdx,:) = altClassIdx(index(:,2:(1+options.k))); % as subject is included in distance matrix
            end
        end
    end
end

% calculate weights to reconstruct data point from neighbours
W = zeros(options.k,nSmp);
for ii=1:nSmp
    z = data(neighborhood(ii,:),:)-repmat(data(ii,:),options.k,1); % shift ith pt to origin
    C = z*z';                                                      % local covariance
    C = C + eye(size(C))*tol*trace(C);                             % regularization
    W(:,ii) = C\ones(options.k,1);                                 % solve Cw=1
    W(:,ii) = W(:,ii)/sum(W(:,ii));                                % enforce sum(w)=1
end

% build adjacency matrix based on these weights
M = sparse(1:nSmp,1:nSmp,ones(1,nSmp),nSmp,nSmp,4*options.k*nSmp);
for ii=1:nSmp
    w = W(:,ii);
    jj = neighborhood(ii,:)';
    M(ii,jj) = M(ii,jj) - w';
    M(jj,ii) = M(jj,ii) - w;
    M(jj,jj) = M(jj,jj) + w*w';
end
M = max(M,M');
M = sparse(M);

sampleMean = mean(data);
data = (data - repmat(sampleMean,nSmp,1));


M = -M;
for i=1:size(M,1)
    M(i,i) = M(i,i) + 1;
end

% linear graph embedding
% LGE options - see LGE.m - possible use of regularisation etc
[eigvector, eigvalue] = LGE(M,options.ReducedDim, data);


eigIdx = find(eigvalue < 1e-10);
eigvalue (eigIdx) = [];
eigvector(:,eigIdx) = [];

end
