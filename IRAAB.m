%% Author: Yunpeng Shi
%% © Regents of the University of Minnesota. All rights reserved.
%% Compute the IR-AAB statistic proposed in [1]

%% Input Parameters: 
%% AdjMat: n by n adjacency matrix

%% GammaMat: 3 by edge_num matrix of pairwise directions. Each column is a pairwise direction. edge_num is the number of edges.
%% The pairwise directions (edges) are ordered by searching row by row for nonzero elements of upper triangle part of AdjMat, 
%% from left to right, top row to bottom row.

%% nsample: number of triangles sampled for each edge. Default value = 50
%% niter: number of total iterations for the reweighting procedure. Default value = 10
%% verbose: whether report progress while the algorithm is running. Default value = true

%% Output:
%% IRAABVec: the vector of IR-AAB statistics. Each element is the IR-AAB statistic of the corresponding edge

%% Reference
%% [1] Yunpeng Shi and Gilad Lerman. "Estimation of Camera Locations in Highly Corrupted Scenarios: ...
%% All About that Base, No Shape Trouble." CVPR 2018

function IRAABVec = IRAAB(AdjMat, GammaMat, nsample, niter, verbose)
    if ~exist('nsample','var')
        nsample = 50;
    end
    if ~exist('niter','var')
        niter = 10;
    end
    if ~exist('verbose','var')
        verbose = true;
    end
    n = size(AdjMat,1); % number of cameras
    m = size(GammaMat,2); % number of edges
    % 2d indices of edges, i<j
    [Ind_j, Ind_i] = find(tril(AdjMat,-1)); 
    GammaMat3d = zeros(3,n,n);
    % Matrix of codegree:
    % CoDeg(i,j) = 0 if i and j are not connected, otherwise,
    % CoDeg(i,j) = # of vertices that are connected to both i and j
    CoDeg = (AdjMat*AdjMat).*AdjMat;
    AdjPos = AdjMat;
    % label positive codegree elements as -1
    AdjPos(CoDeg>0)=-1;
    AdjPosLow = tril(AdjPos);
    AdjPosLow(AdjPosLow==0)=[];
    % find 1d indices of edges with positive codegree
    IndPos = find(AdjPosLow<0);

    if verbose
        disp('Sampling Triangles...')
    end
    % CoIndMat(:,l)= triangles sampled that contains l-th edge
    % e.g. l=(3,5), then CoIndMat(:,l)=[2,9,8,...] means that...
    % triangles 352, 359, 358,... are sampled
    for l = IndPos
        i = Ind_i(l); j = Ind_j(l);
       CoIndMat(:,l)= datasample(find(AdjMat(:,i).*AdjMat(:,j)), nsample);
    end
    if verbose
        disp('Triangle Sampling Finished!')
        disp('Computing Naive AAB ...')
    end
    % store pairwise directions in 3 by n by n tensor
    % construct edge index matrix (for 2d-to-1d index conversion)
    for l = 1:m
        i=Ind_i(l);j=Ind_j(l);
        GammaMat3d(:,j,i)=GammaMat(:,l);
        GammaMat3d(:,i,j)=-GammaMat(:,l);
        IndMat(i,j)=l;
        IndMat(j,i)=l;
    end

    Xki = zeros(3,m,nsample);
    Xjk = zeros(3,m,nsample);
    for l = IndPos
        Xki(:,l,:) = GammaMat3d(:,Ind_i(l),CoIndMat(:,l));
        Xjk(:,l,:) = -GammaMat3d(:,Ind_j(l),CoIndMat(:,l));
    end
    % Xki stores gamma_ki of all triangles ijk
    % Xki has nsample blocks. Each block is 3 by m (m gamma_ki's)
    % i corresponds to edge (i,j), k corresponds to a sampled triangle
    Xki = reshape(Xki,[3,m*nsample]);
    % Xjk stores gamma_jk of all triangles ijk
    % Xjk has nsample blocks. Each block is 3 by m (m gamma_jk's)
    % j corresponds to edge (i,j), k corresponds to a sampled triangle
    Xjk = reshape(Xjk,[3,m*nsample]);
    % Compute Naive AAB statistic using the AAB formula
    % If l-th edge is (i,j), then X(k,l) is the dot product between
    % gamma_ij and gamma_ki. Y and Z are similar
    X = (reshape(sum(Xki.*kron(ones(1,nsample),GammaMat),1),[m,nsample]))';
    Y = (reshape(sum(Xjk.*kron(ones(1,nsample),GammaMat),1),[m,nsample]))';
    Z = (reshape(sum(Xki.*Xjk,1),[m,nsample]))';
    S = 1.0*(X<(Y.*Z)).*(Y<(X.*Z));
    % AAB formula in matrix form
    SAABMat0 = abs(acos(S.*(X.^2+Y.^2-2*X.*Y.*Z)./(1-Z.^2)+(S-1.0).*min(X,Y)));
    % Taking average for each column to obtain the Naive AAB for each edge
    IRAABVec = mean(SAABMat0,1);
    if verbose
        disp('Naive AAB Computed!')
    end
    % compute maximal/minimal AAB inconsistency
    maxAAB = max(max(SAABMat0));
    minAAB = min(min(SAABMat0));
    if verbose
        disp('Reweighting Procedure Started ...')
    end
    for iter = 1:niter
        % parameter controling the decay rate of reweighting function
        tau = pi/(maxAAB-(maxAAB-minAAB)/niter*(iter-1));
        Ski = zeros(nsample, m);
        Sjk = zeros(nsample, m);
        for l = IndPos
            i = Ind_i(l); j=Ind_j(l);
            Ski(:,l) = IRAABVec(IndMat(i,CoIndMat(:,l)));
            Sjk(:,l) = IRAABVec(IndMat(j,CoIndMat(:,l)));
        end
        Smax = max(Ski, Sjk);
        % compute weight matrix (nsample by m)
        WeightMat = exp(-tau*Smax);
        weightsum = sum(WeightMat,1);
        % normalize so that each column sum up to 1
        WeightMat = bsxfun(@rdivide,WeightMat,weightsum);
        SAABMat = WeightMat.*SAABMat0;
        % IR-AAB at current iteration
        IRAABVec = sum(SAABMat,1);
        if verbose
            fprintf('Reweighting Iteration %d Completed!\n',iter)   
        end
    end
    if verbose
        disp('Completed!')
    end
end

