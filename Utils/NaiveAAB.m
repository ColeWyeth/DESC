%% Author: Yunpeng Shi
%% © Regents of the University of Minnesota. All rights reserved.
%% Compute the Naive AAB statistic proposed in [1]

%% Input Parameters: 
%% AdjMat: n by n adjacency matrix

%% GammaMat: 3 by edge_num matrix of pairwise directions. Each column is a pairwise direction. edge_num is the number of edges.
%% The pairwise directions (edges) are ordered by searching row by row for nonzero elements of upper triangle part of AdjMat, 
%% from left to right, top row to bottom row.

%% nsample: number of triangles sampled for each edge. Default value = 50
%% verbose: whether report progress while the algorithm is running. Default value = true

%% Output:
%% NaiveAABVec: the vector of Naive AAB statistics. Each element is the Naive AAB statistic of the corresponding edge

%% Reference
%% [1] Yunpeng Shi and Gilad Lerman. "Estimation of Camera Locations in Highly Corrupted Scenarios: ...
%% All About that Base, No Shape Trouble." CVPR 2018.
function NaiveAABVec = NaiveAAB(AdjMat, GammaMat, nsample, verbose)
    if ~exist('nsample','var')
        nsample = 50;
    end
    if ~exist('verbose','var')
        verbose = true;
    end
    n = size(AdjMat,1); % number of cameras
    m = size(GammaMat,2); % number of edges
    % 2d indices of edges, i<j
    [Ind_j, Ind_i] = find(tril(AdjMat,-1)); 
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
    GammaMat3d = zeros(3,n,n);
    for l = 1:m
        i=Ind_i(l);j=Ind_j(l);
        GammaMat3d(:,j,i)=GammaMat(:,l);
        GammaMat3d(:,i,j)=-GammaMat(:,l);
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
    SAABMat = abs(acos(S.*(X.^2+Y.^2-2*X.*Y.*Z)./(1-Z.^2)+(S-1.0).*bsxfun(@min, X, Y)));
    % Taking average for each column to obtain the Naive AAB for each edge
    NaiveAABVec = mean(SAABMat,1);
    if verbose
        disp('Completed!')
    end
end



