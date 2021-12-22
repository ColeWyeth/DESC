%% Authors: Cole Wyeth, Yunpeng Shi
%% 
%%------------------------------------------------
%% Input Parameters: 
%% Ind: edge_num by 2 "edge indices matrix". Each row is the index of an edge (i,j). that is sorted as (1,2), (1,3), (1,4),... (2,3), (2,4),.... 
%% edge_num is the number of edges.
%% RijMat: 3 by 3 by edge_num tensor that stores the given relative rotations corresponding to Ind



%% Output:
%% R_est: Estimated rotations (3x3xn)

function Rest = linprog_sij(Ind, RijMat, params)

    nsample = params.nsample; 
    
    % building the graph   
    Ind_i = Ind(:,1);
    Ind_j = Ind(:,2);
    n=max(Ind,[],'all');
    m=size(Ind_i,1); % number of edges
    AdjMat = sparse(Ind_i,Ind_j,1,n,n); % Adjacency matrix
    AdjMat = full(AdjMat + AdjMat');
    
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


    % CoIndMat(:,l)= triangles sampled that contains l-th edge
    % e.g. l=(3,5), then CoIndMat(:,l)=[2,9,8,...] means that...
    % triangles 352, 359, 358,... are sampled
    for l = IndPos
        i = Ind_i(l); j = Ind_j(l);
       CoIndMat(:,l)= datasample(find(AdjMat(:,i).*AdjMat(:,j)), nsample);
    end

    RijMat4d = zeros(3,3,n,n);
    % store pairwise directions in 3 by n by n tensor
    % construct edge index matrix (for 2d-to-1d index conversion)
    for l = 1:m
        i=Ind_i(l);j=Ind_j(l);
        RijMat4d(:,:,i,j)=RijMat(:,:,l);
        RijMat4d(:,:,j,i)=(RijMat(:,:,l))';
        IndMat(i,j)=l;
        IndMat(j,i)=l;
    end
    
    % Now compute S0Mat
    Rki0 = zeros(3,3,m,nsample);
    Rjk0 = zeros(3,3,m,nsample);
    for l = IndPos
        Rki0(:,:,l,:) = RijMat4d(:,:,CoIndMat(:,l), Ind_i(l));
        Rjk0(:,:,l,:) = RijMat4d(:,:,Ind_j(l),CoIndMat(:,l));
    end
    Rki0Mat = reshape(Rki0,[3,3,m*nsample]);
    Rjk0Mat = reshape(Rjk0,[3,3,m*nsample]);
    Rij0Mat = reshape(kron(ones(1,nsample),reshape(RijMat,[3,3*m])), [3,3,m*nsample]);
    R_cycle0 = zeros(3,3,m*nsample);
    R_cycle = zeros(3,3,m*nsample);
    for j = 1:3
      R_cycle0 = R_cycle0 + bsxfun(@times,Rij0Mat(:,j,:),Rjk0Mat(j,:,:));
    end
    for j = 1:3
      R_cycle = R_cycle + bsxfun(@times,R_cycle0(:,j,:),Rki0Mat(j,:,:));
    end
    R_trace = (reshape(R_cycle(1,1,:)+R_cycle(2,2,:)+R_cycle(3,3,:), [m,nsample]))';
    S0Mat = abs(acos((R_trace-1)./2))/pi;
    
    A = sparse(nsample*2*m, m);
    b = zeros(1, nsample*2*m);

    for l = 1:m
        i = Ind_i(l);
        j = Ind_j(l);
        if mod(l, 1000) == 0
            disp('next 1000 done');
        end
        offset = nsample*2*(l-1); % previously filled
        for kInd = 1:nsample
            % Then CoIndMat(kInd,l) is the vertex k
            k = CoIndMat(kInd, l);
            A(offset + 2*kInd - 1, l) = 1;
            A(offset + 2*kInd -1, IndMat(i,k)) = -1;
            A(offset + 2*kInd - 1, IndMat(j,k)) = -1;
            b(offset + 2*kInd - 1) = S0Mat(kInd, l); %possibly reversed

            A(offset + 2*kInd, l) = -1; 
            A(offset + 2*kInd, IndMat(i,k)) = -1;
            A(offset + 2*kInd, IndMat(j,k)) = -1;
            b(offset + 2*kInd) = -S0Mat(kInd, l); %possibly reversed
        end
    end

    f = ones(1,m);
    lb = zeros(1,m);
    ub = ones(1,m); 

    SVec = linprog(f, A, b, [],[], lb, ub)
    R_est = GCW(Ind, AdjMat, RijMat, SVec);
    
end