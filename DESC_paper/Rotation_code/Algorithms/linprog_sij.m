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
    CoDeg((CoDeg==0)&(AdjMat>0))=-1;
    CoDeg_low = tril(CoDeg,-1);
    CoDeg_vec = CoDeg_low(:);
    CoDeg_vec(CoDeg_vec==0)=[];
    CoDeg_pos_ind = find(CoDeg_vec>0);
    CoDeg_vec_pos = CoDeg_vec(CoDeg_pos_ind);
    % we will use exactly nsample cycles for each edge
    CoDeg_zero_ind = find(CoDeg_vec<0);
    m_pos = length(CoDeg_pos_ind); % number of edges with cycles
    % m_cycle should be replaced by m_pos * nsample

    CoDeg_pos_ind_long = zeros(1,m);
    CoDeg_pos_ind_long(CoDeg_pos_ind) = 1:m_pos;

    Ind_ij = zeros(nsample,m_pos);
    Ind_jk = zeros(nsample,m_pos);
    Ind_ki = zeros(nsample,m_pos);

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
    
    second_term_edges = cell(1, m_pos);
    second_term_cycles = cell(1, m_pos);
    CoIndMat = zeros(nsample, m_pos);
    
    for l = 1:m_pos
        IJ = CoDeg_pos_ind(l); % For each edge with cycles
        i=Ind_i(IJ); j=Ind_j(IJ);
        CoInd_ij= find(AdjMat(:,i).*AdjMat(:,j)); % k for edge i,j
        CoInd_ij = datasample(CoInd_ij, nsample, 'Replace', true);
        for k_ind = 1:nsample
            second_term_edges{CoDeg_pos_ind_long(IndMat(i, CoInd_ij(k_ind)))}(end+1) = l;
            second_term_cycles{CoDeg_pos_ind_long(IndMat(i, CoInd_ij(k_ind)))}(end+1) = k_ind;
            second_term_edges{CoDeg_pos_ind_long(IndMat(j, CoInd_ij(k_ind)))}(end+1) = l;
            second_term_cycles{CoDeg_pos_ind_long(IndMat(j, CoInd_ij(k_ind)))}(end+1) = k_ind;
        end
        CoIndMat(:, l) = CoInd_ij;
        Ind_ij(:, l) =  IJ; % index by cycle, edge, get edge (ij)
        Ind_jk(:, l) =  IndMat(j,CoInd_ij); % index by cycle, get jk
        Ind_ki(:, l) =  IndMat(CoInd_ij,i); % index by cycle, get ik
    end

    Rki0 = zeros(3,3,m_pos,nsample);
    Rjk0 = zeros(3,3,m_pos,nsample);
    for l = 1:m_pos
        Rki0(:,:,l,:) = RijMat4d(:,:,CoIndMat(:,l), Ind_i(l));
        Rjk0(:,:,l,:) = RijMat4d(:,:,Ind_j(l),CoIndMat(:,l));
    end
    Rki0Mat = reshape(Rki0,[3,3,m_pos*nsample]);
    Rjk0Mat = reshape(Rjk0,[3,3,m_pos*nsample]);
    Rij0Mat = reshape(kron(ones(1,nsample),reshape(RijMat(:,:,CoDeg_pos_ind),[3,3*m_pos])), [3,3,m_pos*nsample]);
    
    R_cycle0 = zeros(3,3,m_pos*nsample);
    R_cycle = zeros(3,3,m_pos*nsample);
    
    for j = 1:3
      R_cycle0 = R_cycle0 + bsxfun(@times,Rij0Mat(:,j,:),Rjk0Mat(j,:,:));
    end

    for j = 1:3
      R_cycle = R_cycle + bsxfun(@times,R_cycle0(:,j,:),Rki0Mat(j,:,:));
    end
    R_trace = (reshape(R_cycle(1,1,:)+R_cycle(2,2,:)+R_cycle(3,3,:), [m_pos,nsample]))';
    S0Mat = (abs(acos((R_trace-1)./2))/pi);

    S_vec = ones(1,m);

    
    A = sparse(nsample*2*m_pos, m_pos);
    b = zeros(1, nsample*2*m_pos);

    for l = 1:m_pos
        IJ = CoDeg_pos_ind(l);
        % repair all of this to use m_pos instead of m
        i = Ind_i(IJ);
        j = Ind_j(IJ);
        if mod(l, 1000) == 0
            disp('next 1000 done');
        end
        offset = nsample*2*(l-1); % previously filled
        for kInd = 1:nsample
            % Then CoIndMat(kInd,l) is the vertex k
            k = CoIndMat(kInd, l);
            A(offset + 2*kInd - 1, l) = 1;
            A(offset + 2*kInd -1, CoDeg_pos_ind_long(IndMat(i,k))) = -1;
            A(offset + 2*kInd - 1, CoDeg_pos_ind_long(IndMat(j,k))) = -1;
            b(offset + 2*kInd - 1) = S0Mat(kInd, l); %possibly reversed

            A(offset + 2*kInd, l) = -1; 
            A(offset + 2*kInd, CoDeg_pos_ind_long(IndMat(i,k))) = -1;
            A(offset + 2*kInd, CoDeg_pos_ind_long(IndMat(j,k))) = -1;
            b(offset + 2*kInd) = -S0Mat(kInd, l); %possibly reversed
        end
    end

    f = ones(1,m_pos);
    lb = zeros(1,m_pos);
    ub = ones(1,m_pos); 

    % not confident in this indexing, have to check writeup
    S_vec(CoDeg_pos_ind(1:m_pos)) = linprog(f', A, b', [],[], lb', ub')
    R_est = GCW(Ind, AdjMat, RijMat, SVec);
    
end