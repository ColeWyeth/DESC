%% Authors: Cole Wyeth, Yunpeng Shi
%% 
%%------------------------------------------------
%% Input Parameters: 
%% Ind: edge_num by 2 "edge indices matrix". Each row is the index of an edge (i,j). that is sorted as (1,2), (1,3), (1,4),... (2,3), (2,4),.... 
%% edge_num is the number of edges.
%% RijMat: 3 by 3 by edge_num tensor that stores the given relative rotations corresponding to Ind



%% Output:
%% R_est: Estimated rotations (3x3xn)

function [Rest, S_vec] = linprog_sij(Ind, RijMat)

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
    
    % we will sample roughly a quarter of the median number of cycles
    nsample = max(ceil(median(CoDeg_vec_pos)/4), 30); 

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
    S_vec(CoDeg_pos_ind(1:m_pos)) = linprog(f', A, b', [],[], lb', ub');
    %Rest = GCW(Ind, AdjMat, RijMat, S_vec);
    
    % Below is the original GCW code
    d=3;
    mat_size = ones(1,n)*d;
    cum_ind = [0,cumsum(mat_size)];
    Rij_blk = zeros(n*d);
    for k = 1:m
       i = Ind_i(k); j=Ind_j(k);
       Rij_blk((cum_ind(i)+1):cum_ind(i+1), (cum_ind(j)+1):cum_ind(j+1))= RijMat(:,:,k);    
    end
    
    Rij_blk = Rij_blk+Rij_blk';
    
    %%% Spectral 
    beta_T = 5; %beta/rate;
    SMat = sparse(Ind_i, Ind_j, S_vec, n, n);
    SMat = SMat + SMat';
    Weights = exp(-beta_T.*SMat).*AdjMat;
    Weights = diag(1./sum(Weights,2))*Weights;
    Weights = kron(Weights, ones(d));    
    RijW = Rij_blk.*Weights;
    clear 'Rij_blk';
    
    
    [V,~] = eigs(RijW,d,'la');
    V(:,1) = V(:,1)*sign(det(V(1:d,:))); % ensure det = 1
    R_est_GCW = zeros(d,d,n);
    for i=1:n
       Ri = V((cum_ind(i)+1):cum_ind(i+1), :); 
       [Ur,~,Vr] = svd(Ri);
       S0 = diag([ones(1,d-1),det(Ur*Vr')]);
       R_est_GCW(:,:,i) = Ur*S0*Vr';
       
    end

    % Improved rotation recovery
    changeThreshold=1e-3;
    RR = permute(RijMat, [2,1,3]);
    I = [Ind_i Ind_j]';
    Rinit = R_est_GCW;
    maxIters = 200;
    delt = 1e-16;
    quant_ratio = 1;
    %quant_ratio_min = sum(SVec<=cutoff)/length(SVec);
    quant_ratio_min = 0.8;
    thresh = quantile(S_vec,quant_ratio);
    top_threshold = 1e4;
    right_threshold = 1e-4;
    
    N=max(max(I));%Number of cameras or images or nodes in view graph
    
    
    %Convert Rij to Quaternion form without function call
    QQ=[RR(1,1,:)+RR(2,2,:)+RR(3,3,:)-1, RR(3,2,:)-RR(2,3,:),RR(1,3,:)-RR(3,1,:),RR(2,1,:)-RR(1,2,:)]/2;
    QQ=reshape(QQ,4,size(QQ,3),1)';
    QQ(:,1)=sqrt((QQ(:,1)+1)/2);
    QQ(:,2:4)=(QQ(:,2:4)./repmat(QQ(:,1),[1,3]))/2;
    
    
    %initialize
    Q=[Rinit(1,1,:)+Rinit(2,2,:)+Rinit(3,3,:)-1, Rinit(3,2,:)-Rinit(2,3,:),Rinit(1,3,:)-Rinit(3,1,:),Rinit(2,1,:)-Rinit(1,2,:)]/2;
    Q=reshape(Q,4,size(Q,3),1)';
    Q(:,1)=sqrt((Q(:,1)+1)/2);
    Q(:,2:4)=(Q(:,2:4)./repmat(Q(:,1),[1,3]))/2;
    
    
    % Formation of A matrix.
    m=size(I,2);
    
    i=[[1:m];[1:m]];i=i(:);
    j=I(:);
    s=repmat([-1;1],[m,1]);
    k=(j~=1);
    Amatrix=sparse(i(k),j(k)-1,s(k),m,N-1);
    
    w=zeros(size(QQ,1),4);W=zeros(N,4);
    
    score=inf;    Iteration=1;
    
    
    %Weights = sqrt(exp(-tau*SVec.^2))';
    
    %Weights = 1./(SVec.^0.75)'; Weights(Weights>1e8)=1e8;
    
    Weights = (1./(S_vec.^0.75)');
    %Weights = SIGMA./(S_vec.^2+SIGMA^2)'; 
    %Weights = ((S_vec<0.005)'); Weights(Weights>1e8)=1e8;
    
    %Weights = sqrt(exp(-50*SVec.^2))'.*((SVec<thresh)'); Weights(Weights>1e4)=1e4;
    Weights(Weights>top_threshold)= top_threshold;
    Weights(S_vec>thresh)=right_threshold; 
    
    while((score>changeThreshold)&&(Iteration<maxIters))
        lam = 1/(Iteration+1);
        %lam = 1/(Iteration^2+1);
        %lam = 0; %full residual
        %lam = 1; %full cemp
        %lam = 0.5;
        %lam = exp(-Iteration);
        
        %tau=beta;
        i=I(1,:);j=I(2,:);
    
        % w=Qij*Qi
        w(:,:)=[ (QQ(:,1).*Q(i,1)-sum(QQ(:,2:4).*Q(i,2:4),2)),...  %scalar terms
            repmat(QQ(:,1),[1,3]).*Q(i,2:4) + repmat(Q(i,1),[1,3]).*QQ(:,2:4) + ...   %vector terms
            [QQ(:,3).*Q(i,4)-QQ(:,4).*Q(i,3),QQ(:,4).*Q(i,2)-QQ(:,2).*Q(i,4),QQ(:,2).*Q(i,3)-QQ(:,3).*Q(i,2)] ];   %cross product terms
    
        % w=inv(Qj)*w=inv(Qj)*Qij*Qi
        w(:,:)=[ (-Q(j,1).*w(:,1)-sum(Q(j,2:4).*w(:,2:4),2)),...  %scalar terms
            repmat(-Q(j,1),[1,3]).*w(:,2:4) + repmat(w(:,1),[1,3]).*Q(j,2:4) + ...   %vector terms
            [Q(j,3).*w(:,4)-Q(j,4).*w(:,3),Q(j,4).*w(:,2)-Q(j,2).*w(:,4),Q(j,2).*w(:,3)-Q(j,3).*w(:,2)] ];   %cross product terms
    
    
%         s2=sqrt(sum(w(:,2:4).*w(:,2:4),2));
%         w(:,1)=2*atan2(s2,w(:,1));
%         i=w(:,1)<-pi;  w(i,1)=w(i,1)+2*pi;  i=w(:,1)>=pi;  w(i,1)=w(i,1)-2*pi;
%         B=w(:,2:4).*repmat(w(:,1)./s2,[1,3]);
    % Here is an alternative solution for the above 4 lines. This may be
    % marginally faster. But use of this is not recomended as the domain of
    % acos is bounded which may result in truncation error when the solution
    % comes near optima. Usage of atan2 justifies omition of explicit
    % quaternion normalization at every stage.
        i=w(:,1)<0;w(i,:)=-w(i,:);
        theta2=acos(w(:,1));
        B=((w(:,2:4).*repmat((2*theta2./sin(theta2)),[1,3])));
        
        
        B(isnan(B))=0;% This tackles the devide by zero problem.
      
        W(1,:)=[1 0 0 0];
        
        %W(2:end,2:4)=Amatrix\B;
        %if(N<=1000)
            W(2:end,2:4)=(sparse(1:length(Weights),1:length(Weights),Weights,length(Weights),length(Weights))*Amatrix)\(repmat(Weights,[1,size(B,2)]).*B);
        %else
            %W(2:end,2:4)=Amatrix\B;
        %end
        
         score=sum(sqrt(sum(W(2:end,2:4).*W(2:end,2:4),2)))/N;
        
        theta=sqrt(sum(W(:,2:4).*W(:,2:4),2));
        W(:,1)=cos(theta/2);
        W(:,2:4)=W(:,2:4).*repmat(sin(theta/2)./theta,[1,3]);
        
        W(isnan(W))=0;
        
        Q=[ (Q(:,1).*W(:,1)-sum(Q(:,2:4).*W(:,2:4),2)),...  %scalar terms
            repmat(Q(:,1),[1,3]).*W(:,2:4) + repmat(W(:,1),[1,3]).*Q(:,2:4) + ...   %vector terms
            [Q(:,3).*W(:,4)-Q(:,4).*W(:,3),Q(:,4).*W(:,2)-Q(:,2).*W(:,4),Q(:,2).*W(:,3)-Q(:,3).*W(:,2)] ];   %cross product terms
    
        Iteration=Iteration+1;
       % disp(num2str([Iteration score toc]));
       
        R=zeros(3,3,N);
        for i=1:size(Q,1)
            R(:,:,i)=q2R(Q(i,:));
        end
    
    
    %[~, MSE_mean,MSE_median, ~] = GlobalSOdCorrectRight(R, R_orig);
    
    
    %fprintf('CEMP_IRLS_new:  mean %f median %f\n',MSE_mean, MSE_median); 
    
        
        E=(Amatrix*W(2:end,2:4)-B); 
        residual_standard = sqrt(sum(E.^2,2))/pi;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ESVec = (1-lam)*residual_standard + lam * S_vec';
        %ESVec = residual_standard .* (lam * S_vec' + (1-lam) * ones(m,1) * mean(S_vec));
        %ESVec = max(residual_standard, S_vec');
        %Weights = sqrt(exp(-tau*(SVec'.^2)));
        %Weights = 1./(ESVec+delt);
        %Weights = (ESVec<1/tau);
        %Weights = 1./(ESVec.^0.75); Weights(Weights>1e8)=1e8;
        Weights = (1./(ESVec.^0.75));
        %Weights = ((SVec<0.005)'); Weplights(Weights>1e8)=1e8;
        %SIGMA = max(SIGMA/2, sigma_final/10);
        %SIGMA = sigma_final/10;
        %Weights = SIGMA./(ESVec.^2+SIGMA^2); 
        
        
        %Weights = exp(-beta.*Err);
        
        %Weights=SIGMA./( sum(E.^2,2) + SIGMA^2 );
        
        %top_threshold = min(top_threshold*2, 1e4);
        right_threshold = 1e-4;
        quant_ratio = max(quant_ratio_min, quant_ratio-0.05);
        thresh = quantile(ESVec,quant_ratio);
        Weights(Weights>top_threshold)= top_threshold;
        Weights(ESVec>thresh)=right_threshold; 
        %G=2*(repmat(Weights.*Weights,[1,size(Amatrix,2)]).*Amatrix)'*E;
        %G=2*Amatrix'*sparse(1:length(Weights),1:length(Weights),Weights.*Weights,length(Weights),length(Weights))*E;
        
        %score=norm(W(2:end,2:4));

    end
    %toc
    
    R=zeros(3,3,N);
    for i=1:size(Q,1)
        R(:,:,i)=q2R(Q(i,:));
    end
    
    
    if(Iteration>=maxIters);disp('Max iterations reached');end

   Rest = R;

end