function R_est = GCW(Ind, AdjMat, RijMat, S_vec)
    n=max(Ind,[],'all');
    Ind_i = Ind(:,1);
    Ind_j = Ind(:,2);
    m=size(Ind_i,1); % number of edges
    d=3;
    mat_size = ones(1,n)*d;
    cum_ind = [0,cumsum(mat_size)];
    Rij_blk = zeros(n*d);
    for k = 1:m
       i = Ind_i(k); j=Ind_j(k);
       Rij_blk((cum_ind(i)+1):cum_ind(i+1), (cum_ind(j)+1):cum_ind(j+1))= RijMat(:,:,k);    
    end

    Rij_blk = Rij_blk+Rij_blk';    

    SMat_sq = sparse(Ind_i, Ind_j, S_vec, n, n);
    SMat_sq = SMat_sq + SMat_sq';
    %Weights = exp(-params.beta.*SMat_sq).*AdjMat;
    Weights = (1 ./ (SMat_sq.^(3/2) + 1e-8)) .* AdjMat;
    Weights = diag(1./sum(Weights,2))*Weights; % normalize
    Weights = kron(Weights, ones(d));    
    RijW = Rij_blk.*Weights;
    clear 'Rij_blk';


    [V,~] = eigs(RijW,d,'la');
    V(:,1) = V(:,1)*sign(det(V(1:d,:))); % ensure det = 1
    R_est = zeros(d,d,n);
    for i=1:n
       Ri = V((cum_ind(i)+1):cum_ind(i+1), :); 
       [Ur,~,Vr] = svd(Ri);
       S0 = diag([ones(1,d-1),det(Ur*Vr')]);
       R_est(:,:,i) = Ur*S0*Vr';

    end

end