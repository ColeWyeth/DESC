maxit   = 10; 
 niter = 10;
 incre= 1;
nsample=200;
n=100;
p=0.5;
q=0.5;
sigma=0;
G = rand(n,n) < p;
G = tril(G,-1);
% generate adjacency matrix
AdjMat = G + G'; 
[Ind_j, Ind_i] = find(G==1);
edge_num = length(Ind_i);
% generate 3D gaussian locations
TMat_gt = randn(3,n);
% compute ti-tj
GammaMat_gt = TMat_gt(:,Ind_i)-TMat_gt(:,Ind_j);
% normalize to get the ground truth pairwise directions
gamma_norm = sqrt(sum(GammaMat_gt.^2,1));
GammaMat_gt = bsxfun(@rdivide,GammaMat_gt,gamma_norm);
%add noise and corruption~U(S^2)
GammaMat = GammaMat_gt;
noiseInd = rand(1,edge_num)>=q;
% indices of corrupted edges
corrInd = logical(1-noiseInd);
% gaussian corruption/noise
corr_noise = randn(3,edge_num);
corr_noise_norm = sqrt(sum(corr_noise.^2,1));
GammaMat(:, noiseInd) = GammaMat_gt(:, noiseInd)+sigma*corr_noise(:, noiseInd);
GammaMat(:, corrInd) = corr_noise(:, corrInd);
gamma_norm = sqrt(sum(GammaMat.^2,1));
% normalize pairwise directions so that corruption ~U(S^2)
GammaMat = bsxfun(@rdivide,GammaMat,gamma_norm);
% compute the error of each pairwise direction
true_error = abs(acos(sum(GammaMat_gt.*GammaMat,1)));
tijMat = GammaMat;






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

    
        disp('Sampling Triangles...')
    
    % CoIndMat(:,l)= triangles sampled that contains l-th edge
    % e.g. l=(3,5), then CoIndMat(:,l)=[2,9,8,...] means that...
    % triangles 352, 359, 358,... are sampled
    for l = IndPos
        i = Ind_i(l); j = Ind_j(l);
       CoIndMat(:,l)= datasample(find(AdjMat(:,i).*AdjMat(:,j)), nsample);
    end
  
        disp('Triangle Sampling Finished!')
        disp('Computing Naive AAB ...')
    
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
   
        disp('Naive AAB Computed!')
    
    % compute maximal/minimal AAB inconsistency
    maxAAB = max(max(SAABMat0));
    minAAB = min(min(SAABMat0));
    
        disp('Reweighting Procedure Started ...')
   
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
        Smax = Ski+Sjk;
        % compute weight matrix (nsample by m)
        WeightMat = exp(-tau*Smax);
        weightsum = sum(WeightMat,1);
        % normalize so that each column sum up to 1
        WeightMat = bsxfun(@rdivide,WeightMat,weightsum);
        SAABMat = WeightMat.*SAABMat0;
        % IR-AAB at current iteration
        IRAABVec = sum(SAABMat,1);
       
            fprintf('Reweighting Iteration %d Completed!\n',iter)   
        
    end
    
        disp('Completed!')
    







t_start = tic;

tolIRLS = 1e-5;       
tolQuad = 1e-10;
delt = 1e-8;
 
maxitQuad = 200;
staglim = 5;


n = size(AdjMat,1);
d = size(tijMat,1);
ss_num = size(tijMat,2);

[Ind_j, Ind_i] = find(tril(AdjMat,-1));
j_Vec_Lmat = vec([Ind_i Ind_j]');
i_Vec_Lmat = kron([1:ss_num]',ones(2,1));
val_Vec_Lmat = kron(ones(ss_num,1),[1;-1]);
l_mat = sparse(i_Vec_Lmat,j_Vec_Lmat,val_Vec_Lmat,ss_num,n,2*ss_num);
Lmat = kron(l_mat,speye(d));
V = kron(ones(n,1),speye(d));

%% Matrix of cost function
Mmat = [Lmat -sparse([1:d*ss_num]',kron([1:ss_num]',ones(d,1)),tijMat(:),d*ss_num,ss_num,d*ss_num)];

%% Start IRLS loops
count = 1;
%wMat = speye(d*ss_num);
optsQuad = optimset('Algorithm','interior-point-convex','MaxIter',maxitQuad,...
    'TolFun',tolQuad,'Display','off');
cost_val_old = 1;
cost_vec = []; stagcnt = 0;

%wMat = kron(sparse([1:ss_num]',[1:ss_num]', (IRAABVec<0.05) + delt, ss_num,ss_num,ss_num),speye(d));
wMat = kron(sparse([1:ss_num]',[1:ss_num]', exp(-10* IRAABVec), ss_num,ss_num,ss_num),speye(d));

 %t_alph_est = quadprog(Mmat'*wMat*Mmat, sparse(d*n+ss_num,1), ...
       % [-sparse(ss_num,d*n) -speye(ss_num)], -ones(ss_num,1), [V' sparse(d,ss_num)], sparse(d,1),[],[],[],optsQuad);

NRMSEVec=[];


while (count <= maxit)&&(stagcnt <= staglim)
    
    %% Compute solution of quadratic program
    t_alph_est = quadprog(Mmat'*wMat*Mmat, sparse(d*n+ss_num,1), ...
        [-sparse(ss_num,d*n) -speye(ss_num)], -ones(ss_num,1), [V' sparse(d,ss_num)], sparse(d,1),[],[],[],optsQuad);
    
    %% Update IRLS weights
    residual_vec = reshape(Mmat*t_alph_est,d,ss_num);
    residual_norms = sqrt(bsxfun(@dot,residual_vec,residual_vec));
    
    Ski = zeros(nsample, m);
    Sjk = zeros(nsample, m);
        for l = IndPos
            i = Ind_i(l); j=Ind_j(l);
            Ski(:,l) = residual_norms(IndMat(i,CoIndMat(:,l)));
            Sjk(:,l) = residual_norms(IndMat(j,CoIndMat(:,l)));
        end
        Smax = Ski+Sjk;
        % compute weight matrix (nsample by m)
        WeightMat = exp(-tau*Smax);
        weightsum = sum(WeightMat,1);
        % normalize so that each column sum up to 1
        WeightMat = bsxfun(@rdivide,WeightMat,weightsum);
        SAABMat = WeightMat.*SAABMat0;
        % IR-AAB at current iteration
        IRAABVec = sum(SAABMat,1);
        ES_max = max(residual_norms, IRAABVec);
    
   % wMat = kron(sparse([1:ss_num]',[1:ss_num]',1./(ES_max + delt),ss_num,ss_num,ss_num),speye(d));
    wMat = kron(sparse([1:ss_num]',[1:ss_num]', (ES_max<1/tau) + delt,ss_num,ss_num,ss_num),speye(d));
   %wMat = kron(sparse([1:ss_num]',[1:ss_num]', exp(-tau*ES_max) ,ss_num,ss_num,ss_num),speye(d));
    
   
  % wMat = kron(sparse([1:ss_num]',[1:ss_num]', exp(-tau*residual_norms) ,ss_num,ss_num,ss_num),speye(d));
   
    tau=tau+incre;
    
    %% Check convergence of cost function
    cost_val = sum(residual_norms);
    cost_diff = abs(cost_val_old - cost_val)/cost_val_old;
    if (cost_diff <= tolIRLS)
        stagcnt = stagcnt + 1;
        display(stagcnt);
    else
        stagcnt = 0;
    end
    cost_val_old = cost_val;
    cost_vec = [cost_vec; cost_val];
    fprintf(' IRLS iteration %d done! \n',count);
    count = count + 1;
    t_est = reshape(t_alph_est(1:d*n),d,n);
    alph = t_alph_est(d*n+1:end);
    [t_fit, t_opt, c_opt, NRMSE, MSE_trans]=SimpleTransScaleRemove(t_est, TMat_gt, 'L2');
    NRMSEVec=[NRMSEVec; NRMSE];
end


NRMSEVec
