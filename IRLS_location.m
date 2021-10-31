maxit   = 100;  

n=100;
p=0.5;
q=0.8;
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


t_start = tic;

tolIRLS = 1e-5;       
tolQuad = 1e-10;
delt = 1e-16;

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
i = 1;
wMat = speye(d*ss_num);
optsQuad = optimset('Algorithm','interior-point-convex','MaxIter',maxitQuad,...
    'TolFun',tolQuad,'Display','off');
cost_val_old = 1;
cost_vec = []; stagcnt = 0;
a=0.5;
NRMSEVec=[];
while (i <= maxit)&&(stagcnt <= staglim)
    
    %% Compute solution of quadratic program
    t_alph_est = quadprog(Mmat'*wMat*Mmat, sparse(d*n+ss_num,1), ...
        [-sparse(ss_num,d*n) -speye(ss_num)], -ones(ss_num,1), [V' sparse(d,ss_num)], sparse(d,1),[],[],[],optsQuad);
    
    %% Update IRLS weights
    residual_vec = reshape(Mmat*t_alph_est,d,ss_num);
    residual_norms = sqrt(bsxfun(@dot,residual_vec,residual_vec));
    
    
    wMat = kron(sparse([1:ss_num]',[1:ss_num]',1./(sqrt(residual_norms.^(2) + delt)),ss_num,ss_num,ss_num),speye(d));
   % wMat = kron(sparse([1:ss_num]',[1:ss_num]', (residual_norms<1/a) + delt,ss_num,ss_num,ss_num),speye(d));
   % wMat = kron(sparse([1:ss_num]',[1:ss_num]', exp(-a*residual_norms) ,ss_num,ss_num,ss_num),speye(d));
    
    a=a+0.5;
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
    %fprintf(' IRLS iteration %d done! \n',i);
    i = i + 1;
    t_est = reshape(t_alph_est(1:d*n),d,n);
    alph = t_alph_est(d*n+1:end);
    [t_fit, t_opt, c_opt, NRMSE, MSE_trans]=SimpleTransScaleRemove(t_est, TMat_gt, 'L2');
    NRMSEVec=[NRMSEVec; NRMSE];
end



t_end = toc(t_start);

NRMSEVec

%evalc('[t_est, ~] = LocationEstimByLUD(AdjMat,GammaMat,optsIRLS);');


