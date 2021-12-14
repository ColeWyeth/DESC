rng(1);
n_sample = 20;
gcw_beta = 3;
learning_rate = 0.01;
learning_iters = 300;
AdjMat=data.AdjMat;
Hmat=data.Hmat;
G_gt=data.G_gt;
R_orig=data.R;
Focal=data.Focal_gt;
selected = (Focal>10^(-5));
AdjMat = AdjMat(selected,selected);
Hmat = Hmat(logical(kron(selected,ones(3,1))),logical(kron(selected,ones(3,1))));
G_gt = G_gt(logical(kron(selected,ones(3,1))),logical(kron(selected,ones(3,1))));
R_orig = R_orig(:,:,selected);

[S_k,C_k] = graphconncomp(sparse(AdjMat));
compSizes_k = zeros(S_k,1);
for i = 1:S_k
    compSizes_k(i) = sum(C_k == i);
end
[~, i_Csize] = max(compSizes_k);
selected = (C_k == i_Csize(1));

AdjMat = AdjMat(selected,selected);
Hmat = Hmat(logical(kron(selected,ones(3,1))),logical(kron(selected,ones(3,1))));
G_gt = G_gt(logical(kron(selected,ones(3,1))),logical(kron(selected,ones(3,1))));
R_orig = R_orig(:,:,selected);

[Ind_j, Ind_i] = find(tril(AdjMat,-1));
n = size(AdjMat,1);
m = size(Ind_i,1);
RijMat = zeros(3,3,m);
for k=1:m
    i = Ind_i(k); j = Ind_j(k);
    RijMat(:,:,k) = Hmat((3*i-2):(3*i), (3*j-2):(3*j));
end

Rij_orig = zeros(3,3,m);
for k = 1:m
    i=Ind_i(k); j=Ind_j(k); 
    Rij_orig(:,:,k)=R_orig(:,:,i)*(R_orig(:,:,j)');
end


R_err = zeros(3,3,m);
for j = 1:3
  R_err = R_err + bsxfun(@times,Rij_orig(:,j,:),RijMat(:,j,:));
end


R_err_trace = (reshape(R_err(1,1,:)+R_err(2,2,:)+R_err(3,3,:), [m,1]))';
ErrVec = abs(acos((R_err_trace-1)./2))/pi;


disp('triangle sampling')
%compute cycle inconsistency

Ind = [Ind_i Ind_j]';

t0=cputime;

% Start of DESC code

params.n_sample = n_sample;
params.beta = gcw_beta;
params.iters = learning_iters;
%params.Gradient = PiecewiseStepSize(learning_rate, 25);
params.Gradient = AdamGradient(0.005, 0.8, 0.8); % 0.001, 0.9, 0.999
%params.Gradient = ConstantStepSize(learning_rate);
params.make_plots = true;
params.R_orig = R_orig; % to plot convergence only
params.ErrVec = ErrVec; % to plot convergence only

%R_est = desc_rotation_sampled(Ind', RijMat, params);
R_est = desc_rotation(Ind', RijMat, params);
    
t1=cputime-t0;


RijMat1 = permute(RijMat, [2,1,3]);


%plot( S_vec,ErrVec, 'r.')

t20 = cputime;
R_est_Huber = AverageSO3Graph(RijMat1, Ind);
t2 = cputime-t20;

t30=cputime;
R_est_L12 = AverageL1(RijMat1,Ind);
t3=cputime-t30;
%R_est_L12_CEMP = AverageL1(RijMat1,Ind, 'Rinit', R_est);

%compute error
[~, MSE_DESC_mean,MSE_DESC_median, ~] = GlobalSOdCorrectRight(R_est, R_orig);
[~, MSE_Huber_mean, MSE_Huber_median,~] = GlobalSOdCorrectRight(R_est_Huber, R_orig);
[~, MSE_L12_mean, MSE_L12_median,~] = GlobalSOdCorrectRight(R_est_L12, R_orig);
%[~, MSE_CEMP_L12_mean,MSE_CEMP_L12_median ~] = GlobalSOdCorrectRight(R_est_L12_CEMP, R_orig);


fprintf('DESC %f %f %f\n',MSE_DESC_mean, MSE_DESC_median, t1); 
fprintf('Huber %f %f %f\n',MSE_Huber_mean, MSE_Huber_median, t2); 
fprintf('L1/2 %f %f %f\n',MSE_L12_mean, MSE_L12_median, t3); 
%fprintf('CEMP+L1/2 %f %f\n',MSE_CEMP_L12_mean, MSE_CEMP_L12_median); 


