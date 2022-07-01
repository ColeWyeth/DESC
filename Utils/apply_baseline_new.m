function [result,data] = apply_baseline_new(data)
%% Code Written Onur Ozyesil
%% [1] O. Ozyesil, A. Singer, R. Basri,    
%%     Stable Camera Motion Estimation Using Convex Programming, 
%%     CVPR 2015

%variable assignment
AdjMat=data.AdjMat;
Hmat=data.Hmat;
G_gt=data.G_gt;
n=data.n;
CorrPts=data.corr;
Rmat_orig=data.R;
t_orig=squeeze(data.t);
K_mats=data.K;
FocalLengths_0=data.Focal_gt;
E=data.E;
E_est=data.E_est;

W_x_full=data.W_x_full; W_y_full=data.W_y_full; W_mask_full=data.W_mask_full;

%% Rotation Estimation

iterNum_IEVM = 5; % Tune as desired: To high gives too low connectivity, too low may produce relatively erroneous estimates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
evalc('[SO3Mats_est, IndVec_IEVM] = IterativeSO3Average(AdjMat,Hmat,iterNum_IEVM);');
SO3Mats_est = permute(SO3Mats_est,[2 1 3]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IEVM_cut = zeros(n,1); IEVM_cut(IndVec_IEVM) = 1; IEVM_cut = logical(IEVM_cut);
n = sum(IEVM_cut);
AdjMat = AdjMat(IEVM_cut,IEVM_cut);
Hmat = Hmat(logical(kron(IEVM_cut,ones(3,1))),logical(kron(IEVM_cut,ones(3,1))));
G_gt = G_gt(logical(kron(IEVM_cut,ones(3,1))),logical(kron(IEVM_cut,ones(3,1))));
CorrPts = CorrPts(IEVM_cut,IEVM_cut);
Rmat_orig = Rmat_orig(:,:,IEVM_cut);
t_orig = t_orig(:,IEVM_cut);
K_mats=K_mats(:,:,IEVM_cut);
FocalLengths_0=FocalLengths_0(:,IEVM_cut);

E=E(logical(kron(IEVM_cut,ones(3,1))),logical(kron(IEVM_cut,ones(3,1))));
E_est=E_est(IEVM_cut,IEVM_cut);

W_x_full=W_x_full(:,IEVM_cut); W_y_full=W_y_full(:,IEVM_cut); W_mask_full=W_mask_full(:,IEVM_cut);

%% Check for rigidity once more:
rigidityDec = ParallelRigidityTest(AdjMat,3);
if rigidityDec
    fprintf('\n Measurement graph is parallel rigid :) \n');
    PRcomp2 = logical(ones(n,1));
else
    fprintf('\n Measurement graph is not parallel rigid! \n Extracting largest maximally parallel rigid component... \n');
    [LarMaxRigCompInds2,~] = LargestMaxParRigComp(AdjMat,3);
    PRcomp2 = logical(zeros(n,1)); PRcomp2(LarMaxRigCompInds2) = true;
    n = sum(PRcomp2);
    AdjMat = AdjMat(PRcomp2,PRcomp2);
    Hmat = Hmat(logical(kron(PRcomp2,ones(3,1))),logical(kron(PRcomp2,ones(3,1))));
    G_gt = G_gt(logical(kron(PRcomp2,ones(3,1))),logical(kron(PRcomp2,ones(3,1))));    
    CorrPts = CorrPts(PRcomp2,PRcomp2);
    Rmat_orig = Rmat_orig(:,:,PRcomp2);
    t_orig = t_orig(:,PRcomp2);
    SO3Mats_est = SO3Mats_est(:,:,PRcomp2);
    K_mats=K_mats(:,:,PRcomp2);
    FocalLengths_0=FocalLengths_0(:,PRcomp2);
    
    E=E(logical(kron(PRcomp2,ones(3,1))),logical(kron(PRcomp2,ones(3,1))));
    E_est=E_est(PRcomp2,PRcomp2);
    
    W_x_full=W_x_full(:,PRcomp2); W_y_full=W_y_full(:,PRcomp2); W_mask_full=W_mask_full(:,PRcomp2);

end


%% Convert the data to our format

ComparisonInds = (FocalLengths_0 > 10^(-5)); %neglect element with small focal length
 

Jmat = diag([1 1 -1]);
Rmat_orig2 = zeros(3,3,n);
t_orig2 = zeros(3,n);
SO3Mats_est2 = zeros(3,3,n);
invK_Mats = zeros(3,3,n);

for k = 1:n
    SO3Mats_est2(:,:,k) = Jmat*SO3Mats_est(:,:,k)'*Jmat;
    Rmat_orig2(:,:,k) = Jmat*Rmat_orig(:,:,k)'*Jmat;
    t_orig2(:,k) = -Rmat_orig2(:,:,k)*Jmat*t_orig(:,k);
    %t_orig2(:,k) = -Rmat_orig(:,:,k)'*t_orig(:,k);

    K_mats(2,2,k)=-K_mats(2,2,k);
    invK_Mats(:,:,k) = inv(K_mats(:,:,k));%[1/f 0 -(r/2)/f;0 -1/f (c/2)/f; 0 0 1];%
end
t_orig2_cntrd = t_orig2(:,ComparisonInds) - mean(t_orig2(:,ComparisonInds),2)*ones(1,size(t_orig2(:,ComparisonInds),2));

%% Estimate pairwise line info
evalc('[tijMat,CostMat,Tijcell]= RobustSignedLineEstimInvK(SO3Mats_est2,CorrPts,invK_Mats);');
[SO3Mats_corr2, MSE_rots2, R_global] = GlobalSOdCorrectLeft(SO3Mats_est2, Rmat_orig2);
%% Estimate locations: 

LocEstMethod = 'LUD';
% LocEstMethod = 'SDR'; % Can provide code if desired, skipped for now
% LocEstMethod = 'CLS'; % Can provide code if desired, skipped for now
% LocEstMethod = 'LS'; % Can provide code if desired, skipped for now

optsIRLS.tolIRLS = 1e-4;       
optsIRLS.tolQuad = 1e-8;
optsIRLS.delt = 1e-12;     
optsIRLS.maxit   = 200;  
evalc('[t_est, ~] = LocationEstimByLUD(AdjMat,tijMat,optsIRLS);');

%% resolve global ambiguity
% Note : here we delete those camera with focal length < 10^-5 for
% measuring T ambiguity not for R ambiguity

[t_fit2,t_opt2,c_opt2,NRMSE_LUD2,~] = SimpleTransScaleRemove(R_global*t_est(:,ComparisonInds), t_orig2_cntrd,'L1');
 figure; plot3(t_fit2(1,:),t_fit2(2,:),t_fit2(3,:),'b*');
 hold on; plot3(t_orig2_cntrd(1,:),t_orig2_cntrd(2,:),t_orig2_cntrd(3,:),'r*');
 


%% Translation Error
errT_mean=mean(sqrt(sum((t_fit2 - t_orig2_cntrd).^2)));
errT_median=median(sqrt(sum((t_fit2 - t_orig2_cntrd).^2)));

%% Rotation Error in degrees
% Delete those cameras with f < 10^-5 for comparison

[E_res,e,normE]=CompareRotations(SO3Mats_corr2(:,:,ComparisonInds),Rmat_orig2(:,:,ComparisonInds));
errR_mean=E_res(1); errR_median=E_res(2);
errRnorm_mean=normE(1); errRnorm_median=normE(2);

%% post processing data collection
% ignore the comparisonid for the purpose of data colelction. use it only
% for  removing ambiguity
data.ComparisonInds=ComparisonInds;
data.t=t_orig; %original format of GT data i.e. w/o J and R' rectification
data.R=Rmat_orig; %original format of GT data w/oJ
data.n=size(data.t,2);
data.nPairs=data.n*(data.n-1)/2;
data.AdjMat=AdjMat;
data.keep=logical(double(AdjMat));
data.corr=CorrPts;
data.Hmat=Hmat;
data.G_gt=G_gt;
data.K=K_mats;
data.invK=invK_Mats;
data.E=E; data.E_gt=E;
data.E_est=E_est;
data.Focal_gt=FocalLengths_0;

%% collect result.
result.TOut=zeros(3,data.n);
result.TOut(:,ComparisonInds)=t_fit2+mean(t_orig2(:,ComparisonInds),2)*ones(1,size(t_orig2(:,ComparisonInds),2));
result.T_out=t_est;
result.Tran_err_mean=errT_mean;
result.Tran_err_median=errT_median;
result.Rot_err_mean=errR_mean;
result.Rot_err_median=errR_median;
result.Rot_fro_err_mean=errRnorm_mean;
result.Rot_fro_err_median=errRnorm_median;

ROut=SO3Mats_corr2; R_out=SO3Mats_est2; %make this into cell

% calculate estimated E
EOut = zeros(3*data.n);
errs_f_gt = nan(data.n);
errs_ff_gt = nan(data.n);
J=diag([1 1 -1]);

for i=1:data.n
    result.ROut{i}=ROut(:,:,i); result.R_out{i}=R_out(:,:,i);
    for j=i+1:data.n
        if (i==j)
            continue;
        end
        
        EOut(s3(i),s3(j)) = J*R_out(:,:,i)'*crossProdMat(result.T_out(:,i) - result.T_out(:,j))*R_out(:,:,j)*J;
        errs_f_gt(i,j) = getFerr(s3(EOut,i,j),s3(data.E,i,j));
        %F error delete
        errs_ff_gt(i,j) = getFFerr(s3(EOut,i,j),s3(data.E,i,j),data.K(:,:,i),data.K(:,:,j));
    
        errs_f_gt(j,i)=errs_f_gt(i,j);
        errs_ff_gt(j,i)=errs_ff_gt(i,j);
        EOut(s3(j),s3(i))=EOut(s3(i),s3(j))';
    end
end
result.E_out=EOut;
result.errorF=nanmean(errs_f_gt(:));
result.errorF_median=nanmedian(errs_f_gt(:));

result.errorFF=nanmean(errs_ff_gt(:));
result.errorFF_median=nanmedian(errs_ff_gt(:));

% correct J sign of R and T

for i=1:n
    result.R_out{i}=J*result.R_out{i}'*J;
    result.T_out(:,i)=J*result.T_out(:,i);
    result.ROut{i}=J*result.ROut{i}'*J;
    result.TOut(:,i)=-J*result.TOut(:,i);
end
result.CostMat=CostMat;

data.W_mask_full=W_mask_full; data.W_x_full=W_x_full; data.W_y_full=W_y_full;


kp_idx=(sum(data.W_mask_full,2)>1);
data.W_x_full=data.W_x_full(kp_idx,:); data.W_y_full=data.W_y_full(kp_idx,:);
data.W_mask_full=data.W_mask_full(kp_idx,:);


end




