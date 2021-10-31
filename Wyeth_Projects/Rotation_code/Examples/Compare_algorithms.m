% parameters with uniform topology
n =200; p=0.5; q=0.8; sigma=0.0; model='uniform';
% parameters with nonuniform topology
p_node_crpt=0.5; p_edge_crpt=0.75; sigma_in=0.5; sigma_out=4; crpt_type='adv';


% generate data with uniform topology
% model_out = Uniform_Topology(n,p,q,sigma,model);

% for self-consistent corruption (in MPLS paper) run:
% q=0.45;
% model_out = Rotation_Graph_Generation(n,p,q,sigma,'self-consistent');


model_out = Nonuniform_Topology(n,p, p_node_crpt,p_edge_crpt, sigma_in, sigma_out, crpt_type);

final_beta = 5; % temp

Ind = model_out.Ind; % matrix of edge indices (m by 2)
RijMat = model_out.RijMat; % given corrupted and noisy relative rotations
ErrVec = model_out.ErrVec; % ground truth corruption levels
R_orig = model_out.R_orig; % ground truth rotations

% set CEMP defult parameters
CEMP_parameters.max_iter = 6;
CEMP_parameters.reweighting = 2.^((1:6)-1);
CEMP_parameters.nsample = 50;
CEMP_parameters.gcw_beta = final_beta;

% set MPLS default parameters
MPLS_parameters.stop_threshold = 1e-3;
MPLS_parameters.max_iter = 100;
MPLS_parameters.reweighting = CEMP_parameters.reweighting(end);
MPLS_parameters.thresholding = [0.95,0.9,0.85,0.8];
MPLS_parameters.cycle_info_ratio = 1./((1:MPLS_parameters.max_iter)+1);

% set DESC default parameters 
DESC_parameters.iters = 200; 
DESC_parameters.beta = final_beta;
DESC_parameters.n_sample = 50;

% For dense graphs with sufficient uncorrupted 3-cycles for all edges, 
% the following parameters may work even better: reweighting parameter can gradually
% increase (in ICML paper we fix beta=32 for MPLS). One can increasingly weigh 3-cycle
% consistency information and ignore residual nformation (in ICML paper we
% gradually ignore cycle information and weigh residual more). 

% MPLS_parameters.reweighting = 0.1*1.5.^((1:15)-1);
% MPLS_parameters.cycle_info_ratio = 1-1./((1:MPLS_parameters.max_iter)+1);



% run MPLS and CEMP+MST
[R_MPLS, R_CEMP_MST] = MPLS(Ind,RijMat,CEMP_parameters, MPLS_parameters);

% run Spectral
R_SP = Spectral(Ind, RijMat);

% run CEMP+GCW
R_CEMP_GCW = CEMP_GCW(Ind, RijMat, CEMP_parameters);

% run IRLS
R_IRLS_GM = IRLS_GM(RijMat, Ind);
R_IRLS_L12 = IRLS_L12(RijMat,Ind);

% run DESC
R_DESC = desc_rotation(Ind, RijMat, DESC_parameters); 

% run sampled DESC
R_DESC_SAMPLED = desc_rotation_sampled(Ind, RijMat, DESC_parameters);

% rotation alignment for evaluation
[~, ~, mean_error_CEMP_MST, median_error_CEMP_MST] = Rotation_Alignment(R_CEMP_MST, R_orig);
[~, ~, mean_error_MPLS, median_error_MPLS] = Rotation_Alignment(R_MPLS, R_orig);
[~, ~, mean_error_SP, median_error_SP] = Rotation_Alignment(R_SP, R_orig);
[~, ~, mean_error_CEMP_GCW, median_error_CEMP_GCW] = Rotation_Alignment(R_CEMP_GCW, R_orig);
[~, ~, mean_error_IRLS_GM, median_error_IRLS_GM] = Rotation_Alignment(R_IRLS_GM, R_orig);
[~, ~, mean_error_IRLS_L12, median_error_IRLS_L12] = Rotation_Alignment(R_IRLS_L12, R_orig);
[~, ~, mean_error_DESC, median_error_DESC] = Rotation_Alignment(R_DESC, R_orig); 
[~, ~, mean_error_DESC_SAMPLED, median_error_DESC_SAMPLED] = Rotation_Alignment(R_DESC_SAMPLED, R_orig);

% Report estimation error
sz = [6 3];
varTypes = {'string','double','double'};
varNames = {'Algorithms','MeanError','MedianError'};
Results = table('Size',sz,'VariableTypes',varTypes, 'VariableNames',varNames);
Results(1,:)={'Spectral', mean_error_SP, median_error_SP};
Results(2,:)={'IRLS-GM', mean_error_IRLS_GM, median_error_IRLS_GM};
Results(3,:)={'IRLS-L0.5', mean_error_IRLS_L12, median_error_IRLS_L12};
Results(4,:)={'CEMP+MST', mean_error_CEMP_MST, median_error_CEMP_MST};
Results(5,:)={'CEMP+GCW', mean_error_CEMP_GCW, median_error_CEMP_GCW};
Results(6,:)={'MPLS', mean_error_MPLS, median_error_MPLS};
Results(7,:)={'DESC', mean_error_DESC, median_error_DESC};
Results(8,:)={'DESC', mean_error_DESC_SAMPLED, median_error_DESC_SAMPLED};


Results

