% parameters with uniform topology
n=100; p=0.5; q=0.4; sigma=0.0; model='uniform';
% parameters with nonuniform topology
p_node_crpt=0.5; p_edge_crpt=0.75; sigma_in=0.5; sigma_out=4; crpt_type='adv';


% generate data with uniform topology
model_out = Uniform_Topology(n,p,q,sigma,model);

% for self-consistent corruption (in MPLS paper) run:
% q=0.45;
% model_out = Rotation_Graph_Generation(n,p,q,sigma,'self-consistent');


%model_out = Nonuniform_Topology(n,p, p_node_crpt,p_edge_crpt, sigma_in, sigma_out, crpt_type);

final_beta = 5; % temp

Ind = model_out.Ind; % matrix of edge indices (m by 2)
RijMat = model_out.RijMat; % given corrupted and noisy relative rotations
ErrVec = model_out.ErrVec; % ground truth corruption levels
R_orig = model_out.R_orig; % ground truth rotations

% set DESC default parameters 
lr = 1;
DESC_parameters.iters = 300; 
DESC_parameters.learning_rate = lr;
DESC_parameters.Gradient = PiecewiseStepSize(lr, 25);
%DESC_parameters.Gradient = AdamGradient(0.001, 0.9, 0.999); 
DESC_parameters.beta = final_beta;
DESC_parameters.n_sample = 15;
DESC_parameters.make_plots = true;
DESC_parameters.R_orig = R_orig;
DESC_parameters.ErrVec = ErrVec;

% For dense graphs with sufficient uncorrupted 3-cycles for all edges, 
% the following parameters may work even better: reweighting parameter can gradually
% increase (in ICML paper we fix beta=32 for MPLS). One can increasingly weigh 3-cycle
% consistency information and ignore residual nformation (in ICML paper we
% gradually ignore cycle information and weigh residual more). 

% MPLS_parameters.reweighting = 0.1*1.5.^((1:15)-1);
% MPLS_parameters.cycle_info_ratio = 1-1./((1:MPLS_parameters.max_iter)+1);
tic;
% run DESC
R_DESC = desc_rotation(Ind, RijMat, DESC_parameters); 
desc_time = toc;

tic;
% run sampled DESC
R_DESC_SAMPLED = desc_rotation_sampled(Ind, RijMat, DESC_parameters);
desc_sampled_time = toc;

tic;
% run matrix DESC
R_DESC_MATRIX = desc_rotation_matrix(Ind, RijMat, DESC_parameters);
desc_matrix_time = toc;

tic;
% run GCW baseline (edge weights are calculated as above but wijk is
% default)
DESC_parameters.iters = 0;
DESC_parameters.make_plots = false;
R_GCW = desc_rotation(Ind, RijMat, DESC_parameters);
gcw_time = toc;

% rotation alignment for evaluation
[~, ~, mean_error_DESC, median_error_DESC] = Rotation_Alignment(R_DESC, R_orig); 
[~, ~, mean_error_DESC_SAMPLED, median_error_DESC_SAMPLED] = Rotation_Alignment(R_DESC_SAMPLED, R_orig);
[~, ~, mean_error_DESC_MATRIX, median_error_DESC_MATRIX] = Rotation_Alignment(R_DESC_MATRIX, R_orig);
[~, ~, mean_error_GCW, median_error_GCW] = Rotation_Alignment(R_GCW, R_orig);

% Report estimation error
sz = [4 4];
varTypes = {'string','double','double','double'};
varNames = {'Algorithms','MeanError','MedianError','Runtime'};
Results = table('Size',sz,'VariableTypes',varTypes, 'VariableNames',varNames);
Results(1,:)={'DESC', mean_error_DESC, median_error_DESC, desc_time};
Results(2,:)={'DESC_SAMPLED', mean_error_DESC_SAMPLED, median_error_DESC_SAMPLED, desc_sampled_time};
Results(3,:)={'DESC_MATRIX', mean_error_DESC_MATRIX, median_error_DESC_MATRIX, desc_matrix_time};
Results(4,:)={'GCW', mean_error_GCW, median_error_GCW, gcw_time};


Results

