%% Authors: Cole Wyeth, Yunpeng Shi
%% 
%%------------------------------------------------
%% This version uses a constant number of loops for each edge.
%% The hope is that this can speed up the simplex projection
%% Input Parameters: 
%% Ind: edge_num by 2 "edge indices matrix". Each row is the index of an edge (i,j). that is sorted as (1,2), (1,3), (1,4),... (2,3), (2,4),.... 
%% edge_num is the number of edges.
%% RijMat: 3 by 3 by edge_num tensor that stores the given relative rotations corresponding to Ind



%% Output:
%% R_est: Estimated rotations (3x3xn)

function [R_est, S_vec] = desc_rotation_matrix(Ind, RijMat, params)

    %n_sample = params.n_sample; 
    
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
    
    % we will sample roughly a quarter of the median number of cycles
    n_sample = ceil(median(CoDeg_vec_pos)/4);
    
    % we will use exactly n_sample cycles for each edge
    CoDeg_zero_ind = find(CoDeg_vec<0);
    m_pos = length(CoDeg_pos_ind); % number of edges with cycles
    % m_cycle should be replaced by m_pos * n_sample

    CoDeg_pos_ind_long = zeros(1,m);
    CoDeg_pos_ind_long(CoDeg_pos_ind) = 1:m_pos;

    Ind_ij = zeros(n_sample,m_pos);
    Ind_jk = zeros(n_sample,m_pos);
    Ind_ki = zeros(n_sample,m_pos);

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
    CoIndMat = zeros(n_sample, m_pos);
    
    for l = 1:m_pos
        IJ = CoDeg_pos_ind(l); % For each edge with cycles
        i=Ind_i(IJ); j=Ind_j(IJ);
        CoInd_ij= find(AdjMat(:,i).*AdjMat(:,j)); % k for edge i,j
        CoInd_ij = datasample(CoInd_ij, n_sample, 'Replace', true);
        for k_ind = 1:n_sample
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

    Rki0 = zeros(3,3,m_pos,n_sample);
    Rjk0 = zeros(3,3,m_pos,n_sample);
    for l = 1:m_pos
        Rki0(:,:,l,:) = RijMat4d(:,:,CoIndMat(:,l), Ind_i(l));
        Rjk0(:,:,l,:) = RijMat4d(:,:,Ind_j(l),CoIndMat(:,l));
    end
    Rki0Mat = reshape(Rki0,[3,3,m_pos*n_sample]);
    Rjk0Mat = reshape(Rjk0,[3,3,m_pos*n_sample]);
    Rij0Mat = reshape(kron(ones(1,n_sample),reshape(RijMat(:,:,CoDeg_pos_ind),[3,3*m_pos])), [3,3,m_pos*n_sample]);
    
    R_cycle0 = zeros(3,3,m_pos*n_sample);
    R_cycle = zeros(3,3,m_pos*n_sample);
    
    for j = 1:3
      R_cycle0 = R_cycle0 + bsxfun(@times,Rij0Mat(:,j,:),Rjk0Mat(j,:,:));
    end

    for j = 1:3
      R_cycle = R_cycle + bsxfun(@times,R_cycle0(:,j,:),Rki0Mat(j,:,:));
    end
    R_trace = (reshape(R_cycle(1,1,:)+R_cycle(2,2,:)+R_cycle(3,3,:), [m_pos,n_sample]))';
    S0Mat = (abs(acos((R_trace-1)./2))/pi);

    S_vec = ones(1,m);


    wijk = ones(n_sample, m_pos)/n_sample;
    for l=1:m_pos
        IJ = CoDeg_pos_ind(l);
        S_vec(IJ) = wijk(:,l)' * S0Mat(:, l);
    end


        disp('Initialization completed!')

        disp('Reweighting Procedure Started ...')


    S_vec_last = S_vec;
    %%%%%%%%%%%%%
    %learning_rate = params.learning_rate;%0.01;
    learning_iters = params.iters;
    rm=1;
    proj=1;
    
    grad_second_term = zeros(n_sample, m_pos);
    
    % Convergence plotting information
    svec_errors = zeros(1,0);
    obj_vals = zeros(1,0);
    MSE_means = zeros(1,0);
    MSE_medians = zeros(1, 0);

    patience = 50;
    misses = 0;
    for iter = 1:learning_iters
       %step_size  = (learning_rate/(2^fix(iter/25)));
       %step_size  = learning_rate;
       
       for l = 1:m_pos % for each edge ij 
           IJ = CoDeg_pos_ind(l);
           i=Ind_i(IJ); j=Ind_j(IJ);
           ind = sub2ind(size(wijk), second_term_cycles{l}, second_term_edges{l});
           grad_second_term(:, l) = sum(wijk(ind));
           grad_second_term(:, l) = grad_second_term(:,l).*S0Mat(:,l);
       end   

       grad = S_vec(Ind_jk)+S_vec(Ind_ki)+grad_second_term;

       if rm==1
           nv = ones(1,n_sample)/(n_sample^0.5);
           grad = grad - nv'*(nv*grad); % Riemmanian Project 
       end 

       %wijk = wijk - step_size*grad;
       wijk = wijk + params.Gradient.GetStep(grad);
       
       wijk = SimplexProj(wijk')';

       for l = 1:m_pos
           IJ = CoDeg_pos_ind(l);
           S_vec(IJ) = wijk(:,l)' * S0Mat(:, l);
       end

        average_change = mean(abs(S_vec - S_vec_last));
        obj_vals(end+1) = sum(wijk.*(S_vec(Ind_jk) + S_vec(Ind_ki)), 'all');
        
        if params.make_plots
            svec_errors(end+1) = mean(abs(params.ErrVec - S_vec));
            R_est = GCW(Ind, AdjMat, RijMat, S_vec);
            [~, MSE_means(end+1),MSE_medians(end+1), ~] = GlobalSOdCorrectRight(R_est, params.R_orig);
        end

        fprintf('iter %d: average change in S_vec %f, objective value: %f\n', iter, average_change, obj_vals(end));
    
        if iter > 1 & obj_vals(end-1) - obj_vals(end) < 10^(-5)
            misses = misses + 1;
            if misses >= patience
                break
            end
        else 
            misses = 0;
        end
        S_vec_last = S_vec;

    end

    R_est = GCW(Ind, AdjMat, RijMat, S_vec);
    
    if params.make_plots
        figure
        tiledlayout(2,2);
        
        nexttile
        plot(svec_errors);
        title('Convergence of Corruption Estimate Vector (SVec, matrix)');
        xlabel('Iteration number');
        ylabel('Average distance to true corruption');
        
        nexttile
        plot(obj_vals);
        title('Convergence of Objective Function (matrix)');
        xlabel('Iteration number');
        ylabel('Value of Objective Function');
        
        nexttile
        plot(MSE_means);
        title('Convergence of Rotation Estimate, Mean (matrix)');
        xlabel('Iteration number');
        ylabel('Mean Error in R estimate (degrees)');
        ylim([0 inf]);
        
        nexttile
        plot(MSE_medians);
        title('Convergence of Rotation Estimate, Median (matrix)');
        xlabel('Iteration number');
        ylabel('Median Error in R estimate (degrees)');
        ylim([0 inf]);
    end
