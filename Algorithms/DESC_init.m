%% Authors: Cole Wyeth, Yunpeng Shi
%% 
%%------------------------------------------------
%% Input Parameters: 
%% Ind: edge_num by 2 "edge indices matrix". Each row is the index of an edge (i,j). that is sorted as (1,2), (1,3), (1,4),... (2,3), (2,4),.... 
%% edge_num is the number of edges.
%% RijMat: 3 by 3 by edge_num tensor that stores the given relative rotations corresponding to Ind



%% Output:
%% R_est: Estimated rotations (3x3xn)

function [R_est, S_vec] = DESC_init(Ind, RijMat, params)

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
    n_sample = max(ceil(median(CoDeg_vec_pos)/4), 30); 
    
    CoDeg_vec_pos_sampled = min(CoDeg_vec_pos, n_sample);
    % we will only use at most n_sample cycles for each edge
    CoDeg_zero_ind = find(CoDeg_vec<0);
    %cum_ind = [0;cumsum(CoDeg_vec_pos)];
    cum_ind = [0;cumsum(CoDeg_vec_pos_sampled)]; 
    m_pos = length(CoDeg_pos_ind); % number of edges with cycles
    m_cycle = cum_ind(end); % number of cycles we will use 
    
    CoDeg_pos_ind_long = zeros(1,m);
    CoDeg_pos_ind_long(CoDeg_pos_ind) = 1:m_pos;

    Ind_ij = zeros(1,m_cycle);
    Ind_jk = zeros(1,m_cycle);
    Ind_ki = zeros(1,m_cycle);

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

    Rjk0Mat = zeros(3,3,m_cycle);
    Rki0Mat = zeros(3,3,m_cycle);
    IJK = zeros(1,m_cycle);
    IKJ = zeros(1,m_cycle);
    JKI = zeros(1,m_cycle);

    IJK_Mat = zeros(n,m_pos);

    for l = 1:m_pos
        IJ = CoDeg_pos_ind(l); % For each edge with cycles
        i=Ind_i(IJ); j=Ind_j(IJ);
        CoInd_ij= find(AdjMat(:,i).*AdjMat(:,j)); % k for edge i,j
        if length(CoInd_ij) >= n_sample
            CoInd_ij = datasample(CoInd_ij, n_sample, 'Replace', false);
        end
        Ind_ij((cum_ind(l)+1):cum_ind(l+1)) =  IJ; % index by cycle, get edge (ij)
        Ind_jk((cum_ind(l)+1):cum_ind(l+1)) =  IndMat(j,CoInd_ij); % index by cycle, get jk
        Ind_ki((cum_ind(l)+1):cum_ind(l+1)) =  IndMat(CoInd_ij,i); % index by cycle, get ik
        Rjk0Mat(:,:,(cum_ind(l)+1):cum_ind(l+1)) =  RijMat4d(:,:,j,CoInd_ij); 
        % index by cycle, get rotation j to  k
        Rki0Mat(:,:,(cum_ind(l)+1):cum_ind(l+1)) =  RijMat4d(:,:,CoInd_ij,i);
        % index by cycle, get rotation k to i
        IJK((cum_ind(l)+1):cum_ind(l+1)) = CoInd_ij; % index by cycle, get k
        IJK_Mat(1:CoDeg_vec_pos_sampled(l),l) =  CoInd_ij;   
        % index by cycle number, edge (with cycles), get k
    end

    % index by cycle number of edge (1 to it's total number of cycles), edge index:
    % determine whether cycle IKJ was sampled for edge IK
    IKJ_appears = false(n, m_pos);
    % determine whether cycle JKI was sampled for edge JK
    JKI_appears = false(n, m_pos);
    for l = 1:m_pos
        IJ = CoDeg_pos_ind(l);
        i=Ind_i(IJ); j=Ind_j(IJ);
        IK = CoDeg_pos_ind_long(IndMat(i,IJK((cum_ind(l)+1):cum_ind(l+1))));
        range_l = (cum_ind(l)+1):cum_ind(l+1);
        % index of each ik among edges with cycles (it has a cycle through
        % j)
        IK_cum = cum_ind(IK);
        [J_ind, ~] = find(IJK_Mat(:,IK)==j); % index of j among cycles of ik
        % has to depend on edge
        IKJ_appears(1:CoDeg_vec_pos_sampled(l), l) = any(IJK_Mat(:,IK)==j);
        % TODO: after sampling j may not be present
        % store cum_ind(l) range as a variable then index it with the above
        IKJ(range_l(IKJ_appears(:, l))) = IK_cum(IKJ_appears(:,l)) + J_ind; 
        % index by cycle, get cycle number of ik through j 

        JK = CoDeg_pos_ind_long(IndMat(j,IJK((cum_ind(l)+1):cum_ind(l+1))));
        % index of each jk among edges with cycles (it has a cycle through
        % i)
        JK_cum = cum_ind(JK);
        [I_ind, ~] = find(IJK_Mat(:,JK)==i);  
        JKI_appears(1:CoDeg_vec_pos_sampled(l), l) = any(IJK_Mat(:,JK)==i);
        JKI(range_l(JKI_appears(:,l))) = JK_cum(JKI_appears(:,l)) + I_ind;
        % index by cycle, get cycle number of jk through i
    end

    Rij0Mat = RijMat(:,:,Ind_ij);
    % Rotations for edges with cycles

    disp('compute R cycle')
    R_cycle0 = zeros(3,3,m_cycle);
    % intermediary (rotation ij * jk)
    R_cycle = zeros(3,3,m_cycle);
    % net rotation for the full cycle
    for j = 1:3
      R_cycle0 = R_cycle0 + bsxfun(@times,Rij0Mat(:,j,:),Rjk0Mat(j,:,:));
    end

    for j = 1:3
      R_cycle = R_cycle + bsxfun(@times,R_cycle0(:,j,:),Rki0Mat(j,:,:));
    end

    disp('S0Mat')
    R_trace = reshape(R_cycle(1,1,:)+R_cycle(2,2,:)+R_cycle(3,3,:),[1,m_cycle]);
    S0_long = abs(acos((R_trace-1)./2))/pi;
    S_vec = ones(1,m);


    wijk = ones(1,m_cycle);
    for l=1:m_pos
        IJ = CoDeg_pos_ind(l);
        weight = wijk((cum_ind(l)+1):cum_ind(l+1));
        wijk((cum_ind(l)+1):cum_ind(l+1)) = weight/sum(weight);
        S_vec(IJ) = wijk((cum_ind(l)+1):cum_ind(l+1)) * (S0_long((cum_ind(l)+1):cum_ind(l+1)))';
    end


        disp('Initialization completed!')

        disp('Reweighting Procedure Started ...')

    sum_ikj = zeros(1,m_cycle);
    sum_jki = zeros(1,m_cycle);

    S_vec_last = S_vec;
    %%%%%%%%%%%%%
    %learning_rate = params.learning_rate;%0.01;
    learning_iters = params.iters;
    rm=1;
    proj=1;
    
    % Convergence plotting information
    svec_errors = zeros(1,0);
    obj_vals = zeros(1,0);
    MSE_means = zeros(1,0);
    MSE_medians = zeros(1,0);

    patience = 30;
    misses = 0;
    for iter = 1:learning_iters
           %tic; 
           %step_size  = (learning_rate/(2^fix(iter/25)));
           %step_size  = learning_rate;
           for l = 1:m_pos % for each edge ij 
               IJ = CoDeg_pos_ind(l);
               i=Ind_i(IJ); j=Ind_j(IJ); 
               range_l = (cum_ind(l)+1):cum_ind(l+1);
               sum_ikj(range_l(IKJ_appears(:,l))) = sum(wijk(IKJ(range_l(IKJ_appears(:,l)))));
               sum_jki(range_l(JKI_appears(:,l))) = sum(wijk(JKI(range_l(JKI_appears(:,l)))));
           end   

           grad_long = S_vec(Ind_jk)+S_vec(Ind_ki)+(sum_ikj+sum_jki).*S0_long; % Random: normrnd(0,1, size(S_vec(Ind_jk)));

           for l = 1:m_pos
               IJ = CoDeg_pos_ind(l);
               nsample = CoDeg_vec_pos_sampled(l);
               grad = grad_long((cum_ind(l)+1):cum_ind(l+1));  
               nv = ones(1,nsample)/(nsample^0.5);
               if rm==1
                    grad = grad - (grad*nv')*nv; % Riemmanian Project 
               end 
               grad_long((cum_ind(l)+1):cum_ind(l+1)) = grad; 
           end
%                w_new = ...
%                wijk((cum_ind(l)+1):cum_ind(l+1)) + params.Gradient.GetStep(grad);
           wijk = wijk + params.Gradient.GetStep(grad_long);
           for l = 1:m_pos
               IJ = CoDeg_pos_ind(l);
               nsample = CoDeg_vec_pos_sampled(l);
               w_new = wijk((cum_ind(l)+1):cum_ind(l+1));
               if proj==1
                       % proj to simplex
                       w = sort(w_new); 
                       Ti = 0; 
                       for i = 1:nsample
                           if sum(w(i:end)-w(i)) < 1
                               Ti = i;  
                               break 
                           end
                       end
                       T = w(Ti) - (1 - sum(w(Ti:end)-w(Ti)))/length(w(Ti:end)); 
                       wijk((cum_ind(l)+1):cum_ind(l+1)) = max(w_new - T, 0);
               else
                   wijk((cum_ind(l)+1):cum_ind(l+1)) ...
                   = wijk((cum_ind(l)+1):cum_ind(l+1))/sum(wijk((cum_ind(l)+1):cum_ind(l+1)));
               end
               S_vec(IJ) = wijk((cum_ind(l)+1):cum_ind(l+1)) * (S0_long((cum_ind(l)+1):cum_ind(l+1)))';
           end

    average_change = mean(abs(S_vec - S_vec_last));
    obj_vals(end+1) = wijk*(S_vec(Ind_jk)' + S_vec(Ind_ki)');

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
    %fprintf('%d: %f\n',iter,mean(abs(S_vec - ErrVec)))
    %meanErrors(iter) = mean(abs(S_vec - ErrVec));
    %one_iter_runtime = toc; 
    %fprintf('one iteration runtime: %f\n', one_iter_runtime);
    end

    R_est = GCW(Ind, AdjMat, RijMat, S_vec);
    
    if params.make_plots
        figure
        
%         plot(1:10, log10(MSE_means(:, 1:10)));
%         title('Convergence of Rotation Estimate, Mean Error (q=0.3)');
%         xlabel('Iteration number');
%         ylabel('log mean error in R estimate (degrees)');
        dlmwrite('linear_convergence_rotation_error.csv', MSE_means,'delimiter',',','-append');
        dlmwrite('linear_convergence_svec_error.csv', svec_errors,'delimiter',',','-append');
        ylim([0 10]);
        
        tiledlayout(2,2);
        
        nexttile
        plot(log10(svec_errors));
        title('Convergence of Corruption Estimate Vector (SVec, sampled)');
        xlabel('Iteration number');
        ylabel('log average distance to true corruption');
        
        nexttile
        plot(obj_vals);
        title('Convergence of Objective Function (sampled)');
        xlabel('Iteration number');
        ylabel('Value of Objective Function');
        
        nexttile
        plot(log10(MSE_means));
        title('Convergence of Rotation Estimate, Mean (sampled)');
        xlabel('Iteration number');
        ylabel('log mean error in R estimate (degrees)');
        %ylim([0 inf]);
        
        nexttile
        plot(MSE_medians);
        title('Convergence of Rotation Estimate, Median (sampled)');
        xlabel('Iteration number');
        ylabel('Median Error in R estimate (degrees)');
        ylim([0 inf]);
    end

end