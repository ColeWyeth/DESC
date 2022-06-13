%% Authors: Cole Wyeth, Yunpeng Shi
%% 
%%------------------------------------------------
%% Input Parameters: 
%% Ind: edge_num by 2 "edge indices matrix". Each row is the index of an edge (i,j). that is sorted as (1,2), (1,3), (1,4),... (2,3), (2,4),.... 
%% edge_num is the number of edges.
%% RijMat: 3 by 3 by edge_num tensor that stores the given relative rotations corresponding to Ind



%% Output:
%% R_est: Estimated rotations (3x3xn)

function [R_init, R_est, S_vec] = desc_irls_geodesic_rotation_sampled(Ind, RijMat, params)

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

 
    codeg = max(0, CoDeg_vec);
    
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

    end


    nbin=ceil(m/10);
    hs_value = histcounts(S_vec,0:(1/nbin):1);
    [~, hs_peak] = max(hs_value);
    inflection_pt = hs_peak/nbin*2;
    sigma_final = inflection_pt * sqrt(3);
    SIGMA = 0.1;
    %SIGMA = sigma_final/10

    niter = 100;
    %R_est_GCW = GCW_GM(Ind, AdjMat, RijMat, S_vec, SIGMA);
    R_est_GCW = GCW(Ind, AdjMat, RijMat, S_vec);

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
    
    
        s2=sqrt(sum(w(:,2:4).*w(:,2:4),2));
        w(:,1)=2*atan2(s2,w(:,1));
        i=w(:,1)<-pi;  w(i,1)=w(i,1)+2*pi;  i=w(:,1)>=pi;  w(i,1)=w(i,1)-2*pi;
        B=w(:,2:4).*repmat(w(:,1)./s2,[1,3]);
    % Here is an alternative solution for the above 4 lines. This may be
    % marginally faster. But use of this is not recomended as the domain of
    % acos is bounded which may result in truncation error when the solution
    % comes near optima. Usage of atan2 justifies omition of explicit
    % quaternion normalization at every stage.
    %     i=w(:,1)<0;w(i,:)=-w(i,:);
    %     theta2=acos(w(:,1));
    %     B=((w(:,2:4).*repmat((2*theta2./sin(theta2)),[1,3])));
        
        
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

   R_est = R;
   R_init = R_est_GCW;
    
    

Iteration
    
    if params.make_plots
        figure
        tiledlayout(2,2);
        
        nexttile
        plot(svec_errors);
        title('Convergence of Corruption Estimate Vector (SVec, sampled)');
        xlabel('Iteration number');
        ylabel('Average distance to true corruption');
        
        nexttile
        plot(obj_vals);
        title('Convergence of Objective Function (sampled)');
        xlabel('Iteration number');
        ylabel('Value of Objective Function');
        
        nexttile
        plot(MSE_means);
        title('Convergence of Rotation Estimate, Mean (sampled)');
        xlabel('Iteration number');
        ylabel('Mean Error in R estimate (degrees)');
        ylim([0 inf]);
        
        nexttile
        plot(MSE_medians);
        title('Convergence of Rotation Estimate, Median (sampled)');
        xlabel('Iteration number');
        ylabel('Median Error in R estimate (degrees)');
        ylim([0 inf]);
    end

end