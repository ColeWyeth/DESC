tic
n=500;
nsample=50;
nfold = 100;
p=0.5;
q=0.5;
sigma=0.1;
beta=1;
beta_max=40;
rate=2;
G = rand(n,n) < p;
G = tril(G,-1);
% generate adjacency matrix
AdjMat = G + G'; 
[Ind_j, Ind_i] = find(G==1);
m = length(Ind_i);

%generate rotation matrices
R_orig = zeros(3,3,n);
for i = 1:n
    [Q,~] = qr(randn(3));
    R_orig(:,:,i)=Q;
end

Rij_orig = zeros(3,3,m);
for k = 1:m
    i=Ind_i(k); j=Ind_j(k); 
    Rij_orig(:,:,k)=R_orig(:,:,i)*(R_orig(:,:,j)');
end
RijMat = Rij_orig;
noiseIndLog = rand(1,m)>=q;
% indices of corrupted edges
corrIndLog = logical(1-noiseIndLog);
noiseInd=find(noiseIndLog);
corrInd=find(corrIndLog);
RijMat(:,:,noiseInd)= ...
RijMat(:,:,noiseInd)+sigma*randn(3,3,length(noiseInd));
for k = noiseInd
    [U, ~, V]= svd(RijMat(:,:,k));
    S0 = diag([1,1,det(U*V')]);
    RijMat(:,:,k) = U*S0*V';
end    

for k = corrInd
    [Q,~] = qr(randn(3));
    RijMat(:,:,k)=Q;
end    

toc

tic
disp('triangle sampling')
%compute cycle inconsistency


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


% CoIndMat(:,l)= triangles sampled that contains l-th edge
% e.g. l=(3,5), then CoIndMat(:,l)=[2,9,8,...] means that...
% triangles 352, 359, 358,... are sampled
for l = IndPos
    i = Ind_i(l); j = Ind_j(l);
   CoIndMat(:,l)= datasample(find(AdjMat(:,i).*AdjMat(:,j)), nsample);
end

disp('Triangle Sampling Finished!')

toc

tic
disp('Initializing')

disp('build 4d tensor')



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
toc


disp('assign coInd')

tic
Rki0 = zeros(3,3,m,nsample);
Rjk0 = zeros(3,3,m,nsample);
for l = IndPos
Rki0(:,:,l,:) = RijMat4d(:,:,CoIndMat(:,l), Ind_i(l));
Rjk0(:,:,l,:) = RijMat4d(:,:,Ind_j(l),CoIndMat(:,l));
end
toc

disp('reshape')
tic
Rki0Mat = reshape(Rki0,[3,3,m*nsample]);
Rjk0Mat = reshape(Rjk0,[3,3,m*nsample]);
Rij0Mat = reshape(kron(ones(1,nsample),reshape(RijMat,[3,3*m])), [3,3,m*nsample]);
toc

disp('compute R cycle')
tic
R_cycle0 = zeros(3,3,m*nsample);
R_cycle = zeros(3,3,m*nsample);
for j = 1:3
  R_cycle0 = R_cycle0 + bsxfun(@times,Rij0Mat(:,j,:),Rjk0Mat(j,:,:));
end

for j = 1:3
  R_cycle = R_cycle + bsxfun(@times,R_cycle0(:,j,:),Rki0Mat(j,:,:));
end
toc

disp('S0Mat')
tic
R_trace = (reshape(R_cycle(1,1,:)+R_cycle(2,2,:)+R_cycle(3,3,:), [m,nsample]))';
S0Mat = abs(acos((R_trace-1)./2))/pi;

toc

SVec = mean(S0Mat,1);

    disp('Initialization completed!')

% compute maximal/minimal AAB inconsistency
maxS0 = max(max(S0Mat));
minS0 = min(min(S0Mat));

tic
    disp('Reweighting Procedure Started ...')

beta_max = min(beta_max, 1/minS0);   
iter = 0;
while beta <= beta_max
    iter = iter+1;
    % parameter controling the decay rate of reweighting function
    beta = beta*rate;
    Ski = zeros(nsample, m);
    Sjk = zeros(nsample, m);
    for l = IndPos
        i = Ind_i(l); j=Ind_j(l);
        Ski(:,l) = SVec(IndMat(i,CoIndMat(:,l)));
        Sjk(:,l) = SVec(IndMat(j,CoIndMat(:,l)));
    end
    Smax = Ski+Sjk;
    % compute weight matrix (nsample by m)
    WeightMat = exp(-beta*Smax);
    weightsum = sum(WeightMat,1);
    % normalize so that each column sum up to 1
    WeightMat = bsxfun(@rdivide,WeightMat,weightsum);
    SMat = WeightMat.*S0Mat;
    % IR-AAB at current iteration
    SVec = sum(SMat,1);

        fprintf('Reweighting Iteration %d Completed!\n',iter)   

end

    disp('Completed!')

toc

R_err = zeros(3,3,m);
for j = 1:3
  R_err = R_err + bsxfun(@times,Rij_orig(:,j,:),RijMat(:,j,:));
end


R_err_trace = (reshape(R_err(1,1,:)+R_err(2,2,:)+R_err(3,3,:), [m,1]))';
ErrVec = abs(acos((R_err_trace-1)./2))/pi;


disp('build spanning tree');
tic


SVec2=[SVec; 1:m]';
SVec_sorted = sortrows(SVec2);
Ind_sorted = SVec_sorted(:,2);
min_tol_vec = quantile(SVec,linspace(0,1,nfold+1));
tol_ind = 2;
min_tol=min_tol_vec(tol_ind);

added=zeros(1,n);
R_est = zeros(3,3,n);
R_est(:,:,1)=eye(3);
added(1)=1;
node = 1;
lroot =1;
nodevec = zeros(1,n);
nodevec(1)=1;
l=1;
lempty=1;
while l<n
    node_cand = find((AdjMat(node,:).*(1-added))==1);
    if isempty(node_cand)
        lempty = lempty-1;
        if lempty==0
            lempty = l; tol_ind = tol_ind+1; min_tol = min_tol_vec(tol_ind);
            fprintf('tolerace too low, increase to %f.\n',min_tol);
        end
        node = nodevec(lempty);
        fprintf('next candidate is empty, back to node index %d.\n',lempty);
    else
        edge_cand = IndMat(node,node_cand);
        SVec_cand = SVec2(edge_cand,:);
        [min_val, min_ind]= min(SVec_cand(:,1));
        if min_val>min_tol
           lempty = lempty-1;
           if lempty==0
            lempty = l; tol_ind = tol_ind+1; min_tol = min_tol_vec(tol_ind);
            fprintf('tolerace too low, increase to %f.\n',min_tol);
           end
            node = nodevec(lempty);
            fprintf('next candidates are all corrupted, back to node index %d.\n',lempty);
        else
            edge_next = SVec_cand(min_ind,2);
            i = Ind_i(edge_next); j = Ind_j(edge_next);
            if node == j
                node_next = i;
                R_est(:,:,node_next)=RijMat(:,:,edge_next)*R_est(:,:,node);
            else
                node_next = j;
                R_est(:,:,node_next)=(RijMat(:,:,edge_next))'*R_est(:,:,node);
            end
            added(node_next)=1;
            l=l+1;
            lempty = l;
            nodevec(l)=node_next;
            node = node_next;
        end
    end
    
    
end

toc
%compute error
[R_fit, MSE_rot, R_opt] = GlobalSOdCorrectRight(R_est, R_orig);
MSE_rot



