
MSE_CEMP1=0; 
MSE_Huber_CEMP1=0; 
MSE_IRLS1=0; 
MSE_L121=0; 
MSE_CEMP_L121=0; 



for seed=2020:2029
rng(seed)
tic
n=200;
nsample=50;
p=0.5;
q=0.48;
sigma=1;
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
    Q=randn(3);
    [U, ~, V]= svd(Q);
    S0 = diag([1,1,det(U*V')]);  
    R_orig(:,:,i)=U*S0*V';
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

%rng(3)
R_corr = zeros(3,3,n);

for i = 1:n
    Q=randn(3);
    [U, ~, V]= svd(Q);
    S0 = diag([1,1,det(U*V')]);  
    R_corr(:,:,i)=U*S0*V';
end


for k = corrInd
    %Q=randn(3);
    %[U, ~, V]= svd(Q);
    %S0 = diag([1,1,det(U*V')]);
    %RijMat(:,:,k) = U*S0*V';
    i=Ind_i(k); j=Ind_j(k); 
    Q=R_corr(:,:,i)*(R_corr(:,:,j)')+sigma*randn(3,3);
    [U, ~, V]= svd(Q);
    S0 = diag([1,1,det(U*V')]);
    RijMat(:,:,k) = U*S0*V';
end    

toc

R_err = zeros(3,3,m);
for j = 1:3
  R_err = R_err + bsxfun(@times,Rij_orig(:,j,:),RijMat(:,j,:));
end


R_err_trace = (reshape(R_err(1,1,:)+R_err(2,2,:)+R_err(3,3,:), [m,1]))';
ErrVec = abs(acos((R_err_trace-1)./2))/pi;


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
    IndMat(j,i)=-l;
end



disp('assign coInd')


Rki0 = zeros(3,3,m,nsample);
Rjk0 = zeros(3,3,m,nsample);
for l = IndPos
Rki0(:,:,l,:) = RijMat4d(:,:,CoIndMat(:,l), Ind_i(l));
Rjk0(:,:,l,:) = RijMat4d(:,:,Ind_j(l),CoIndMat(:,l));
end


disp('reshape')

Rki0Mat = reshape(Rki0,[3,3,m*nsample]);
Rjk0Mat = reshape(Rjk0,[3,3,m*nsample]);
Rij0Mat = reshape(kron(ones(1,nsample),reshape(RijMat,[3,3*m])), [3,3,m*nsample]);


disp('compute R cycle')

R_cycle0 = zeros(3,3,m*nsample);
R_cycle = zeros(3,3,m*nsample);
for j = 1:3
  R_cycle0 = R_cycle0 + bsxfun(@times,Rij0Mat(:,j,:),Rjk0Mat(j,:,:));
end

for j = 1:3
  R_cycle = R_cycle + bsxfun(@times,R_cycle0(:,j,:),Rki0Mat(j,:,:));
end


disp('S0Mat')

R_trace = (reshape(R_cycle(1,1,:)+R_cycle(2,2,:)+R_cycle(3,3,:), [m,nsample]))';
S0Mat = abs(acos((R_trace-1)./2))/pi;



SVec = mean(S0Mat,1);

    disp('Initialization completed!')

% compute maximal/minimal AAB inconsistency
maxS0 = max(max(S0Mat));
minS0 = min(min(S0Mat));


    disp('Reweighting Procedure Started ...')

beta_max = min(beta_max, 1/minS0);   
iter = 0;
while beta <= beta_max/rate
    iter = iter+1;
    % parameter controling the decay rate of reweighting function
    beta = beta*rate;
    Ski = zeros(nsample, m);
    Sjk = zeros(nsample, m);
    for l = IndPos
        i = Ind_i(l); j=Ind_j(l);
        Ski(:,l) = SVec(abs(IndMat(i,CoIndMat(:,l))));
        Sjk(:,l) = SVec(abs(IndMat(j,CoIndMat(:,l))));
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




disp('build spanning tree');



Indfull_i = [Ind_i;Ind_j];
Indfull_j = [Ind_j;Ind_i];
Sfull = [SVec, SVec];
DG = sparse(Indfull_i,Indfull_j,Sfull);
[tree,~]=graphminspantree(DG);
[T1, T2, ~] = find(tree);
sizetree=size(T1,1);
AdjTree = zeros(n);
for k=1:sizetree
    i=T1(k); j=T2(k);
    AdjTree(i,j)=1;
    AdjTree(j,i)=1;
end
%[~, rootnodes]=max(sum(AdjTree));
rootnodes = 1;
added=zeros(1,n);
R_est = zeros(3,3,n);
R_est(:,:,rootnodes)=eye(3);
added(rootnodes)=1;
newroots = [];
while sum(added)<n
    for node_root = rootnodes
        leaves = find((AdjTree(node_root,:).*(1-added))==1);
        newroots = [newroots, leaves];
        for node_leaf=leaves
            edge_leaf = IndMat(node_leaf,node_root);
            if edge_leaf>0
                R_est(:,:,node_leaf)=RijMat(:,:,abs(edge_leaf))*R_est(:,:,node_root);
            else
                R_est(:,:,node_leaf)=(RijMat(:,:,abs(edge_leaf)))'*R_est(:,:,node_root);
            end
            added(node_leaf)=1;
        end
    end
    rootnodes = newroots;
end




RijMat1 = permute(RijMat, [2,1,3]);
Ind = [Ind_i Ind_j]';
%R_est_L1 = BoxMedianSO3Graph(RijMat1,Ind);
%R_est_L1_CEMP = BoxMedianSO3Graph(RijMat1,Ind, R_est);
R_est_Huber = RobustMeanSO3Graph(RijMat1,Ind);
R_est_Huber_CEMP = RobustMeanSO3Graph(RijMat1,Ind, [], R_est);

R_est_IRLS = AverageSO3Graph(RijMat1,Ind);
%R_est_IRLS_CEMP = AverageSO3Graph(RijMat1,Ind, 'Rinit', R_est);


R_est_L12 = AverageL1(RijMat1,Ind);
R_est_L12_CEMP = L1(RijMat1,Ind, [], R_est);

%compute error
[~, MSE_CEMP, ~, ~] = GlobalSOdCorrectRight(R_est, R_orig);
%[~, MSE_L1, ~] = GlobalSOdCorrectRight(R_est_L1, R_orig);
%[~, MSE_L1_CEMP, ~] = GlobalSOdCorrectRight(R_est_L1_CEMP, R_orig);
%[~, MSE_Huber, ~] = GlobalSOdCorrectRight(R_est_Huber, R_orig);
[~, MSE_Huber_CEMP, ~, ~] = GlobalSOdCorrectRight(R_est_Huber_CEMP, R_orig);
[~, MSE_IRLS, ~] = GlobalSOdCorrectRight(R_est_IRLS, R_orig);
%[~, MSE_CEMP_IRLS, ~] = GlobalSOdCorrectRight(R_est_IRLS_CEMP, R_orig);
[~, MSE_L12, ~, ~] = GlobalSOdCorrectRight(R_est_L12, R_orig);
[~, MSE_CEMP_L12, ~, ~] = GlobalSOdCorrectRight(R_est_L12_CEMP, R_orig);

MSE_CEMP1=MSE_CEMP1+MSE_CEMP; 
MSE_Huber_CEMP1=MSE_Huber_CEMP1+MSE_Huber_CEMP; 
MSE_IRLS1=MSE_IRLS1+MSE_IRLS; 
MSE_L121=MSE_L121+MSE_L12; 
MSE_CEMP_L121=MSE_CEMP_L121+MSE_CEMP_L12; 

end

MSE_CEMP1=MSE_CEMP1/(seed-2019); 
MSE_Huber_CEMP1=MSE_Huber_CEMP1/(seed-2019); 
MSE_IRLS1=MSE_IRLS1/(seed-2019); 
MSE_L121=MSE_L121/(seed-2019); 
MSE_CEMP_L121=MSE_CEMP_L121/(seed-2019); 

fprintf('CEMP %f\n',MSE_CEMP1); 
%fprintf('L1 %f\n',MSE_L1); 
%fprintf('L1+CEMP %f\n',MSE_L1_CEMP); 
%fprintf('Huber %f\n',MSE_Huber); 
fprintf('Huber+CEMP %f\n',MSE_Huber_CEMP1); 
fprintf('IRLS %f\n',MSE_IRLS1); 
%fprintf('CEMP+IRLS %f\n',MSE_CEMP_IRLS); 
fprintf('L1/2 %f\n',MSE_L121); 
fprintf('CEMP+L1/2 %f\n',MSE_CEMP_L121); 
[MSE_CEMP1,MSE_Huber_CEMP1,MSE_IRLS1,MSE_L121,MSE_CEMP_L121]



