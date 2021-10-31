rng(3)
tic
n=200;
p=1;
q=0.49;
sigma=0;
beta=1;
beta_max=40;
rate=1.2;
adv = 1;
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

rng(3)
R_corr = zeros(3,3,n);

for i = 1:n
    Q=randn(3);
    [U, ~, V]= svd(Q);
    S0 = diag([1,1,det(U*V')]);  
    R_corr(:,:,i)=U*S0*V';
end

if adv==0
    for k = corrInd
        Q=randn(3);
        [U, ~, V]= svd(Q);
        S0 = diag([1,1,det(U*V')]);
        RijMat(:,:,k) = U*S0*V';
    end    
else 
    for k = corrInd
        i=Ind_i(k); j=Ind_j(k); 
        Q=R_corr(:,:,i)*(R_corr(:,:,j)')+sigma*randn(3,3);
        [U, ~, V]= svd(Q);
        S0 = diag([1,1,det(U*V')]);
        RijMat(:,:,k) = U*S0*V';
    end  
end
toc

R_err = zeros(3,3,m);
for j = 1:3
  R_err = R_err + bsxfun(@times,Rij_orig(:,j,:),RijMat(:,j,:));
end


R_err_trace = (reshape(R_err(1,1,:)+R_err(2,2,:)+R_err(3,3,:), [m,1]))';
ErrVec = abs(acos((R_err_trace-1)./2))/pi;
% histogram(ErrVec);


disp('Codegree')
tic
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
CoDeg_zero_ind = find(CoDeg_vec<0);
cum_ind = [0;cumsum(CoDeg_vec_pos)];
m_pos = length(CoDeg_pos_ind);
m_cycle = cum_ind(end);

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
toc


% CoIndMat{l}= triangles sampled that contains l-th edge
% e.g. l=(3,5), then CoIndMat{l}=[2,9,8,...] means that...
% triangles 352, 359, 358,... are sampled


disp('cumind')

tic
Rjk0Mat = zeros(3,3,m_cycle);
Rki0Mat = zeros(3,3,m_cycle);

for l = 1:m_pos
    IJ = CoDeg_pos_ind(l);
    i=Ind_i(IJ); j=Ind_j(IJ);
    CoInd_ij= find(AdjMat(:,i).*AdjMat(:,j));
    Ind_ij((cum_ind(l)+1):cum_ind(l+1)) =  IJ;
    Ind_jk((cum_ind(l)+1):cum_ind(l+1)) =  IndMat(j,CoInd_ij);
    Ind_ki((cum_ind(l)+1):cum_ind(l+1)) =  IndMat(CoInd_ij,i);
    Rjk0Mat(:,:,(cum_ind(l)+1):cum_ind(l+1)) =  RijMat4d(:,:,j,CoInd_ij);
    Rki0Mat(:,:,(cum_ind(l)+1):cum_ind(l+1)) =  RijMat4d(:,:,CoInd_ij,i);
end
toc
tic
Rij0Mat = RijMat(:,:,Ind_ij);
toc


disp('compute R cycle')
tic
R_cycle0 = zeros(3,3,m_cycle);
R_cycle = zeros(3,3,m_cycle);
for j = 1:3
  R_cycle0 = R_cycle0 + bsxfun(@times,Rij0Mat(:,j,:),Rjk0Mat(j,:,:));
end

for j = 1:3
  R_cycle = R_cycle + bsxfun(@times,R_cycle0(:,j,:),Rki0Mat(j,:,:));
end
toc

disp('S0Mat')
tic
R_trace = reshape(R_cycle(1,1,:)+R_cycle(2,2,:)+R_cycle(3,3,:),[1,m_cycle]);
S0_long = abs(acos((R_trace-1)./2))/pi;
S0_vec = ones(1,m);

toc


Weight_vec = ones(1,m_cycle);
S0_weight = S0_long.*Weight_vec;

for l=1:m_pos
    IJ = CoDeg_pos_ind(l);
    S0_vec(IJ) = sum(S0_weight((cum_ind(l)+1):cum_ind(l+1)))/sum(Weight_vec((cum_ind(l)+1):cum_ind(l+1)));
end

    disp('Initialization completed!')

tic
    disp('Reweighting Procedure Started ...')

iter = 0;

SVec = S0_vec;
while beta <= beta_max/rate
     err = mean(abs(SVec-ErrVec));
    fprintf('Reweighting Iteration %d Completed! err = %f \n',iter, err)   
    iter = iter+1;
    % parameter controling the decay rate of reweighting function
    beta = beta*rate;
    Sjk = SVec(Ind_jk);
    Ski = SVec(Ind_ki);
    S_sum = Ski+Sjk;
    % compute weight matrix (nsample by m)
    Weight_vec = exp(-beta*S_sum);
    S0_weight = S0_long.*Weight_vec;

    for l=1:m_pos
        IJ = CoDeg_pos_ind(l);
        SVec(IJ) = sum(S0_weight((cum_ind(l)+1):cum_ind(l+1)))/sum(Weight_vec((cum_ind(l)+1):cum_ind(l+1)));
    end
    
    err = mean(abs(SVec-ErrVec));

        
    
end

    disp('Completed!')

toc

mean(abs(SVec-ErrVec))


%subplot(2,2,4)
plot(ErrVec, SVec, 'b.')
title('\sigma=0.2')
xlabel('s_{ij}^*') 
ylabel('s_{ij}(5)') 
