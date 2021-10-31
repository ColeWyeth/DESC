tic
n=200;
nsample=50;
p=0.5;
q=0.3;
sigma=0.1;
beta=1;
beta_max=20;
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

RijMat1 = permute(RijMat, [2,1,3]);
Ind = [Ind_i Ind_j]';
R_est = AverageSO3Graph(RijMat1,Ind);
%compute error
R_err = zeros(3,3,n);
for j = 1:3
  R_err = R_err + bsxfun(@times,R_orig(:,j,:),R_est(:,j,:));
end


R_err_trace = (reshape(R_err(1,1,:)+R_err(2,2,:)+R_err(3,3,:), [n,1]))';
RErrVec = abs(acos((R_err_trace-1)./2))/pi;
mean(RErrVec)

[R_fit, MSE_rot, R_opt] = GlobalSOdCorrectRight(R_est, R_orig);
MSE_rot
