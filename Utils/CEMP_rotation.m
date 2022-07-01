tic
n=500;
nsample=50;
p=1;
q=1;
sigma=0;
G = rand(n,n) < p;
G = tril(G,-1);
% generate adjacency matrix
AdjMat = G + G'; 
[Ind_j, Ind_i] = find(G==1);
m = length(Ind_i);

%generate rotation matrices
R_orig = zeros(3,3*n);
for i = 1:n
    [Q,~] = qr(randn(3));
    R_orig(:,(3*i-2):(3*i))=Q;
end

Rij_orig = zeros(3,3*m);
for k = 1:m
    i=Ind_i(k); j=Ind_j(k); 
    Rij_orig(:,(3*k-2):(3*k))=R_orig(:,(3*i-2):(3*i))*((R_orig(:,(3*j-2):(3*j)))');
end
RijMat = Rij_orig;
noiseIndLog = rand(1,m)>=q;
% indices of corrupted edges
corrIndLog = logical(1-noiseIndLog);
noiseInd=find(noiseIndLog);
corrInd=find(corrIndLog);
RijMat(:,[3*noiseInd-2,3*noiseInd-1,3*noiseInd])= ...
RijMat(:,[3*noiseInd-2,3*noiseInd-1,3*noiseInd])+sigma*randn(3,3*length(noiseInd));
for k = noiseInd
    [U, ~, V]= svd(RijMat(:,(3*k-2):(3*k)));
    S0 = diag([1,1,det(U*V')]);
    RijMat(:,(3*k-2):(3*k)) = U*S0*V';
end    

for k = corrInd
    [Q,~] = qr(randn(3));
    RijMat(:,(3*k-2):(3*k))=Q;
end    


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

 disp('Sampling Triangles...')

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

RijMat4d = zeros(3,3,n,n);
% store pairwise directions in 3 by n by n tensor
% construct edge index matrix (for 2d-to-1d index conversion)
for l = 1:m
    i=Ind_i(l);j=Ind_j(l);
    RijMat4d(:,:,i,j)=RijMat(:,(3*l-2):(3*l));
    RijMat4d(:,:,j,i)=(RijMat(:,(3*l-2):(3*l)))';
    IndMat(i,j)=l;
    IndMat(j,i)=l;
end
S0Mat = zeros(nsample,m);
for l=IndPos
    Rij=RijMat(:,(3*l-2):(3*l));
   for k=1:nsample
       Rjk = RijMat4d(:,:,Ind_j(l),CoIndMat(k,l));
       Rki = RijMat4d(:,:,CoIndMat(k,l), Ind_i(l));
       R_cycle = Rij*Rjk*Rki;
       S0Mat(k,l)=abs(acos((trace(R_cycle)-1)/2))/pi;
   end    
end

toc



