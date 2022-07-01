

n=100;
p=1;
q=0.38;
nsample=200;
niter=10;
V = 2*(binornd(1,0.5,1,n)-0.5);
G = rand(n,n) < p;
G = tril(G,-1);
% generate adjacency matrix
AdjMat = G + G'; 
[Ind_j, Ind_i] = find(G==1);
m = length(Ind_i);

rs_gt = V(Ind_i).*V(Ind_j);
% normalize to get the ground truth pairwise directions
rs = rs_gt;
corrInd = rand(1,m)<=q;
rs(corrInd) = -1*rs(corrInd);
GammaMat3d = zeros(3,n,n);
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


% construct edge index matrix (for 2d-to-1d index conversion)
for l = 1:m
i=Ind_i(l);j=Ind_j(l);
rs2d(i,j)=rs(l);
rs2d(j,i)=rs(l);
IndMat(i,j)=l;
IndMat(j,i)=l;
end

Xki = zeros(nsample,m);
Xjk = zeros(nsample,m);
for l = IndPos
Xki(:,l) = rs2d(Ind_i(l),CoIndMat(:,l));
Xjk(:,l) = rs2d(Ind_j(l),CoIndMat(:,l));
end
Xij = ones(nsample,1)*rs;
SMat0 = 0.5-(Xij.*Xjk.*Xki)*0.5;
% AAB formula in matrix form
SVec = mean(SMat0,1);
maxS = max(max(SMat0));
minS = min(min(SMat0));
for iter = 1:niter
% parameter controling the decay rate of reweighting function
tau = pi/(maxS-(maxS-minS)/niter*(iter-1));
Ski = zeros(nsample, m);
Sjk = zeros(nsample, m);
for l = IndPos
    i = Ind_i(l); j=Ind_j(l);
    Ski(:,l) = SVec(IndMat(i,CoIndMat(:,l)));
    Sjk(:,l) = SVec(IndMat(j,CoIndMat(:,l)));
end
%Smax = max(Ski, Sjk);
Smax = Ski.^2+Sjk.^2;
% compute weight matrix (nsample by m)
WeightMat = exp(-tau*Smax);
weightsum = sum(WeightMat,1);
% normalize so that each column sum up to 1
WeightMat = bsxfun(@rdivide,WeightMat,weightsum);
SMat = WeightMat.*SMat0;
% IR-AAB at current iteration
SVec = sum(SMat,1);
end
error = 0.5-0.5*rs_gt.*rs;
plot(SVec, error,'.')

figure;
tpr = zeros(1,1000); % pick 1000 different thresholds
fpr = zeros(1,1000);
bottom = min(SVec);
top = max(SVec);
lin = linspace(bottom, top, 1000);
corrInd = error>1e-4;
for i = 1:1000
    t = lin(i);
    tp = sum((corrInd).*(SVec>t));
    pos = sum(corrInd);
    fp = sum((1-corrInd).*(SVec>t));
    neg = m-pos;
    tpr(i) = tp/pos;
    fpr(i) = fp/neg;
end
plot(fpr,tpr,'-', 'LineWidth',2, 'Color', [0.9,0.1,0.1])
