rng(2020);

nsample = 50;
beta = 1;
beta_max = 40;
rate = 2;
AdjMat=data.AdjMat;
Hmat=data.Hmat;
G_gt=data.G_gt;
R_orig=data.R;
Focal=data.Focal_gt;
selected = (Focal>10^(-5));
AdjMat = AdjMat(selected,selected);
Hmat = Hmat(logical(kron(selected,ones(3,1))),logical(kron(selected,ones(3,1))));
G_gt = G_gt(logical(kron(selected,ones(3,1))),logical(kron(selected,ones(3,1))));
R_orig = R_orig(:,:,selected);

[S_k,C_k] = graphconncomp(sparse(AdjMat));
compSizes_k = zeros(S_k,1);
for i = 1:S_k
    compSizes_k(i) = sum(C_k == i);
end
[~, i_Csize] = max(compSizes_k);
selected = (C_k == i_Csize(1));

AdjMat = AdjMat(selected,selected);
Hmat = Hmat(logical(kron(selected,ones(3,1))),logical(kron(selected,ones(3,1))));
G_gt = G_gt(logical(kron(selected,ones(3,1))),logical(kron(selected,ones(3,1))));
R_orig = R_orig(:,:,selected);


[Ind_j, Ind_i] = find(tril(AdjMat,-1));
n = size(AdjMat,1);
m = size(Ind_i,1);
RijMat = zeros(3,3,m);
for k=1:m
    i = Ind_i(k); j = Ind_j(k);
    RijMat(:,:,k) = Hmat((3*i-2):(3*i), (3*j-2):(3*j));
end

Rij_orig = zeros(3,3,m);
for k = 1:m
    i=Ind_i(k); j=Ind_j(k); 
    Rij_orig(:,:,k)=R_orig(:,:,i)*(R_orig(:,:,j)');
end

R_err = zeros(3,3,m);
for j = 1:3
  R_err = R_err + bsxfun(@times,Rij_orig(:,j,:),RijMat(:,j,:));
end


R_err_trace = (reshape(R_err(1,1,:)+R_err(2,2,:)+R_err(3,3,:), [m,1]))';
ErrVec = abs(acos((R_err_trace-1)./2))/pi;
disp('triangle sampling')
%compute cycle inconsistency

tic

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
IndPosbin = zeros(1,n);
IndPosbin(IndPos)=1;

% CoIndMat(:,l)= triangles sampled that contains l-th edge
% e.g. l=(3,5), then CoIndMat(:,l)=[2,9,8,...] means that...
% triangles 352, 359, 358,... are sampled

CoIndMat = zeros(nsample,m);
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
SVec(~IndPosbin)=1;

    disp('Initialization completed!')

% compute maximal/minimal AAB inconsistency
maxS0 = max(max(S0Mat));
minS0 = min(min(S0Mat));


    disp('Reweighting Procedure Started ...')

%beta_max = min(beta_max, 1/minS0);   
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
    fprintf('Reweighting Iteration %d Completed!\n',iter);
    SVec(~IndPosbin)=1;

end

    disp('Completed!')



disp('build spanning tree');



Indfull_i = [Ind_i;Ind_j];
Indfull_j = [Ind_j;Ind_i];
Sfull = [SVec, SVec];
%Sfull = [exp(SVec.^2), exp(SVec.^2)];
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




[~, MSE_mean,MSE_median, ~] = GlobalSOdCorrectRight(R_est, R_orig);


fprintf('CEMP_IRLS_new:  mean %f median %f\n',MSE_mean, MSE_median); 


nbin=100;
hs_value = histcounts(SVec,0:(1/nbin):1);
h_diff = zeros(0,nbin);
[~, hs_peak] = max(hs_value);
for i=(hs_peak+1):nbin
    h_diff(i) = hs_value(i)-hs_value(i-1);
end
[~,cut_ind] = min(h_diff);

cutoff = cut_ind/nbin;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This implementation is based on the paper 
% "Efficient and Robust Large-Scale Rotation Averaging." by
% Avishek Chatterjee, Venu Madhav Govindu.
%
% This code robustly performs iteratively reweighted least square relative rotation averaging
%
% function [R] = RobustMeanSO3Graph(RR,I,SIGMA,[Rinit],[maxIters])
% INPUT:        RR = 'm' number of 3 X 3 Relative Rotation Matrices (R_ij) 
%                    stacked as a 3 X 3 X m Matrix
%                    OR
%                    'm' number of 4 X 1 Relative Quaternions (R_ij) 
%                    stacked as a 4 X m  Matrix
%                I = Index matrix (ij) of size (2 X m) such that RR(:,:,p)
%                    (OR RR(:,p) for quaternion representation)  is
%                    the relative rotation from R(:,:,I(1,p)) to R(:,:,I(2,p))
%                    (OR R(:,I(1,p)) and  R(:,I(2,p)) for quaternion representation)
%            SIGMA = Sigma value for M-Estimation in degree (5 degree is preferred)
%                    Default is 5 degree. Put [] for default.
%            Rinit = Optional initial guess. 
%                    Put [] to automatically comput Rinit from spanning tree
%         maxIters = Maximum number of iterations. Default 100
%
% OUTPUT:       R  = 'n' number of 3 X 3 Absolute Rotation matrices stacked as
%                     a  3 X 3 X n Matrix 
%                     OR
%                     'n' number of 4 X 1 Relative Quaternions (R_ij) 
%                     stacked as a 4 X n  Matrix
%
% IMPORTANT NOTES:
% The underlying model or equation is assumed to be: 
% X'=R*X; Rij=Rj*inv(Ri) i.e. camera centered coordinate system is used
% and NOT the geocentered coordinate for which the underlying equations are
% X'=inv(R)*X; Rij=inv(Ri)*Rj. 
% To use geocentered coordinate please transpose the rotations or change
% the sign of the scalar term of the quaternions before feeding into the
% code and also after getting the result.
%
% Feeding of not connected graph is not recomended.
%
% This code is able to handle inputs in both Rotation matrix as well as
% quaternion format. The Format of output is same as that of the input.
%
% Programmer: AVISHEK CHATTERJEE
%             PhD Student (S. R. No. 04-03-05-10-12-11-1-08692)
%             Learning System and Multimedia Lab
%             Dept. of Electrical Engineering
%             INDIAN INSTITUTE OF SCIENCE
%
% Dated:  April 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
changeThreshold=1e-3;
tau = 32;
tau_max = 32;
rate_tau = 1.5;
RR = permute(RijMat, [2,1,3]);
I = [Ind_i Ind_j]';
Rinit = R_est;
SIGMA = 5;
maxIters = 200;
lam = 1;
delt = 1e-16;
quant_ratio = 1;
%quant_ratio_min = sum(SVec<=cutoff)/length(SVec);
quant_ratio_min = 0.8;
thresh = quantile(SVec,quant_ratio);
top_threshold = 1e4;
right_threshold = 1e-4;
%Rinit = zeros(3,3,n);
%Rinit(1,1,:)=1;
%Rinit(2,2,:)=1;
%Rinit(3,3,:)=1;





SIGMA=SIGMA*pi/180;% Degree to radian

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

Weights = (1./(SVec.^0.75)');

%Weights = ((SVec<0.005)'); Weights(Weights>1e8)=1e8;

%Weights = sqrt(exp(-50*SVec.^2))'.*((SVec<thresh)'); Weights(Weights>1e4)=1e4;
Weights(Weights>top_threshold)= top_threshold;
Weights(SVec>thresh)=right_threshold; 

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


[~, MSE_mean,MSE_median, ~] = GlobalSOdCorrectRight(R, R_orig);


fprintf('CEMP_IRLS_new:  mean %f median %f\n',MSE_mean, MSE_median); 

    
    E=(Amatrix*W(2:end,2:4)-B); 
    residual_standard = sqrt(sum(E.^2,2))/pi;
    
    Ski = zeros(nsample, m);
    Sjk = zeros(nsample, m);
    for l = IndPos
        i = Ind_i(l); j=Ind_j(l);
        Ski(:,l) = residual_standard(abs(IndMat(i,CoIndMat(:,l))));
        Sjk(:,l) = residual_standard(abs(IndMat(j,CoIndMat(:,l))));
    end
    Smax = Ski+Sjk;
    % compute weight matrix (nsample by m)
    WeightMat = exp(-tau*Smax);
    %WeightMat = 1./(Smax.^1.5+delt);
   % WeightMat = (Smax<1/tau1)+delt;
    weightsum = sum(WeightMat,1);
    % normalize so that each column sum up to 1
    WeightMat = bsxfun(@rdivide,WeightMat,weightsum);
    EMat = WeightMat.*S0Mat;
    % IR-AAB at current iteration
    EVec = (sum(EMat,1))';
    ESVec = (1-lam)*residual_standard + lam * EVec;
    %ESVec = max(residual_standard, EVec);
    %Weights = sqrt(exp(-tau*(SVec'.^2)));
    %Weights = 1./(ESVec+delt);
    %Weights = (ESVec<1/tau);
    %Weights = 1./(ESVec.^0.75); Weights(Weights>1e8)=1e8;
    Weights = (1./(ESVec.^0.75));
    %Weights = ((SVec<0.005)'); Weplights(Weights>1e8)=1e8;
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
   
    tau = min(rate_tau*tau, tau_max);
end
toc

R=zeros(3,3,N);
for i=1:size(Q,1)
    R(:,:,i)=q2R(Q(i,:));
end


if(Iteration>=maxIters);disp('Max iterations reached');end


[~, MSE_mean,MSE_median, ~] = GlobalSOdCorrectRight(R, R_orig);


fprintf('CEMP_IRLS_new:  mean %f median %f\n',MSE_mean, MSE_median); 
Iteration

