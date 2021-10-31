rng(3)
tic
n=200;
p=0.5;
q=0.5;
sigma=0.0;
adv = 0;



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
toc


% CoIndMat{l}= triangles sampled that contains l-th edge
% e.g. l=(3,5), then CoIndMat{l}=[2,9,8,...] means that...
% triangles 352, 359, 358,... are sampled


disp('cumind')

tic
Rjk0Mat = zeros(3,3,m_cycle);
Rki0Mat = zeros(3,3,m_cycle);
IJK = zeros(1,m_cycle);
IKJ = zeros(1,m_cycle);
JKI = zeros(1,m_cycle);

IJK_Mat = zeros(n,m_pos);
 
for l = 1:m_pos
    IJ = CoDeg_pos_ind(l);
    i=Ind_i(IJ); j=Ind_j(IJ);
    CoInd_ij= find(AdjMat(:,i).*AdjMat(:,j));
    Ind_ij((cum_ind(l)+1):cum_ind(l+1)) =  IJ;
    Ind_jk((cum_ind(l)+1):cum_ind(l+1)) =  IndMat(j,CoInd_ij);
    Ind_ki((cum_ind(l)+1):cum_ind(l+1)) =  IndMat(CoInd_ij,i);
    Rjk0Mat(:,:,(cum_ind(l)+1):cum_ind(l+1)) =  RijMat4d(:,:,j,CoInd_ij);
    Rki0Mat(:,:,(cum_ind(l)+1):cum_ind(l+1)) =  RijMat4d(:,:,CoInd_ij,i);
    
    IJK((cum_ind(l)+1):cum_ind(l+1)) = CoInd_ij;
    IJK_Mat(1:CoDeg_vec_pos(l),l) =  CoInd_ij;   
end

for l = 1:m_pos
    IJ = CoDeg_pos_ind(l);
    i=Ind_i(IJ); j=Ind_j(IJ);
    IK = CoDeg_pos_ind_long(IndMat(i,IJK((cum_ind(l)+1):cum_ind(l+1))));
    IK_cum = cum_ind(IK);
    [J_ind, ~] = find(IJK_Mat(:,IK)==j);  
    IKJ((cum_ind(l)+1):cum_ind(l+1)) = IK_cum + J_ind;
    
    JK = CoDeg_pos_ind_long(IndMat(j,IJK((cum_ind(l)+1):cum_ind(l+1))));
    JK_cum = cum_ind(JK);
    [I_ind, ~] = find(IJK_Mat(:,JK)==i);  
    JKI((cum_ind(l)+1):cum_ind(l+1)) = JK_cum + I_ind;
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
S_vec = ones(1,m);

toc


wijk = ones(1,m_cycle);
for l=1:m_pos
    IJ = CoDeg_pos_ind(l);
    weight = wijk((cum_ind(l)+1):cum_ind(l+1));
    wijk((cum_ind(l)+1):cum_ind(l+1)) = weight/sum(weight);
    S_vec(IJ) = wijk((cum_ind(l)+1):cum_ind(l+1)) * (S0_long((cum_ind(l)+1):cum_ind(l+1)))';
end


    disp('Initialization completed!')

tic
    disp('Reweighting Procedure Started ...')
    
sum_ikj = zeros(1,m_cycle);
sum_jki = zeros(1,m_cycle);

S_vec_last = S_vec;
%%%%%%%%%%%%%
learning_rate = 0.01;
learning_iters = 500;
rm=1;
proj=1;

for iter = 1:learning_iters
       for l = 1:m_pos % for each edge ij 
           IJ = CoDeg_pos_ind(l);
           i=Ind_i(IJ); j=Ind_j(IJ); 
           sum_ikj((cum_ind(l)+1):cum_ind(l+1)) = sum(wijk(IKJ((cum_ind(l)+1):cum_ind(l+1))));
           sum_jki((cum_ind(l)+1):cum_ind(l+1)) = sum(wijk(JKI((cum_ind(l)+1):cum_ind(l+1))));
       end   
       
       grad_long = S_vec(Ind_jk)+S_vec(Ind_ki)+(sum_ikj+sum_jki).*S0_long;

       for l = 1:m_pos
           IJ = CoDeg_pos_ind(l);
           nsample = CoDeg_vec_pos(l);
           grad = grad_long((cum_ind(l)+1):cum_ind(l+1));  
           nv = ones(1,nsample)/(nsample^0.5);
           if rm==1
                grad = grad - (grad*nv')*nv; % Riemmanian Project 
           end 
           %step_size  = (learning_rate/(2^fix(iter/100)));
           step_size  = learning_rate/1.2;
           w_new = ...
           wijk((cum_ind(l)+1):cum_ind(l+1)) - step_size * grad;
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
if mean(abs(S_vec - S_vec_last)) < 10^(-8)
    break 
end
S_vec_last = S_vec;
fprintf('%d: %f\n',iter,mean(abs(S_vec - ErrVec)))
meanErrors(iter) = mean(abs(S_vec - ErrVec));
end


%writematrix(meanErrors, "Data1228/meanErrsPiecewise.csv");
% subplot(2,2,4)
plot(ErrVec, S_vec, 'b.')
title(strcat('\sigma=', num2str(sigma)) )
xlabel('s_{ij}^*') 
ylabel('s_{ij}(5)') 

fprintf('Max error: %f\n', max(abs(S_vec - ErrVec)))
fprintf('Average error: %f\n',mean(abs(S_vec - ErrVec)))
fprintf('Median error: %f\n', median(abs(S_vec - ErrVec)))