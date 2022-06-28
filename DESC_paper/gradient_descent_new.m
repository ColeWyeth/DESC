rng(3)
tic
n=100;
nsample=50;
p=0.5;
q=0.5;
sigma=0.0;
beta=5;
beta_max=40;
rate=2;
adversarial = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Change Parameters Below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
learning_rate = 1;
learning_iters = 20;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[S0Mat, CoIndMat, IndMat, m, Ind_i, Ind_j, ErrVec] = Initialization(n, nsample, p, q, sigma, adversarial);

wijk = ones(nsample, m)/nsample; % initialize wijk, for this algorithm m x nsample is convenient 
wijk_new = wijk;

% Uncomment the lines below to initialize with CEMP 
% beta_max = min(beta_max, 1/minS0);   
% iter = 0;
% while beta <= beta_max/rate
%     iter = iter+1;
%     % parameter controling the decay rate of reweighting function
%     beta = beta*rate;
%     Ski = zeros(nsample, m);
%     Sjk = zeros(nsample, m);
%     for l = IndPos
%         i = Ind_i(l); j=Ind_j(l);
%         Ski(:,l) = SVec(IndMat(i,CoIndMat(:,l)));
%         Sjk(:,l) = SVec(IndMat(j,CoIndMat(:,l)));
%     end
%     
%     Smax = Ski+Sjk;
%     % compute weight matrix (nsample by m)
%     wijk = exp(-beta*Smax);
%     weightsum = sum(wijk,1);
%     % normalize so that each column sum up to 1
%     wijk = bsxfun(@rdivide,wijk,weightsum);
%     SMat = wijk.*S0Mat;
%     % IR-AAB at current iteration
%     SVec = sum(SMat,1);
% 
%         fprintf('Reweighting Iteration %d Completed!\n',iter)   
% 
% end

startEdge = @(l) nsample*(l-1) + 1;
stopEdge = @(l) nsample*l;
meanErrors = zeros(1,learning_iters); 

grad = zeros(1,nsample);           % This will hold the gradient in wij,k for some ij  
n = ones(1,nsample)/(nsample^0.5);

for iter = 1:learning_iters
   for l = 1:m % for each edge ij 
       i = Ind_i(l);
       j = Ind_j(l); 
       for indK = 1:nsample % for each cycle ijk
           k = CoIndMat(indK, l); 
           grad(indK) = wijk(:,IndMat(i,k))'*S0Mat(:,IndMat(i,k)) + wijk(:,IndMat(j,k))'*S0Mat(:,IndMat(j,k)); 
           a = find(IndMat(i,:) ~= 0); % We want cycles starting at a and going through j to i 
           sumji = sum(sum(wijk(:, IndMat(i,a)).*(CoIndMat(:,IndMat(i,a)) == j)));
           a = find(IndMat(j,:) ~= 0); % We want cycles starting at a and going through i to j
           sumij = sum(sum(wijk(:, IndMat(j, a)).*(CoIndMat(:,IndMat(j,a)) == j)));
           grad(indK) = grad(indK) + S0Mat(indK, l) * (sumij + sumji); 
       end
       grad = grad - (grad*n')*n; % Project 
       wijk(:,l) = wijk(:,l) - (learning_rate/(2^fix(iter/50))) * grad';
       
       % Yunpeng normalization 
       w = sort(wijk(:,l)); 
       Ti = 0; 
       for i = 1:length(wijk)
           if sum(w(i:end)-w(i)) < 1
               Ti = i;  
               break 
           end
       end
       T = w(Ti) - (1 - sum(w(Ti:end)-w(Ti)))/length(w(Ti:end)); 
       wijk(:,l) = max(wijk(:,l) - T, 0);
   end
   %wijk = SimplexProj(wijk_new')';
   
   pijk = reshape(wijk, 1, m*nsample);
% Construct SVec from wVec
SVec1 = zeros(1, m);
for l = 1:m
    SVec1(l) = pijk(startEdge(l):stopEdge(l))*S0Mat(:,l);
end

fprintf('%d: %f\n',iter,mean(abs(SVec1 - ErrVec)))
meanErrors(iter) = mean(abs(SVec1 - ErrVec));
end

wijk = reshape(wijk, 1, m*nsample);
% Construct SVec from wVec
SVec = zeros(1, m);
for l = 1:m
    SVec(l) = wijk(startEdge(l):stopEdge(l))*S0Mat(:,l);
end

%writematrix(meanErrors, "Data1228/meanErrsPiecewise.csv");
% subplot(2,2,4)
plot(ErrVec, SVec, 'b.')
title(strcat('\sigma=', num2str(sigma)) )
xlabel('s_{ij}^*') 
ylabel('s_{ij}(5)') 

fprintf('Max error: %f\n', max(abs(SVec - ErrVec)))
fprintf('Average error: %f\n',mean(abs(SVec - ErrVec)))
fprintf('Median error: %f\n', median(abs(SVec - ErrVec)))