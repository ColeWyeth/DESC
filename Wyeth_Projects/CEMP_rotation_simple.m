rng(3)
tic
n=100;
nsample=25;
p=0.5;
q=0.5;
sigma=0.0;
beta=1;
beta_max=40;
rate=2;
adversarial = 0;


iternum=5;
tol=0.1;

[S0Mat, CoIndMat, IndMat, m, Ind_i, Ind_j, ErrVec, IndPos] = Initialization(n, nsample, p, q, sigma, adversarial);


WeightMat = ones(nsample, m); 
weightsum = sum(WeightMat,1);
WeightMat = WeightMat ./ weightsum;


for iter=1:iternum
     
    W = sum(WeightMat .* (S0Mat <= tol), 1); % Chance that ij is has 0 corruption 
                                        
    for l = IndPos
        i = Ind_i(l); j=Ind_j(l);
        ks = CoIndMat(:,l); 
        WeightMat(:,l) = W(IndMat(i, ks)) .* W(IndMat(ks,j));
    end
    
    weightsum = sum(WeightMat,1);
    WeightMat(:,weightsum==0)=1;
    weightsum = sum(WeightMat,1);
   % weightsum_rep = repmat(weightsum,[nsample,1]);
    % normalize so that each column sum up to 1
    WeightMat = bsxfun(@rdivide,WeightMat,weightsum);
   
        
    SMat = WeightMat.*S0Mat;
    % IR-AAB at current iteration
    SVec = sum(SMat,1);

        fprintf('Reweighting Iteration %d Completed!\n',iter)   

end

    disp('Completed!')

toc


% subplot(2,2,4)
plot(ErrVec, SVec, 'b.')
title(strcat('\sigma=', num2str(sigma)) )
xlabel('s_{ij}^*') 
ylabel('s_{ij}(5)') 

fprintf('Max error: %f\n', max(abs(SVec - ErrVec)))
fprintf('Average error: %f\n',mean(abs(SVec - ErrVec)))
fprintf('Median error: %f\n', median(abs(SVec - ErrVec)))