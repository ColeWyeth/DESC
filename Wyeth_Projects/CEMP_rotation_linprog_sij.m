rng(3)
tic
n=50;
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

[S0Mat, CoIndMat, IndMat, m, Ind_i, Ind_j, ErrVec] = Initialization(n, nsample, p, q, sigma, adversarial);

% Linear Programming Code Begins
A = sparse(nsample*2*m, m);
b = zeros(1, nsample*2*m);
    
for l = 1:m
    i = Ind_i(l);
    j = Ind_j(l);
    if mod(l, 1000) == 0
        disp('next 1000 done');
    end
    offset = nsample*2*(l-1); % previously filled
    for kInd = 1:nsample
        % Then CoIndMat(kInd,l) is the vertex k
        k = CoIndMat(kInd, l);
        A(offset + 2*kInd - 1, l) = 1;
        A(offset + 2*kInd -1, IndMat(i,k)) = -1;
        A(offset + 2*kInd - 1, IndMat(j,k)) = -1;
        b(offset + 2*kInd - 1) = S0Mat(kInd, l); %possibly reversed
        
        A(offset + 2*kInd, l) = -1; 
        A(offset + 2*kInd, IndMat(i,k)) = -1;
        A(offset + 2*kInd, IndMat(j,k)) = -1;
        b(offset + 2*kInd) = -S0Mat(kInd, l); %possibly reversed
    end
end

f = ones(1,m);
lb = zeros(1,m);
ub = ones(1,m); 

SVec = linprog(f, A, b, [],[], lb, ub)

% subplot(2,2,4)
plot(ErrVec, SVec, 'b.')
title(strcat('\sigma=', num2str(sigma)) )
xlabel('s_{ij}^*') 
ylabel('s_{ij}(5)') 

fprintf('Max error: %f\n', max(abs(SVec' - ErrVec)))
fprintf('Average error %f\n',mean(abs(SVec' - ErrVec)))
fprintf('Median error%f\n', median(abs(SVec' - ErrVec)))