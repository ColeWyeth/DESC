rng(3)
tic
n=50;
nsample=25;
p=1;
q=0.3;
sigma=0.0;
rate=2;
adversarial = 0;
iterations = 10;


[S0Mat, CoIndMat, IndMat, m, Ind_i, Ind_j, ErrVec] = Initialization(n, nsample, p, q, sigma, adversarial);

Aeq = ones(1, nsample);
beq = ones(1); % wijk for each ij sums to 1
% Now we're minimizing the sum over wijk 
% so the matrix must have nsample rows for each 
% edge, used to compute sij from S0Mat(:,ij) 
H = ones(nsample, nsample);
f = zeros(1,nsample);
lb = zeros(1,nsample);
ub = ones(1,nsample);
wijk = ones(1, nsample*m)/nsample; % initialize wijk in row form, steps of nsample 
mu = ones(1,m);
var_ij = ones(1,m); 

for it = 1:iterations
    % First implementation will just follow pseudo-code
    % first calculate mu and sigma_ij
    for l = 1:m % for each edge ij 
        startl = nsample*(l-1) + 1;
        stopl = nsample*l;
        mu(l) = wijk(startl:stopl)*S0Mat(:,l);
        var_ij(l) = wijk(startl:stopl)*(S0Mat(:,l).^2) - mu(l)^2; %sigma^2 = E(X^2) - E(X)^2
    end
    for l = 1:m
        i = Ind_i(l);
        j = Ind_j(l);
        % Now we can do quadratic programming for each edge :0
        for indK1 = 1:nsample
            for indK2 = 1:nsample 
                k1 = CoIndMat(indK1, l);
                k2 = CoIndMat(indK2, l);
                H(indK1, indK2) = (mu(IndMat(i,k1)) + mu(IndMat(j,k1))) * (mu(IndMat(i,k2)) + mu(IndMat(j,k2)));
                if indK1 == indK2
                    H(indK1,indK2) = H(indK1,indK2) + 0.25*(var_ij(IndMat(i,k1)) + var_ij(IndMat(j,k1))); 
                end
            end
        end
        startl = nsample*(l-1) + 1;
        stopl = nsample*l;
        wijk(startl:stopl) = quadprog(H,f,[],[],Aeq,beq,lb,ub);
    end
    
end

% Construct SVec from wVec
SVec = zeros(1, m);
for l = 1:m
    SVec(l) = wijk(startl:stopl)*S0Mat(:,l);
end


% subplot(2,2,4)
plot(ErrVec, SVec, 'b.')
title(strcat('\sigma=', num2str(sigma)) )
xlabel('s_{ij}^*') 
ylabel('s_{ij}(5)') 

fprintf('Max error: %f\n', max(abs(SVec - ErrVec)))
fprintf('Average error %f\n',mean(abs(SVec - ErrVec)))
fprintf('Median error%f\n', median(abs(SVec - ErrVec)))