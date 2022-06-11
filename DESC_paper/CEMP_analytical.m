rng(3)
tic
n=100;
nsample=50;
p=0.5;
q=0.5;
sigma=0.0;
adversarial = 0;
iterations = 10;
c = 1; % Constant factor on variance

[S0Mat, CoIndMat, IndMat, m, Ind_i, Ind_j, ErrVec] = Initialization(n, nsample, p, q, sigma, adversarial);


% Now we optimize E(Y) + cVar(Y) 
wijk = ones(1, nsample*m)/nsample; % initialize wijk in row form, steps of nsample 
mu = ones(1,m);
var_ij = ones(1,m); 
D = ones(1,nsample); % since the matrix is diagonal we only need an array 
b = ones(1,nsample); 

startEdge = @(l) nsample*(l-1) + 1;
stopEdge = @(l) nsample*l;

objective = @(w_k, D_kk, b_k) b_k .* w_k + c * D_kk .* (w_k.^2); % The function we want to minimize
sol = zeros(1, nsample); 

for it = 1:iterations
    % first calculate mu and sigma_ij
    for l = 1:m % for each edge ij 
        mu(l) = wijk(startEdge(l):stopEdge(l))*S0Mat(:,l);
        var_ij(l) = wijk(startEdge(l):stopEdge(l))*(S0Mat(:,l).^2) - mu(l)^2; %sigma^2 = E(X^2) - E(X)^2
    end
    for l = 1:m
        % Now analytically find the optimum for each edge 
        i = Ind_i(l);
        j = Ind_j(l);
        % Now we construct Dkk and bk 
        for indK = 1:nsample
            k = CoIndMat(indK, l);
            b(indK) = mu(IndMat(i,k))^2 + mu(IndMat(j,k))^2;
            D(indK) = (var_ij(IndMat(i,k)) + var_ij(IndMat(j,k))); 
            D(D < 10^-8) = 10^-8;
        end
        gammaChoices = -sort(b(D~=0)); % Test discontinuous gamma values, decreasing
        % now solve for gamma using normalization 
        gammaLast = 0; 
        gammaNext = 0; 
        
        normalFunc =  @(g)sum(max(0,-(b(D~=0)+g)./(2*c*D(D~=0))))-1;% For proper gamma normalization makes this 0 
        
        for kInd = 1:length(gammaChoices) 
           gammaLast = gammaNext;
           gammaNext = gammaChoices(kInd); 
           if normalFunc(gammaNext) >= 0
               break; 
           end
        end
        slope = (normalFunc(gammaNext) - normalFunc(gammaLast))/(gammaNext - gammaLast);
        gamma = -normalFunc(gammaLast)/slope + gammaLast; 
        sol = zeros(1,nsample); 
        sol(D ~= 0) = max(0,-(b(D~=0)+gamma)./(2*c*D(D~=0)));
        % Alternative solution, if there are zeros we can take the better one  
%        bx = min(b(D == 10^-8));
%         if length(bx) > 0
%             x = find(b == bx, 1);
%             altsol = zeros(1,nsample);
%             altsol(x) = 1.0; 
%             if objective(altsol, D, b) < objective(sol, D, b)
%                 display("Found better alternative");
%                 sol = altsol; 
%             end
%         end
        wijk(startEdge(l):stopEdge(l)) = sol; 
    end
    
end

% Construct SVec from wVec
SVec = zeros(1, m);
for l = 1:m
    SVec(l) = wijk(startEdge(l):stopEdge(l))*S0Mat(:,l);
end


% subplot(2,2,4)
plot(ErrVec, SVec, 'b.')
title(strcat('\sigma=', num2str(sigma)) )
xlabel('s_{ij}^*') 
ylabel('s_{ij}(5)') 

fprintf('Max error: %f\n', max(abs(SVec - ErrVec)))
fprintf('Average error %f\n',mean(abs(SVec - ErrVec)))
fprintf('Median error%f\n', median(abs(SVec - ErrVec)))