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

startEdge = @(l) nsample*(l-1) + 1;
stopEdge = @(l) nsample*l;
rangeOf = @(next) next:(next + nsample -1);

% Linear Programming Code Begins
%A = sparse(nsample*2*m, nsample*m);
selements = 6*m*(nsample^2);
srows = zeros(1, selements);
scolumns = zeros(1, selements);
svals = zeros(1, selements); 
next = 1; 

b = zeros(1, nsample*2*m);
    
Aeq = sparse(m, nsample*m);
beq = ones(1, m); % wijk for each ij sums to 1
% Now we're minimizing the sum over wijk 
% so the matrix must have nsample rows for each 
% edge, used to compute sij from S0Mat(:,ij) 

for l = 1:m
    i = Ind_i(l);
    j = Ind_j(l);
    offset = nsample*2*(l-1); % previously filled
    startl = nsample*(l-1) + 1;
    stopl = nsample*l;
    Aeq(l,startl:stopl) = ones(1,nsample); % Weights sum to one 
    for kInd = 1:nsample
        % Then CoIndMat(kInd,l) is the vertex k
        k = CoIndMat(kInd, l);
        
        %A(offset + 2*kInd - 1, startl:stopl) = S0Mat(:,l);
        srows(rangeOf(next)) = (offset + 2*kInd -1)*ones(1,nsample);
        scolumns(rangeOf(next)) = startl:stopl; 
        svals(rangeOf(next)) = S0Mat(:,l); 
        next = next + nsample; 
        
        edgeik = IndMat(i,k);
%         A(offset + 2*kInd -1, startik:stopik) = -1.*S0Mat(:,IndMat(i,k)); % -sij = -sum wijk*dijk
        srows(rangeOf(next)) = (offset + 2*kInd -1)*ones(1,nsample);
        scolumns(rangeOf(next)) = startEdge(edgeik):stopEdge(edgeik); 
        svals(rangeOf(next)) = -1.*S0Mat(:,IndMat(i,k));
        next = next + nsample; 

        edgejk = IndMat(j,k);
%         A(offset + 2*kInd -1, startjk:stopjk) = -1.*S0Mat(:,IndMat(j,k)); % -sij = -sum wijk*dijk
        srows(rangeOf(next)) = (offset + 2*kInd -1)*ones(1,nsample);
        scolumns(rangeOf(next)) = startEdge(edgejk):stopEdge(edgejk); 
        svals(rangeOf(next)) = -1.*S0Mat(:,IndMat(j,k));
        next = next + nsample; 
        
        b(offset + 2*kInd - 1) = S0Mat(kInd, l);
        
        %A(offset + 2*kInd, startl:stopl) = -1.*S0Mat(:,l); 
        srows(rangeOf(next)) = (offset + 2*kInd)*ones(1,nsample);
        scolumns(rangeOf(next)) = startl:stopl; 
        svals(rangeOf(next)) = -1*S0Mat(:,l); 
        next = next + nsample; 
        
        %A(offset + 2*kInd, startik:stopik) = -1.*S0Mat(:,edgeik);
        srows(rangeOf(next)) = (offset + 2*kInd)*ones(1,nsample);
        scolumns(rangeOf(next)) = startEdge(edgeik):stopEdge(edgeik); 
        svals(rangeOf(next)) = -1.*S0Mat(:,IndMat(i,k));
        next = next + nsample;
        
        %A(offset + 2*kInd, startjk:stopjk) = -1.*S0Mat(:,edgejk);
        srows(rangeOf(next)) = (offset + 2*kInd)*ones(1,nsample);
        scolumns(rangeOf(next)) = startEdge(edgejk):stopEdge(edgejk); 
        svals(rangeOf(next)) = -1.*S0Mat(:,IndMat(j,k));
        next = next + nsample; 
        
        b(offset + 2*kInd) = -S0Mat(kInd, l); 
    end
end
A = sparse(srows, scolumns, svals, 2*nsample*m, nsample*m);

f = ones(1,m*nsample);
lb = zeros(1,m*nsample);
ub = ones(1,m*nsample); 

wVec = linprog(f, A, b, Aeq, beq, lb, ub);

% Construct SVec from wVec
SVec = zeros(1, m);
for l = 1:m
    startl = nsample*(l-1) + 1;
    stopl = nsample*l;
    SVec(l) = S0Mat(:,l)'*wVec(startl:stopl);
end


% subplot(2,2,4)
plot(ErrVec, SVec, 'b.')
title(strcat('\sigma=', num2str(sigma)) )
xlabel('s_{ij}^*') 
ylabel('s_{ij}(5)') 

fprintf('Max error: %f\n', max(abs(SVec - ErrVec)))
fprintf('Average error %f\n',mean(abs(SVec - ErrVec)))
fprintf('Median error%f\n', median(abs(SVec - ErrVec)))