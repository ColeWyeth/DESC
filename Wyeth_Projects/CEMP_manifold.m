rng(3)
tic
n=100;
nsample=50;
p=0.5;
q=0.5;
sigma=0;
beta=5;
beta_max=40;
rate=2;
adversarial = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Change Parameters Below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
learning_rate = 1;
learning_iters = 40;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[S0Mat, CoIndMat, IndMat, m, Ind_i, Ind_j, ErrVec] = Initialization(n, nsample, p, q, sigma, adversarial);

startEdge = @(l) nsample*(l-1) + 1;
stopEdge = @(l) nsample*l;
rangeOf = @(next) next:(next + nsample -1); 

%A = sparse(nsample*m, nsample*m); 
selements = 2*m*(nsample^2);
srows = zeros(1, selements);
scolumns = zeros(1, selements);
svals = zeros(1, selements); 
next = 1; 


for l = 1:m % for each edge ij 
   i = Ind_i(l);
   j = Ind_j(l); 
   for indK = 1:nsample % for each cycle ijk
       rowInd = startEdge(l) + indK - 1; 

       k = CoIndMat(indK, l); 

       %grad(indK) = wijk(:,IndMat(i,k))'*S0Mat(:,IndMat(i,k)) + wijk(:,IndMat(j,k))'*S0Mat(:,IndMat(j,k)); 

       ik = IndMat(i,k) ;
       %A(rowInd, startEdge(ik):stopEdge(ik)) = S0Mat(:, ik)'; % dik,l
       srows(rangeOf(next)) = rowInd*ones(1,nsample); 
       scolumns(rangeOf(next)) = startEdge(ik):stopEdge(ik); 
       svals(rangeOf(next)) = S0Mat(:, ik)'; 
       next = next + nsample; 

       jk = IndMat(j,k); 
       %A(rowInd, startEdge(jk):stopEdge(jk)) = S0Mat(:, jk)'; % djk,l
       srows(rangeOf(next)) = rowInd*ones(1,nsample); 
       scolumns(rangeOf(next)) = startEdge(jk):stopEdge(jk); 
       svals(rangeOf(next)) = S0Mat(:, jk)';
       next = next + nsample; 
   end
end
A = sparse(srows, scolumns, svals, nsample*m, nsample*m);

A = A' + A; 
manifold = multinomialfactory(nsample, m); 
problem.M = manifold;

columnify = @(w) reshape(w, 1, m*nsample)'; 
matrixify = @(w) reshape(w, nsample, m); 

problem.cost = @(x) -columnify(x)'*(A*columnify(x));
problem.egrad = @(x) matrixify(-2*A*columnify(x)); 

% options = optimset();
% options.Delta_bar = 30000;
[wijk, cost, info, options] = trustregions(problem);

wijk = reshape(wijk, 1, m*nsample);
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
fprintf('Average error: %f\n',mean(abs(SVec - ErrVec)))
fprintf('Median error: %f\n', median(abs(SVec - ErrVec)))