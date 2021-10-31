function X = SimplexProj(Y)
% Wang and Carreira-Perpinan, 2013: https://arxiv.org/pdf/1309.1541.pdf
    [N,D] = size(Y);

    X = sort(Y,2,'descend');

    Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));

    X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);