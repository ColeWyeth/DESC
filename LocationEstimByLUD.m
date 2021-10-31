%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Header goes here
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [t_est, out] = LocationEstimByLUD(AdjMat, tijMat, opts)

t_start = tic;
%% General algorithmic parameters:
%%
if ~isfield(opts, 'tolIRLS');       opts.tolIRLS = 1e-4;   end    % IRLS Tolerance
if ~isfield(opts, 'tolQuad');       opts.tolQuad = 1e-8;   end    % Inner loop tolerance
if ~isfield(opts, 'delt');          opts.delt = 1e-12;     end    % IRLS regularization parameter
if ~isfield(opts, 'maxit');         opts.maxit = 200;      end    % Maximum number of iterations
if ~isfield(opts, 'maxitQuad');     opts.maxitQuad = 200;  end    % Maximum number of inner iterations
if ~isfield(opts, 'staglim');       opts.staglim = 5;      end    % Number of iteration for stagnation


tolIRLS      = opts.tolIRLS;
tolQuad      = opts.tolQuad;
delt         = opts.delt;
maxit        = opts.maxit;
maxitQuad    = opts.maxitQuad;
staglim      = opts.staglim;


n = size(AdjMat,1);
d = size(tijMat,1);
ss_num = size(tijMat,2);

[Ind_j, Ind_i] = find(tril(AdjMat,-1));
j_Vec_Lmat = vec([Ind_i Ind_j]');
i_Vec_Lmat = kron([1:ss_num]',ones(2,1));
val_Vec_Lmat = kron(ones(ss_num,1),[1;-1]);
l_mat = sparse(i_Vec_Lmat,j_Vec_Lmat,val_Vec_Lmat,ss_num,n,2*ss_num);
Lmat = kron(l_mat,speye(d));
V = kron(ones(n,1),speye(d));

%% Matrix of cost function
Mmat = [Lmat -sparse([1:d*ss_num]',kron([1:ss_num]',ones(d,1)),tijMat(:),d*ss_num,ss_num,d*ss_num)];

%% Start IRLS loops
i = 1;
wMat = speye(d*ss_num);
optsQuad = optimset('Algorithm','interior-point-convex','MaxIter',maxitQuad,...
    'TolFun',tolQuad,'Display','off');
cost_val_old = 1;
cost_vec = []; stagcnt = 0;

while (i <= maxit)&&(stagcnt <= staglim)
    
    %% Compute solution of quadratic program
    t_alph_est = quadprog(Mmat'*wMat*Mmat, sparse(d*n+ss_num,1), ...
        [-sparse(ss_num,d*n) -speye(ss_num)], -ones(ss_num,1), [V' sparse(d,ss_num)], sparse(d,1),[],[],[],optsQuad);
    
    %% Update IRLS weights
    residual_vec = reshape(Mmat*t_alph_est,d,ss_num);
    residual_norms = sqrt(bsxfun(@dot,residual_vec,residual_vec));
    wMat = kron(sparse([1:ss_num]',[1:ss_num]',1./(sqrt(residual_norms.^(2) + delt)),ss_num,ss_num,ss_num),speye(d));
    
    %% Check convergence of cost function
    cost_val = sum(residual_norms);
    cost_diff = abs(cost_val_old - cost_val)/cost_val_old;
    if (cost_diff <= tolIRLS)
        stagcnt = stagcnt + 1;
    else
        stagcnt = 0;
    end
    cost_val_old = cost_val;
    cost_vec = [cost_vec; cost_val];
    %fprintf(' IRLS iteration %d done! \n',i);
    i = i + 1;
end

t_est = reshape(t_alph_est(1:d*n),d,n);
alph = t_alph_est(d*n+1:end);

t_end = toc(t_start);

out.cost_vec = cost_vec;
out.alph = alph;
out.TotalTime = t_end;
if (i > maxit)
    if (stagcnt <= staglim)
        out.flag = -2;
    else
        out.flag = -1;
    end
else
    out.flag = 0;
end

return