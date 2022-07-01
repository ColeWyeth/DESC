%%*************************************************************************
%% This function estimates the signed scale and translation between two
%% sets of n points ({ti} and {ti^0}) in R^d in the least squares sense,
%% by solving the optimization problem (1)
%%
%% (1)       min           sum_i norm(ti^0 - (c*ti+t))^2
%%    {c in R, t in R^d}
%%
%% Using the estimated signed scale and translation, it provides the best
%% fit of {ti} to {ti^0} under such transformations.
%%
%% Author: Onur Ozyesil
%%*************************************************************************
%% Input:
%% ti  : d-by-n matrix of points to be registered ({ti} above)
%% ti0 : d-by-n matrix of reference points ({ti^0} above)
%%
%% Output:
%% t_fit     : ti registered to ti0 (t_fit = c_opt*ti + t_opt)
%% t_opt     : Optimal shift (in the sense of (1))
%% c_opt     : Optimal signed scale (in the sense of (1))
%% NRMSE     : Normalized root-mean-squared-error (see Eq.(5.2) in [1])
%% MSE_trans : Mean squared error of the transformation
%%*************************************************************************
%%
%%        UPDATE HEADER!!
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [t_fit, t_opt, c_opt, NRMSE, MSE_trans] = SimpleTransScaleRemove(ti, ti0, method)

n = size(ti,2); d = size(ti,1);

if strcmp(method,'L2')
    %% Closed form solution of the least squares problem (1)
    x_opt = pinv([n*eye(d) sum(ti,2);sum(ti,2)' sum(bsxfun(@dot,ti,ti))])*[sum(ti0,2);sum(bsxfun(@dot,ti,ti0))];
    t_opt = x_opt(1:d);
    c_opt = x_opt(end);
else
    cvx_begin quiet
        variable c_opt;
        variable t_opt(d);
        minimize (sum(norms(ti0 - (c_opt*ti+t_opt*ones(1,n)),2,1)))
    cvx_end
end
%% Evaluate the fit and the error
t_fit = c_opt*ti+t_opt*ones(1,n);
t_err = ti0 - t_fit;
MSE_trans = (1/n)*sum(bsxfun(@dot,t_err,t_err));
ti0_mean = mean(ti0,2);
ti0_ctrd = ti0 - ti0_mean*ones(1,n);
NRMSE = sqrt(sum(bsxfun(@dot,t_err,t_err))/sum(bsxfun(@dot,ti0_ctrd,ti0_ctrd)));

return