%%*************************************************************************  
%% This function iteratively estimates (possibly a subset of) n rotations 
%% Ri in SO(3), given a subset of the estimates of ratios Rij = Ri^(-1)*Rj.
%%
%% At each iteration, rotations are estimated by SO(3) averaging algo of
%% [7] in our paper "Robust Camera Location Estimation by Convex 
%% Programming". The outlier ratio measurements are removed for robust 
%% rotation estimation, see [1] for details.
%%
%% [1] O. Ozyesil, A. Singer, R. Basri,    
%%     Stable Camera Motion Estimation Using Convex Programming, 
%%     arxiv preprint (arXiv:1312.5047).
%% Author: Onur Ozyesil
%%*************************************************************************  
%% Input: 
%% AdjMat : n-by-n adjacency matrix of the measurement graph
%% H      : 3n-by-3n matrix of estimates of ratios Rij, ij'th 3-by-3 
%%          block Hij of H is equal to the measurement of Rij (if there
%%          is no measurement for ij, Hij = 0_{3x3})
%% iterNum: Number of iterations 
%%
%% Output:
%% SO3Mats_est : 3-by-3-by-nn matrix of estimated rotations (nn might be 
%%               different than n)
%% IndVec_k    : Indices of estimated rotations (a subset of [1:n], with
%%               |IndVec_k| = nn)
%%*************************************************************************

function [SO3Mats_est, IndVec_k] = IterativeSO3Average(AdjMat,H,iterNum)

n = size(AdjMat,1);
IndVec_k = [1:n];
prunFactor = 1.45; % Adjusts level of outlier rejection
H_k = H;
Adj_k = AdjMat;
for k = 1:iterNum+1
    % Apply EVM
    % NOT USING EVM !!! I.e., we previously had:
    % SO3Mats_est = SynchronizeSOdByEVM(H_k, Adj_k, 3);
    % Instead of EVM, apply SO3 averaging
    [Ind_j,Ind_i] = find(tril(Adj_k,-1));
    IndMat = [Ind_i';Ind_j'];
    EdgeNum = length(Ind_i);
    RijMats = zeros(3,3,EdgeNum);
    for ss = 1:EdgeNum
        i_s = Ind_i(ss);
        j_s = Ind_j(ss);
        RijMats(:,:,ss) = H_k(3*(j_s-1)+1:3*j_s,3*(i_s-1)+1:3*i_s);
    end
    SO3Mats_est = permute(AverageSO3Graph(RijMats,IndMat),[2 1 3]);%,'Rinit',Rinit);
    
    if (k <= iterNum)
        % Determine outlier measurements
        BlockErrMat = ComputeBlockErrors(H_k,SO3Mats_est,Adj_k);
        cutLevel = nanmean(vec(BlockErrMat(BlockErrMat ~= 0))) + ...
                prunFactor*nanstd(vec(BlockErrMat(BlockErrMat ~= 0)));
        % Remove outlier measurements
        Adj_new = Adj_k.*(BlockErrMat <= cutLevel);
        % Take largest connected component of prunned measurement graph
        [S_k,C_k] = graphconncomp(sparse(Adj_new));
        compSizes_k = zeros(S_k,1);
        for i = 1:S_k
            compSizes_k(i) = sum(C_k == i);
        end
        [~, i_Csize] = max(compSizes_k);
        IndLocal_k = find(C_k == i_Csize(1));
        IndVec_k = IndVec_k(IndLocal_k);
        Adj_k = Adj_new(IndLocal_k,IndLocal_k);
        HkInds = zeros(3*length(IndLocal_k),1);
        HkInds(1:3:end) = 3*(IndLocal_k-1)+1;
        HkInds(2:3:end) = 3*(IndLocal_k-1)+2;
        HkInds(3:3:end) = 3*IndLocal_k;
        H_k = H_k(HkInds,HkInds);
        fprintf('Iteration %d completed \n', k);
    end
end

return