%% Author: Yunpeng Shi
%% © Regents of the University of Minnesota. All rights reserved.
%% This code is for demonstrating the performance of AAB statistics given synthetic data...
%% generated from uniform corruption model

%Generate synthetic data from uniform corruption model
UCModel = UniformCorruptionModel(200, 0.5, 0.2, 0);
AdjMat=UCModel.AdjMat; TMat_gt = UCModel.TMat_gt; GammaMat = UCModel.GammaMat; 
GammaMat_gt = UCModel.GammaMat_gt; true_error = UCModel.true_error;

% Implement IR-AAB with default sample size and total iterations
IRAABVec = IRAAB(AdjMat, GammaMat);

% Plots:
sigma = 0; % noise level of synthetic data
% Scatter plot: IR-AAB versus true error. Each point represents a pairwise
% direction. Blue--uncorrupted; Red--corrupted
figure;
n = size(AdjMat,1); m = size(GammaMat,2);
pointsize=10;
rgb_table = [linspace(0,1,m)',linspace(0.3,0.3,m)',linspace(1,0,m)'];
% edges are recognized as corrupted if true error > arcsin(sigma)
% 1e-4 is to avoid the effect of round-off error
scatter(true_error, IRAABVec , pointsize, true_error>(asin(sigma))+1e-4 ,'filled');
xlim([0 pi]);
colormap(rgb_table)

% ROC curve
figure;
tpr = zeros(1,1000); % pick 1000 different thresholds
fpr = zeros(1,1000);
bottom = min(IRAABVec);
top = max(IRAABVec);
lin = linspace(bottom, top, 1000);
% edges are recognized as corrupted if true error > arcsin(sigma)
% 1e-4 is to avoid the effect of round-off error
corrInd = true_error>(asin(sigma))+1e-4;
for i = 1:1000
    t = lin(i);
    tp = sum((corrInd).*(IRAABVec>t));
    pos = sum(corrInd);
    fp = sum((1-corrInd).*(IRAABVec>t));
    neg = m-pos;
    tpr(i) = tp/pos;
    fpr(i) = fp/neg;
end
plot(fpr,tpr,'-', 'LineWidth',2, 'Color', [0.9,0.1,0.1])

