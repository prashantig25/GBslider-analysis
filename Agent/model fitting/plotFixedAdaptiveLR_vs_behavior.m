%% ========================================================================
%  Script: Fixed vs Adaptive LR -- Scatter/lsline/Spearman vs Behavioral Measures
%  ------------------------------------------------------------------------
%  Scatter plots, each with a least-squares line (lsline) and Spearman's
%  rho/p in the title, for:
%    1) Fixed LR vs Adaptive LR
%    2) Fixed LR vs Economic performance (ecoperf)
%    3) Fixed LR vs Absolute estimation error (absEstError)
%    4) Adaptive LR vs Economic performance (ecoperf)
%    5) Adaptive LR vs Absolute estimation error (absEstError)
%  Produces three such figures: one using the general, aggregate-across-
%  both-conditions values for all four measures, and one each for the
%  Both/Perceptual condition-specific versions. See individualDifferences_
%  paramCorrelations.m's design notes for how both are computed:
%  - General: ecoperf/absEstError via calculateEstimationError(data,
%    subj_ids) on the full (both-conditions-mixed) dataset; Fixed/Adaptive
%    LR as columns 1/2 of the signed-PE learning-rate regression
%    (betas_signed_wo_rewunc_obj.mat).
%  - Both/Perceptual: ecoperf/absEstError recomputed directly on each
%    condition's trial subset; Fixed/Adaptive LR from that condition's own
%    signed-PE regression refit (betas_signed_wo_rewunc_obj_both.mat /
%    _perceptual.mat).
%  ========================================================================
clc; clearvars;

% PATH STUFF -- anchor all loads to the repo root, regardless of MATLAB's
% current working directory.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end

% Marker color -- matches the "binned analysis data" color used for
% scatter dots elsewhere in this repo (colors_rgb's 13th output).
[~,~,~,~,~,~,~,~,~,~,~,~,binned_dots,~,~,~,~,~,~,~] = colors_rgb();

%% =================== LOAD BEHAVIORAL / LR MEASURES =======================
data = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'preprocessed_data.mat'));
subj_ids = unique(data.ID);

% General (aggregate-across-both-conditions) values.
[ecoperf, absEstError] = calculateEstimationError(data, subj_ids);
betas_signed = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'betas_signed_wo_rewunc_obj.mat'));
fixedLR = betas_signed(:, 1);
adaptiveLR = betas_signed(:, 2);

% Both/Perceptual condition-specific values -- recomputed directly on each
% condition's trial subset (condition 1 = Both, condition 2 = Perceptual),
% not derived from the general values above.
[ecoperf_Both, absEstError_Both] = calculateEstimationError(data(data.condition == 1, :), subj_ids);
[ecoperf_Perceptual, absEstError_Perceptual] = calculateEstimationError(data(data.condition == 2, :), subj_ids);

betas_signed_Both = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'betas_signed_wo_rewunc_obj_both.mat'));
fixedLR_Both = betas_signed_Both(:, 1);
adaptiveLR_Both = betas_signed_Both(:, 2);

betas_signed_Perceptual = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'betas_signed_wo_rewunc_obj_perceptual.mat'));
fixedLR_Perceptual = betas_signed_Perceptual(:, 1);
adaptiveLR_Perceptual = betas_signed_Perceptual(:, 2);

%% =================== SCATTER / LSLINE / SPEARMAN PLOTS =====================
plot_labels = {'General', 'Both', 'Perceptual'};
fixedLR_bySet = {fixedLR, fixedLR_Both, fixedLR_Perceptual};
adaptiveLR_bySet = {adaptiveLR, adaptiveLR_Both, adaptiveLR_Perceptual};
ecoperf_bySet = {ecoperf, ecoperf_Both, ecoperf_Perceptual};
absEstError_bySet = {absEstError, absEstError_Both, absEstError_Perceptual};

for s = 1:length(plot_labels)
    set_label = plot_labels{s};
    fixedLR_s = fixedLR_bySet{s};
    adaptiveLR_s = adaptiveLR_bySet{s};
    ecoperf_s = ecoperf_bySet{s};
    absEstError_s = absEstError_bySet{s};

    figure('Position', [100, 100, 1500, 300]);
    sgtitle(sprintf('%s: Fixed/Adaptive LR vs behavioral measures', set_label), 'FontWeight', 'bold');

    subplot(1, 5, 1);
    plot_scatter_lsline_spearman(fixedLR_s, adaptiveLR_s, 'Fixed LR', 'Adaptive LR', 'Fixed vs Adaptive LR', binned_dots);

    subplot(1, 5, 2);
    plot_scatter_lsline_spearman(fixedLR_s, ecoperf_s, 'Fixed LR', 'Economic performance', 'Fixed LR vs Ecoperf', binned_dots);

    subplot(1, 5, 3);
    plot_scatter_lsline_spearman(fixedLR_s, absEstError_s, 'Fixed LR', 'Abs. estimation error', 'Fixed LR vs Abs. Est. Error', binned_dots);

    subplot(1, 5, 4);
    plot_scatter_lsline_spearman(adaptiveLR_s, ecoperf_s, 'Adaptive LR', 'Economic performance', 'Adaptive LR vs Ecoperf', binned_dots);

    subplot(1, 5, 5);
    plot_scatter_lsline_spearman(adaptiveLR_s, absEstError_s, 'Adaptive LR', 'Abs. estimation error', 'Adaptive LR vs Abs. Est. Error', binned_dots);
end

%% ========================================================================
%  Local function: scatter + lsline + Spearman's rho/p title
%  ========================================================================
function plot_scatter_lsline_spearman(x, y, xlabel_str, ylabel_str, title_str, marker_color)
    scatter(x, y, 30, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', marker_color);
    hold on;
    ls = lsline;
    ls.Color = 'k';
    ls.LineWidth = 1.2;
    xlabel(xlabel_str);
    ylabel(ylabel_str);
    [rho, pval] = corr(x, y, 'rows', 'complete', 'Type', 'Spearman');
    title(sprintf('%s\n\\itrho\\rm = %.2f, \\itp\\rm = %.3f', title_str, rho, pval), 'FontWeight', 'normal', 'FontSize', 9);
    box off;
end
