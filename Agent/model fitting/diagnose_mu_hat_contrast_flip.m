%% ========================================================================
%  Diagnostic: does mu_hat need a contrast-based flip to stay comparable
%  to q_transformed in the OBJECTIVE-PE basicRL model?
%  ------------------------------------------------------------------------
%  Read-only diagnostic -- does NOT refit or modify anything. Reconstructs
%  each subject's trial-by-trial q_transformed trajectory from already-
%  fitted basicRL (objective-PE, fixed-sigma, Both condition) parameters
%  (params_basicRL_fixedSigma_Both_multiSP.mat), via
%  fitSlider_ALLmodels.reconstruct_basicRL_integrated_trajectory (the same
%  recursive update as nll_basicRL_integrated, exposing q_transformed
%  instead of computing NLL).
%
%  Logic: mu_hat is congruence-corrected only, so it's anchored to "the
%  more rewarding option" -- which equals a=s (and hence q_transformed's
%  reference frame) only in contrast==0 blocks. If mu_hat needs the same
%  contrast-based flip we already applied to q_anchor in the subjective-PE
%  model, then plotting q_transformed against mu_hat should show:
%    - contrast==0 points clustering near the diagonal (q_transformed ~= mu_hat)
%    - contrast==1 points clustering near the anti-diagonal (q_transformed ~= 1-mu_hat)
%  instead of both clustering near the diagonal.
%  ========================================================================
clc; clearvars;

% PATH STUFF -- same convention as fitReducedModelSpace_fixedSigma.m
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');

[dataBoth, ~, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

perceptualSigma = importdata(fullfile(save_dir, 'sigma_perceptualChoice.mat'));
fixedSigma = perceptualSigma.sigma;

% Already-fitted objective-PE basicRL parameters (Both condition, fixed sigma)
params_basicRL = importdata(fullfile(save_dir, 'params_basicRL_fixedSigma_Both_multiSP.mat'));

%% Reconstruct q_transformed per trial, per subject, and pool across subjects
all_q_transformed = [];
all_mu_hat = [];
all_contrast = [];

for n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % this subject's trial-level data
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % same recoding the fitting pipeline uses
    sigma_n = fixedSigma(n);
    params_n = [params_basicRL.alpha(n), params_basicRL.kappa(n), sigma_n];

    q_transformed_trial = fitSlider_ALLmodels.reconstruct_basicRL_integrated_trajectory( ...
        params_n, subj.mu_hat, subj.blocks, rewards, subj.condiff);

    all_q_transformed = [all_q_transformed; q_transformed_trial]; %#ok<AGROW>
    all_mu_hat = [all_mu_hat; subj.mu_hat]; %#ok<AGROW>
    all_contrast = [all_contrast; subj.contrast]; %#ok<AGROW>
end

is0 = all_contrast == 0;
is1 = all_contrast == 1;

%% Scatter plot: q_transformed vs mu_hat, split by contrast
figure('Position', [100, 100, 700, 650]);
hold on;

h0 = scatter(all_mu_hat(is0), all_q_transformed(is0), 18, [0.2 0.4 0.8], 'filled', 'DisplayName', 'contrast = 0');
h0.MarkerFaceAlpha = 0.4;
h1 = scatter(all_mu_hat(is1), all_q_transformed(is1), 18, [0.85 0.3 0.2], 'filled', 'DisplayName', 'contrast = 1');
h1.MarkerFaceAlpha = 0.4;

plot([0 1], [0 1], 'k--', 'LineWidth', 1.2, 'DisplayName', 'q_{transformed} = mu\_hat (diagonal)');
plot([0 1], [1 0], 'k:', 'LineWidth', 1.2, 'DisplayName', 'q_{transformed} = 1-mu\_hat (anti-diagonal)');

xlabel('mu\_hat (subject slider report)', 'FontSize', 12);
ylabel('q\_{transformed} (reconstructed model belief)', 'FontSize', 12);
title('Objective-PE basicRL: does mu\_hat need a contrast-based flip?', 'FontSize', 13);
legend('Location', 'bestoutside');
xlim([0 1]); ylim([0 1]);
axis square;
grid on;
hold off;

%% Quantify: Spearman correlation within each contrast condition
r0 = corr(all_mu_hat(is0), all_q_transformed(is0), 'Type', 'Spearman', 'Rows', 'complete');
r1_direct = corr(all_mu_hat(is1), all_q_transformed(is1), 'Type', 'Spearman', 'Rows', 'complete');
r1_flipped = corr(1 - all_mu_hat(is1), all_q_transformed(is1), 'Type', 'Spearman', 'Rows', 'complete');

fprintf('\n=== mu_hat vs q_transformed, by contrast ===\n');
fprintf('contrast == 0: Spearman r(mu_hat, q_transformed)       = %.3f  (n = %d trials)\n', r0, sum(is0));
fprintf('contrast == 1: Spearman r(mu_hat, q_transformed)       = %.3f  (n = %d trials)\n', r1_direct, sum(is1));
fprintf('contrast == 1: Spearman r(1-mu_hat, q_transformed)     = %.3f  (n = %d trials)\n', r1_flipped, sum(is1));
fprintf('\nIf contrast == 1''s correlation is strong NEGATIVE (or weak) for mu_hat but\n');
fprintf('strong POSITIVE for 1-mu_hat, that confirms mu_hat needs the contrast flip.\n');
