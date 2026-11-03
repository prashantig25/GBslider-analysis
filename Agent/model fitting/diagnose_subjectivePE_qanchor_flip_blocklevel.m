%% ========================================================================
%  Diagnostic (block-level): does q_anchor's mu_hat need a contrast-based
%  flip to stay comparable to the block's reward rate in the SUBJECTIVE-PE
%  model?
%  ------------------------------------------------------------------------
%  Read-only, model-free diagnostic -- no fitting needed. The trial-level
%  version of this check (diagnose_subjectivePE_qanchor_flip.m) correlated
%  mu_hat(n-1) against the single next trial's binary reward, which is
%  likely too noisy to detect anything (a single Bernoulli draw has huge
%  sampling variance relative to a slowly-converging belief). This version
%  instead compares each block's LATE (converged) mu_hat against that
%  block's OVERALL empirical reward rate -- both smooth, aggregated
%  quantities, the same kind of comparison that worked cleanly in
%  diagnose_mu_hat_contrast_flip.m (mu_hat vs q_transformed, r ~ 0.54).
%
%  If mu_hat needs a contrast-based flip, contrast==0 blocks should show
%  mu_hat_late ~= block_reward_rate (near the diagonal), while contrast==1
%  blocks should show mu_hat_late ~= 1 - block_reward_rate (near the
%  anti-diagonal) instead.
%  ========================================================================
clc; clearvars;

[dataBoth, ~, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

nLateTrials = 5; % average the last 5 trials of each block for mu_hat_late

%% Compute, per subject per block: mu_hat_late, block_reward_rate, contrast
all_mu_hat_late = [];
all_block_reward_rate = [];
all_contrast = [];

for n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % this subject's trial-level data
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % same recoding the fitting pipeline uses

    blockIDs = unique(subj.blocks, 'stable');
    for b = 1:length(blockIDs)
        inBlock = subj.blocks == blockIDs(b);
        block_mu_hat = subj.mu_hat(inBlock);
        block_rewards = rewards(inBlock);
        block_contrast = subj.contrast(inBlock);

        if length(block_mu_hat) < nLateTrials
            continue; % skip incomplete blocks
        end

        lateIdx = (length(block_mu_hat) - nLateTrials + 1):length(block_mu_hat);
        all_mu_hat_late(end+1,1) = mean(block_mu_hat(lateIdx)); %#ok<AGROW>
        all_block_reward_rate(end+1,1) = mean(block_rewards); %#ok<AGROW>
        all_contrast(end+1,1) = block_contrast(1); % contrast is block-constant
    end
end

is0 = all_contrast == 0;
is1 = all_contrast == 1;

%% Scatter plot: mu_hat_late vs block_reward_rate, split by contrast
figure('Position', [100, 100, 700, 650]);
hold on;

h0 = scatter(all_mu_hat_late(is0), all_block_reward_rate(is0), 30, [0.2 0.4 0.8], 'filled', 'DisplayName', 'contrast = 0');
h0.MarkerFaceAlpha = 0.5;
h1 = scatter(all_mu_hat_late(is1), all_block_reward_rate(is1), 30, [0.85 0.3 0.2], 'filled', 'DisplayName', 'contrast = 1');
h1.MarkerFaceAlpha = 0.5;

plot([0 1], [0 1], 'k--', 'LineWidth', 1.2, 'DisplayName', 'reward rate = mu\_hat\_late (diagonal)');
plot([0 1], [1 0], 'k:', 'LineWidth', 1.2, 'DisplayName', 'reward rate = 1-mu\_hat\_late (anti-diagonal)');

xlabel('mu\_hat\_late (mean of last 5 trials'' slider reports per block)', 'FontSize', 12);
ylabel('block reward rate (mean recoded reward over the whole block)', 'FontSize', 12);
title('Subjective-PE q\_anchor: block-level check for a contrast-based flip', 'FontSize', 13);
legend('Location', 'bestoutside');
xlim([0 1]); ylim([0 1]);
axis square;
grid on;
hold off;

%% Quantify: Spearman correlation within each contrast condition
r0 = corr(all_mu_hat_late(is0), all_block_reward_rate(is0), 'Type', 'Spearman', 'Rows', 'complete');
r1_direct = corr(all_mu_hat_late(is1), all_block_reward_rate(is1), 'Type', 'Spearman', 'Rows', 'complete');
r1_flipped = corr(1 - all_mu_hat_late(is1), all_block_reward_rate(is1), 'Type', 'Spearman', 'Rows', 'complete');

fprintf('\n=== mu_hat_late vs block_reward_rate, by contrast ===\n');
fprintf('contrast == 0: Spearman r(mu_hat_late, reward rate)       = %.3f  (n = %d blocks)\n', r0, sum(is0));
fprintf('contrast == 1: Spearman r(mu_hat_late, reward rate)       = %.3f  (n = %d blocks)\n', r1_direct, sum(is1));
fprintf('contrast == 1: Spearman r(1-mu_hat_late, reward rate)     = %.3f  (n = %d blocks)\n', r1_flipped, sum(is1));
fprintf('\nIf contrast == 1''s correlation is strong NEGATIVE (or weak) for mu_hat_late\n');
fprintf('but strong POSITIVE for 1-mu_hat_late, that confirms q_anchor needs the flip\n');
fprintf('already added. If both contrasts show similar POSITIVE correlation for raw\n');
fprintf('mu_hat_late, the flip was unnecessary and should be reconsidered/reverted.\n');
