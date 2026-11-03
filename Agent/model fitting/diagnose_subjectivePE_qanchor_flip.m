%% ========================================================================
%  Diagnostic: does q_anchor (= mu_hat(n-1)) need a contrast-based flip to
%  stay comparable to rewards(n) in the SUBJECTIVE-PE model?
%  ------------------------------------------------------------------------
%  Read-only, model-free diagnostic -- no fitting, no fitted parameters
%  needed. Directly checks whether raw mu_hat(n-1) predicts rewards(n) the
%  same way in contrast==0 and contrast==1 trials, or whether the
%  relationship inverts in contrast==1 (which would justify the
%  contrast-based flip already added to q_anchor in
%  nll_basicRL_integrated_subjectivePE / nll_RLsigma_VOI_subjectivePE).
%
%  Direct analog of diagnose_mu_hat_contrast_flip.m, which tested the
%  OBJECTIVE-PE model and came back showing NO flip was needed --
%  contradicting the same contrast-reverses-the-better-option assumption
%  that justified the q_anchor fix. If this diagnostic ALSO shows no sign
%  flip between contrast conditions, that's evidence the q_anchor fix
%  should be reconsidered/reverted.
%  ========================================================================
clc; clearvars;

[dataBoth, ~, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

%% Pool (mu_hat(n-1), rewards(n), contrast(n)) across all subjects/trials
all_mu_hat_prev = [];
all_rewards = [];
all_contrast = [];

for n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % this subject's trial-level data
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % same recoding the fitting pipeline uses

    for t = 2:length(subj.mu_hat)
        if subj.blocks(t) ~= subj.blocks(t-1)
            continue; % q_anchor resets to 0.5 here -- mu_hat(t-1) isn't actually used by the model on this trial
        end
        all_mu_hat_prev(end+1,1) = subj.mu_hat(t-1); %#ok<AGROW>
        all_rewards(end+1,1) = rewards(t); %#ok<AGROW>
        all_contrast(end+1,1) = subj.contrast(t); %#ok<AGROW>
    end
end

is0 = all_contrast == 0;
is1 = all_contrast == 1;

%% Binned calibration plot: mean mu_hat(n-1) vs mean rewards(n), by contrast
% rewards(n) is binary, so raw points aren't informative to look at
% directly -- bin mu_hat(n-1) and look at the mean reward rate per bin,
% the same way Fig 2e-style learning curves are typically shown.
nBins = 10;
edges = linspace(0, 1, nBins+1);
bin0_x = NaN(nBins,1); bin0_y = NaN(nBins,1);
bin1_x = NaN(nBins,1); bin1_y = NaN(nBins,1);
for b = 1:nBins
    in_bin0 = is0 & all_mu_hat_prev >= edges(b) & all_mu_hat_prev < edges(b+1);
    in_bin1 = is1 & all_mu_hat_prev >= edges(b) & all_mu_hat_prev < edges(b+1);
    if any(in_bin0)
        bin0_x(b) = mean(all_mu_hat_prev(in_bin0));
        bin0_y(b) = mean(all_rewards(in_bin0));
    end
    if any(in_bin1)
        bin1_x(b) = mean(all_mu_hat_prev(in_bin1));
        bin1_y(b) = mean(all_rewards(in_bin1));
    end
end

figure('Position', [100, 100, 700, 650]);
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1.2, 'DisplayName', 'rewards(n) = mu\_hat(n-1) (diagonal)');
plot([0 1], [1 0], 'k:', 'LineWidth', 1.2, 'DisplayName', 'rewards(n) = 1-mu\_hat(n-1) (anti-diagonal)');
plot(bin0_x, bin0_y, '-o', 'Color', [0.2 0.4 0.8], 'MarkerFaceColor', [0.2 0.4 0.8], 'LineWidth', 1.5, 'DisplayName', 'contrast = 0');
plot(bin1_x, bin1_y, '-o', 'Color', [0.85 0.3 0.2], 'MarkerFaceColor', [0.85 0.3 0.2], 'LineWidth', 1.5, 'DisplayName', 'contrast = 1');

xlabel('mu\_hat(n-1) (subject''s previous slider report)', 'FontSize', 12);
ylabel('mean rewards(n) (recoded reward, this trial)', 'FontSize', 12);
title('Subjective-PE q\_anchor: does mu\_hat(n-1) need a contrast-based flip?', 'FontSize', 13);
legend('Location', 'bestoutside');
xlim([0 1]); ylim([0 1]);
axis square;
grid on;
hold off;

%% Quantify: point-biserial (Spearman) correlation within each contrast condition
r0 = corr(all_mu_hat_prev(is0), all_rewards(is0), 'Type', 'Spearman', 'Rows', 'complete');
r1_direct = corr(all_mu_hat_prev(is1), all_rewards(is1), 'Type', 'Spearman', 'Rows', 'complete');
r1_flipped = corr(1 - all_mu_hat_prev(is1), all_rewards(is1), 'Type', 'Spearman', 'Rows', 'complete');

fprintf('\n=== mu_hat(n-1) vs rewards(n), by contrast ===\n');
fprintf('contrast == 0: Spearman r(mu_hat(n-1), rewards(n))       = %.3f  (n = %d trials)\n', r0, sum(is0));
fprintf('contrast == 1: Spearman r(mu_hat(n-1), rewards(n))       = %.3f  (n = %d trials)\n', r1_direct, sum(is1));
fprintf('contrast == 1: Spearman r(1-mu_hat(n-1), rewards(n))     = %.3f  (n = %d trials)\n', r1_flipped, sum(is1));
fprintf('\nIf contrast == 1''s correlation is strong NEGATIVE (or weak) for mu_hat(n-1)\n');
fprintf('but strong POSITIVE for 1-mu_hat(n-1), that confirms q_anchor needs the flip\n');
fprintf('already added. If both contrasts show similar POSITIVE correlation for raw\n');
fprintf('mu_hat(n-1), the flip was unnecessary and should be reconsidered/reverted.\n');
