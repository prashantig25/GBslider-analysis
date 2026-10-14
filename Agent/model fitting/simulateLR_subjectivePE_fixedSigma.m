%% ========================================================================
%  Script: Simulated Learning Rate vs Contrast Difference -- basicRL/RLsigma,
%  Subjective PE, Fixed Sigma (model analog of Figure 4a)
%  ------------------------------------------------------------------------
%  Figure 4a (Figures/manuscript_figures/main/figure4.m, lines 58-119) bins
%  each real subject's trial-level learning rate (up./pe) by absolute
%  contrast difference and plots the group mean +/- SEM per bin -- built
%  entirely from real behavioral data (preprocessed_data.xlsx), with no
%  model simulation involved. This script builds the model-simulation
%  analog: using each subject's own fitted [alpha, kappa] (sigma fixed
%  from their perceptual-choice fit) from fitReducedModelSpace_
%  fixedSigma.m, simulate one replicate of that subject's task (their own
%  real block/state/condiff design, but freshly-generated choice, reward,
%  and slider report), recompute pe/up/LR on the simulated stream using
%  the same formula figure4.m's real-data pipeline uses (preprocess_LR.m's
%  compute_state_dep_pe), and bin/plot it the same way. One figure per
%  condition (Both, Perceptual), each with two side-by-side subplots
%  (basicRL, RLsigma).
%  ------------------------------------------------------------------------
%  no_integration toggles which observation-handling definition's fitted
%  parameters/simulate function this script uses:
%    false (default) - integrated: belief computed by marginalizing over a
%                       discretized grid of hypothetical observations
%                       (simulate_basicRL_integrated_choice_subjectivePE /
%                       simulate_RLsigma_integrated_choice_subjectivePE),
%                       loading fitReducedModelSpace_fixedSigma.m's
%                       "SECTION 2 OF 4: SUBJECTIVE PE, INTEGRATED" fits.
%    true             - no integration: belief computed from a single
%                       point estimate of the observed condiff
%                       (simulate_basicRL_pointObs_choice_subjectivePE /
%                       simulate_RLsigma_pointObs_choice_subjectivePE),
%                       loading fitReducedModelSpace_fixedSigma.m's
%                       "SECTION 4 OF 4: SUBJECTIVE PE, NO INTEGRATION"
%                       fits.
%  ------------------------------------------------------------------------
%  Design notes:
%  - "No integration" means belief each trial comes from a single point
%    estimate of the observed condiff (normcdf(condiff(n), 0, sigma))
%    rather than integrated over a discretized grid of hypothetical
%    observations -- see fitSlider_ALLmodels.m's nll_basicRL_pointObs
%    header for the full rationale, and simulate_basicRL_pointObs_choice_
%    subjectivePE's header for how that mechanism was carried over into
%    this generative (choice-simulating) version.
%  - beta (softmax inverse-temperature) and p_reward_correct (reward-
%    contingency probability) are not part of this model's free
%    parameters -- there's no fitted value to use. Fixed at beta=3,
%    p_reward_correct=0.7, matching the existing convention in
%    debug_choice_simulation_ALLmodels.m for the integrated simulate
%    functions (the no-integration ones use the same constants).
%  - pe/up are computed directly on the simulated mu_hat/recoded-reward
%    stream via the local compute_simulated_pe_up() function below, which
%    matches preprocess_LR.m's compute_state_dep_pe exactly: pe is
%    state-sign-flipped against mu_hat(n-1), force-zeroed at each block's
%    first trial; up = mu_hat(n) - mu_hat(n-1), left NaN only at the very
%    first trial overall (block boundaries do NOT reset it, matching real
%    data's behavior -- though since pe==0 trials get filtered out before
%    computing LR below, same as figure4.m, this quirk never actually
%    reaches the plotted LR values). Unlike real data, no presentation-
%    flip-convention correction is needed (preprocess_LR.m's compute_mu
%    contrast==1 flip is a real-data logging artifact, not something
%    inherent to the model), so mu_t/mu_t_1 are just the raw simulated
%    mu_hat(n)/mu_hat(n-1) directly.
%  - The pe formula's "recoded reward" uses recode_rewards_choice
%    (choice-relative), not recode_rewards (contrast-based) -- the
%    simulators' raw "reward" output is already choice-relative with no
%    presentation-side ambiguity to correct for (see fitSlider_
%    ALLmodels.m's recode_rewards_choice header).
%  - Single simulated replicate per subject (same posterior-predictive-
%    check convention used elsewhere in this repo, e.g.
%    fitReducedModelSpace_fixedSigma.m's perceptual-sigma fit), not an
%    average over many replicates.
%  - "Contrast difference" for binning is subj.condiff (condiff_relative
%    in the fitting-pipeline data) -- the same underlying per-trial
%    quantity figure4.m calls con_diff_choice under the LR-analysis
%    pipeline's separate export (preprocessed_data.xlsx), just named
%    differently across the two pipelines.
%  - Binning/plotting logic (percentile-edge deciles of abs(condiff),
%    pe==0 & |lr|>2 trial exclusion, per-subject-per-bin mean -> group
%    mean+SEM, scatter+errorbar+lsline+corr title) is copied as closely as
%    possible from figure4.m's panel 4a for a faithful comparison.
%  ========================================================================
clc; clearvars;
rng(123); % for reproducibility of the simulated choice/reward/mu_hat draws

% PATH STUFF -- anchor all reads/writes to Agent/model fitting/, regardless
% of MATLAB's current folder.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');

% See header for what this toggles.
no_integration = false;

% STYLING
line_width = 0.5;
font_size = 11; % axis/tick-label text
title_font_size = 10; % subplot + figure titles
sgtitle_font_size = 9;
dot_size = 70;
font_name = 'Arial';
[~,~,~,~,~,~,~,~,~,~,~,~,binned_dots,~,~,~,~,~,~,~] = colors_rgb(); % binned_dots is colors_rgb's 13th output

%% =================== SIMULATION SETTINGS =================================
% Not fitted parameters -- fixed constants needed only to generate
% synthetic economic choices/rewards (see design notes above).
beta = 3;
p_reward_correct = 0.7;

%% =================== LOAD DATA AND FITTED PARAMETERS ======================
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

conditions = {'Both', 'Perceptual'};
condition_data = {dataBoth, dataPerceptual};

model_names = {'basicRL', 'RLsigma'};
model_labels = {'Basic RL', 'RL + Est. Sensitivity'};

% Params filenames, simulate functions, and the sgtitle's variant label
% all switch together on no_integration -- see header for what each
% branch corresponds to.
if no_integration
    basicRL_file_fmt = 'params_basicRL_subjectivePE_noIntegration_fixedSigma_%s_multiSP.mat';
    RLsigma_file_fmt = 'params_RLsigma_subjectivePE_noIntegration_fixedSigma_%s_multiSP.mat';
    simulate_fun = {@fitSlider_ALLmodels.simulate_basicRL_pointObs_choice_subjectivePE, ...
                     @fitSlider_ALLmodels.simulate_RLsigma_pointObs_choice_subjectivePE};
    variant_label = 'No Integration';
else
    basicRL_file_fmt = 'params_basicRL_subjectivePEv2_fixedSigma_%s_multiSP.mat';
    RLsigma_file_fmt = 'params_RLsigma_subjectivePEv2_fixedSigma_%s_multiSP.mat';
    simulate_fun = {@fitSlider_ALLmodels.simulate_basicRL_integrated_choice_subjectivePE, ...
                     @fitSlider_ALLmodels.simulate_RLsigma_integrated_choice_subjectivePE};
    variant_label = 'Integrated';
end

for c = 1:length(conditions)
    cond_label = conditions{c};
    data_cond = condition_data{c};

    params_basicRL_subjectivePE = importdata(fullfile(save_dir, sprintf(basicRL_file_fmt, cond_label)));
    params_RLsigma_subjectivePE = importdata(fullfile(save_dir, sprintf(RLsigma_file_fmt, cond_label)));
    model_params = {params_basicRL_subjectivePE, params_RLsigma_subjectivePE};

    figure('Position', [100, 100, 600, 280]);
    sgtitle(sprintf('Simulated LR vs Contrast Difference (%s, Subjective PE, Fixed Sigma, %s)', cond_label, variant_label), 'FontWeight', 'bold', 'FontSize', sgtitle_font_size);

    for m = 1:length(model_names)
        params_m = model_params{m};

        % Simulate one replicate per subject on their own real trial
        % design, then pool every subject's valid (pe~=0, |lr|<=2) trials
        % together -- percentile bin edges below are computed on this
        % pooled set, matching figure4.m's binned_data.
        all_lr = [];
        all_condiff = [];
        all_subjID = [];
        for n = 1:numSubjs
            subj = preprocess_fitSlider(data_cond, uniqueID(n));
            theta = [params_m.alpha(n), params_m.kappa(n), params_m.sigma(n)];

            [mu_hat_sim, choice_sim, reward_sim, ~] = simulate_fun{m}(theta, subj.blocks, subj.state, subj.condiff, beta, p_reward_correct);
            recoded_reward_sim = fitSlider_ALLmodels.recode_rewards_choice(reward_sim, choice_sim);

            [pe, up] = compute_simulated_pe_up(mu_hat_sim, recoded_reward_sim, subj.state, subj.blocks);
            lr = up ./ pe;

            valid = pe ~= 0 & abs(lr) <= 2;
            all_lr = [all_lr; lr(valid)];
            all_condiff = [all_condiff; abs(subj.condiff(valid))];
            all_subjID = [all_subjID; repmat(uniqueID(n), sum(valid), 1)];
        end

        nbins = 10;
        bin_edges = prctile(all_condiff, 0:10:100);
        bins = discretize(all_condiff, bin_edges);

        avg_ydata_bins = NaN(nbins, numSubjs);
        avg_behv_bins = NaN(nbins, numSubjs);
        for b = 1:nbins
            for n = 1:numSubjs
                subj_mask = all_subjID == uniqueID(n);
                bins_subj = bins(subj_mask);
                y_data_subj = all_lr(subj_mask);
                binned_data_subj = all_condiff(subj_mask);
                avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b));
                avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b));
            end
        end
        avg_ydata = nanmean(avg_ydata_bins, 2);
        avg_binneddata = nanmean(avg_behv_bins, 2);
        sem_ydata = nanstd(avg_ydata_bins, 0, 2) ./ sqrt(numSubjs);

        % PLOT -- same scatter/lsline/errorbar/scatter sequence as
        % figure4.m's panel 4a (the first, invisible scatter just gives
        % lsline something to fit a line through before the real one is
        % drawn on top of it).
        ax = subplot(1, 2, m);
        scatter(1:nbins, avg_ydata, dot_size, "filled", 'MarkerEdgeColor', "none", 'MarkerFaceColor', "none");
        hold on
        ls = lsline;
        ls.Color = 'k';
        errorbar(1:nbins, avg_ydata, sem_ydata, 'k', 'LineWidth', line_width, 'LineStyle', 'none');
        scatter(1:nbins, avg_ydata, dot_size, "filled", 'MarkerEdgeColor', 'k', 'MarkerFaceColor', binned_dots);
        xlabel('Contrast-difference bins (1 bin = 0.01)', 'FontSize', font_size);
        ylabel('Mean learning rate (LR)', 'FontSize', font_size);
        set(ax, 'Color', 'none', 'FontName', font_name, 'FontSize', font_size, 'LineWidth', line_width);

        % Pad the auto-scaled ylim so the data/errorbars aren't cramped
        % against the top/bottom of the plot box.
        yl = ylim(ax);
        y_pad = 0.2 * diff(yl);
        ylim(ax, [yl(1) - y_pad, yl(2) + y_pad]);

        [rho, pval] = corr(avg_ydata, avg_binneddata, 'rows', 'pairwise');
        title(sprintf('%s\nr = %.2f, p = %.3f', model_labels{m}, rho, pval), 'FontWeight', 'normal', 'FontSize', title_font_size);
    end
end

%% ========================================================================
%  Local function: pe/up from a simulated mu_hat/recoded-reward/state/block
%  stream, matching preprocess_LR.m's compute_state_dep_pe exactly: pe is
%  state-sign-flipped against mu_hat(n-1), force-zeroed at each block's
%  first trial; up = mu_hat(n) - mu_hat(n-1), left NaN only at the very
%  first trial overall (no special reset at later block boundaries,
%  matching real data's behavior).
%  ========================================================================
function [pe, up] = compute_simulated_pe_up(mu_hat, recoded_reward, state, blocks)
    n_trials = length(mu_hat);
    pe = NaN(n_trials, 1);
    up = NaN(n_trials, 1);
    for n = 2:n_trials
        if state(n) == 0
            pe(n) = recoded_reward(n) - mu_hat(n-1);
        else
            pe(n) = (1 - recoded_reward(n)) - mu_hat(n-1);
        end
        up(n) = mu_hat(n) - mu_hat(n-1);
    end
    isBlockStart = [true; diff(blocks) ~= 0];
    pe(isBlockStart) = 0;
end
