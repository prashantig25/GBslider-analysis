%% ========================================================================
%  Script: Individual Differences -- behavior/LR vs. fitted model parameters
%  ------------------------------------------------------------------------
%  Correlates four subject-level measures against each subject's fitted
%  parameters from the three reduced-model-space models (basicRL, RLsigma,
%  bayesianAgent; free-sigma fits from fitReducedModelSpace.m, "Both"
%  condition):
%    1) Economic performance (ecoperf)
%    2) Absolute estimation error (absEstError)
%    3) Fixed LR (signed-PE regression, beta_1)
%    4) Adaptive LR (signed-PE regression, beta_2)
%  vs. each model's alpha/kappa/sigma (kappa/sigma only for bayesianAgent,
%  which has no free learning rate).
%
%  Design notes:
%  - Economic performance and absolute estimation error are recomputed
%    here via calculateEstimationError(data, unique(data.ID)) rather than
%    loaded directly from Data/descriptive data/main study/ecoperf.mat /
%    esterror.mat (the convention corrRegCoeffs_deltaBIC.m uses). Those
%    two files carry no subject-ID column -- their row order comes from an
%    unsorted dir() listing in preprocess_mainstudy.m -- so joining them
%    positionally against the fitted-parameter files below isn't
%    guaranteed safe. Recomputing directly against unique(data.ID) gives
%    an explicit, assertable ordering instead.
%  - Fixed LR / Adaptive LR come from the signed-PE learning-rate
%    regression (Learning-rate analyses/lr_analysis/lr_analysis_obj.m),
%    saved as Data/LR analyses/betas_signed_wo_rewunc_obj.mat, a
%    [numSubjs x 5] matrix with the intercept already dropped: column 1 is
%    the main PE effect ("Fixed LR"), column 2 is labeled "Adaptive LR" by
%    every existing consumer of this file (figure4.m, figure1_SM.m,
%    corrRegCoeffs_deltaBIC.m), so that convention is followed here too --
%    though while debugging the condition-specific refits below,
%    lm.CoefficientNames actually came back as {'(Intercept)','pe',
%    'pe:contrast_diff','pe:salience_1','pe:congruence_1','pe:pe_sign_1'},
%    which would make column 2 pe:contrast_diff, not pe:salience. Worth
%    confirming before trusting "Adaptive LR" = pe:salience anywhere (see
%    conversation from 2026-09-22).
%    Like ecoperf/absEstError, this is also computed per condition:
%    betas_signed_wo_rewunc_obj_both.mat / _perceptual.mat (added to
%    LR_analysis_preprint.m) refit the same model separately on each
%    condition's trials. Fitting per-condition subsets can leave a
%    categorical regressor (e.g. pe_sign) with only one level for a
%    given subject's smaller trial count -- lr_analysis_obj.m's
%    linear_fit/get_coeffs now match coefficients by name rather than
%    position to handle this (NaN for whichever term fitlm drops for that
%    subject, see conversation from 2026-09-22), so fixedLR_Both/
%    fixedLR_Perceptual etc. may contain occasional NaNs even though the
%    aggregate fixedLR does not.
%  - Fitted parameters are loaded from "Both" condition params_*_multiSP.mat
%    files, toggled via fix_sigma below between fitReducedModelSpace.m's
%    free-sigma fits (sigma estimated jointly with alpha/kappa) and
%    fitReducedModelSpace_fixedSigma.m's fixed-sigma fits (sigma pinned to
%    each subject's own perceptual-choice-fit value). safe_saveall() saves
%    these under the literal variable name 'newData' regardless of the
%    caller's variable name, so importdata() (which returns the contents
%    directly, not load()'s field-wrapped struct) is used throughout,
%    matching every other script in this repo that reads these files.
%  - When fix_sigma is true, the sigma column for all three models comes
%    directly from sigma_perceptualChoice.mat rather than from each
%    model's own .sigma field. By construction those should already be
%    identical (fitReducedModelSpace_fixedSigma.m sets
%    sigmaParameter(n) = fixedSigma(n) for every model, where fixedSigma
%    is itself loaded from sigma_perceptualChoice.mat), but reading one
%    explicit source rather than relying on that coincidence across three
%    separate files is safer and matches the "there's one true fixed
%    sigma" framing.
%  - All four measures and the fitted parameters are joined on subject ID
%    via unique(data.ID) / uniqueID from
%    fitSlider_ALLmodels.load_fitting_data() -- both trace back to the
%    same root file (study2.txt) and should cover the same 98 subjects in
%    the same order, but this is asserted explicitly rather than assumed,
%    since a silent ID misalignment would corrupt every correlation below
%    without producing any visible error.
%  - Correlations use Spearman's rho (not Pearson), matching
%    corrRegCoeffs_deltaBIC.m's convention for this exact same class of
%    behavioral individual-differences measures.
%  ========================================================================
clc; clearvars;

% PATH STUFF -- anchor to the repo root so relative loads/saves below work
% regardless of MATLAB's current working directory.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');

% fix_sigma toggles which set of fitted parameters get loaded below:
%   false (default) - free-sigma fits from fitReducedModelSpace.m (sigma
%                      estimated jointly with alpha/kappa per model).
%   true             - fixed-sigma fits from
%                      fitReducedModelSpace_fixedSigma.m (sigma pinned per
%                      subject to their own perceptual-choice-fit value,
%                      loaded directly from sigma_perceptualChoice.mat --
%                      see design notes above).
fix_sigma = true;

%% =================== LOAD BEHAVIORAL / LR MEASURES =======================
data = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'preprocessed_data.mat'));
subj_ids = unique(data.ID);

% preprocessed_data.mat mixes conditions 1 ("Both") and 2 ("Perceptual")
% together (only condition 3, "Reward", was dropped upstream) -- see
% conversation from 2026-09-22. ecoperf and absEstError are therefore
% computed both as the aggregate-across-both-conditions value (unchanged,
% still used by the correlation section below) AND separately per
% condition (used by the MEAN +/- SEM + DOTS section, so that section's
% per-condition subplots show genuinely condition-specific values rather
% than the same aggregate value reused under two different
% best-fit-model groupings). fixedLR/adaptiveLR get the same treatment
% just below.
[ecoperf, absEstError, ~] = calculateEstimationError(data, subj_ids);
[ecoperf_Both, absEstError_Both, ~] = calculateEstimationError(data(data.condition == 1, :), subj_ids);
[ecoperf_Perceptual, absEstError_Perceptual, ~] = calculateEstimationError(data(data.condition == 2, :), subj_ids);

% betas_signed_wo_rewunc_obj.mat is likewise fit on conditions 1+2 mixed.
% betas_signed_wo_rewunc_obj_both.mat / _perceptual.mat (added to
% LR_analysis_preprint.m, see conversation from 2026-09-22) are the same
% signed-PE regression refit separately on each condition's trials, so
% fixedLR/adaptiveLR get the same aggregate + per-condition treatment as
% ecoperf/absEstError above.
betas_signed = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'betas_signed_wo_rewunc_obj.mat'));
fixedLR = betas_signed(:, 1);
adaptiveLR = betas_signed(:, 2);

betas_signed_Both = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'betas_signed_wo_rewunc_obj_both.mat'));
fixedLR_Both = betas_signed_Both(:, 1);
adaptiveLR_Both = betas_signed_Both(:, 2);

betas_signed_Perceptual = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'betas_signed_wo_rewunc_obj_perceptual.mat'));
fixedLR_Perceptual = betas_signed_Perceptual(:, 1);
adaptiveLR_Perceptual = betas_signed_Perceptual(:, 2);

%% =================== LOAD FITTED MODEL PARAMETERS =========================
[~, ~, uniqueID, ~] = fitSlider_ALLmodels.load_fitting_data();

% Both subj_ids and uniqueID are supposed to be the same 98 subjects in
% the same order (see design notes above) -- fail loudly here rather than
% silently pairing the wrong subject's behavioral measures with the wrong
% subject's fitted parameters.
assert(isequal(subj_ids, uniqueID), ...
    ['Subject ID mismatch between LR-analysis data (subj_ids) and ' ...
    'model-fitting data (uniqueID) -- do not proceed until this is resolved.']);

if fix_sigma
    params_basicRL = importdata(fullfile(save_dir, 'params_basicRL_fixedSigma_Both_multiSP.mat'));
    params_RLsigma = importdata(fullfile(save_dir, 'params_RLsigma_fixedSigma_Both_multiSP.mat'));
    params_bayesianAgent = importdata(fullfile(save_dir, 'params_bayesianAgent_fixedSigma_Both_multiSP.mat'));

    perceptualSigma = importdata(fullfile(save_dir, 'sigma_perceptualChoice.mat'));
    fixedSigma = perceptualSigma.sigma;
    assert(length(fixedSigma) == length(uniqueID), ...
        'sigma_perceptualChoice.mat length does not match uniqueID -- do not proceed until this is resolved.');

    sigma_basicRL = fixedSigma;
    sigma_RLsigma = fixedSigma;
    sigma_bayesianAgent = fixedSigma;
else
    params_basicRL = importdata(fullfile(save_dir, 'params_basicRL_RBVoi_Both_multiSP.mat'));
    params_RLsigma = importdata(fullfile(save_dir, 'params_RLSigma_RBVoi_Both_multiSP.mat'));
    params_bayesianAgent = importdata(fullfile(save_dir, 'params_bayesianAgent_integratedBoth_multiSP.mat'));

    sigma_basicRL = params_basicRL.sigma;
    sigma_RLsigma = params_RLsigma.sigma;
    sigma_bayesianAgent = params_bayesianAgent.sigma;
end

%% =================== BUILD INDIVIDUAL-DIFFERENCES TABLE ===================
indivDiff = table();
indivDiff.SubjectID = subj_ids;
indivDiff.ecoperf = ecoperf;
indivDiff.ecoperf_Both = ecoperf_Both;
indivDiff.ecoperf_Perceptual = ecoperf_Perceptual;
indivDiff.absEstError = absEstError;
indivDiff.absEstError_Both = absEstError_Both;
indivDiff.absEstError_Perceptual = absEstError_Perceptual;
indivDiff.fixedLR = fixedLR;
indivDiff.fixedLR_Both = fixedLR_Both;
indivDiff.fixedLR_Perceptual = fixedLR_Perceptual;
indivDiff.adaptiveLR = adaptiveLR;
indivDiff.adaptiveLR_Both = adaptiveLR_Both;
indivDiff.adaptiveLR_Perceptual = adaptiveLR_Perceptual;
indivDiff.basicRL_alpha = params_basicRL.alpha;
indivDiff.basicRL_kappa = params_basicRL.kappa;
indivDiff.basicRL_sigma = sigma_basicRL;
indivDiff.RLsigma_alpha = params_RLsigma.alpha;
indivDiff.RLsigma_kappa = params_RLsigma.kappa;
indivDiff.RLsigma_sigma = sigma_RLsigma;
indivDiff.bayesianAgent_kappa = params_bayesianAgent.kappa;
indivDiff.bayesianAgent_sigma = sigma_bayesianAgent;

safe_saveall(fullfile(save_dir, 'individualDifferences_table.mat'), indivDiff);

%% =================== CORRELATE EACH MEASURE AGAINST EACH FITTED PARAM =====
behavioral_measures = {'ecoperf', 'absEstError', 'fixedLR', 'adaptiveLR'};
behavioral_labels = {'Economic performance', 'Abs. estimation error', 'Fixed LR', 'Adaptive LR'};

model_params = {'basicRL_alpha', 'basicRL_kappa', 'basicRL_sigma', ...
    'RLsigma_alpha', 'RLsigma_kappa', 'RLsigma_sigma', ...
    'bayesianAgent_kappa', 'bayesianAgent_sigma'};
model_param_labels = {'basicRL: alpha', 'basicRL: kappa', 'basicRL: sigma', ...
    'RLsigma: alpha', 'RLsigma: kappa', 'RLsigma: sigma', ...
    'bayesianAgent: kappa', 'bayesianAgent: sigma'};

n_params = length(model_params);

% All four measures now get their own heatmap figure per condition (e.g.
% ecoperf_Both / ecoperf_Perceptual), since all four have
% condition-specific values (see design notes above). Model parameters on
% the y-axis are still the "Both"-condition fits loaded above either way
% -- no Perceptual-condition fitted parameters are loaded in this script
% yet. Each heatmap is a single row (this measure) x 8 columns (model
% params) -- not a square/symmetric matrix, so there's no diagonal to
% mask; every cell is shown, colored by r and annotated with r/p text.
corr_measure_fields = {'ecoperf_Both', 'ecoperf_Perceptual', 'absEstError_Both', 'absEstError_Perceptual', ...
    'fixedLR_Both', 'fixedLR_Perceptual', 'adaptiveLR_Both', 'adaptiveLR_Perceptual'};
corr_measure_labels = {'Economic performance (Both)', 'Economic performance (Perceptual)', ...
    'Abs. estimation error (Both)', 'Abs. estimation error (Perceptual)', ...
    'Fixed LR (Both)', 'Fixed LR (Perceptual)', 'Adaptive LR (Both)', 'Adaptive LR (Perceptual)'};

fprintf('%-30s %-22s %8s %10s\n', 'Behavioral measure', 'Model parameter', 'r', 'p');
fprintf('%s\n', repmat('-', 1, 74));
for m = 1:length(corr_measure_fields)
    r_values = NaN(1, n_params);
    p_values = NaN(1, n_params);
    for p = 1:n_params
        x = indivDiff.(corr_measure_fields{m});
        y = indivDiff.(model_params{p});
        [r, pval] = corr(x, y, 'rows', 'complete', 'Type', 'Spearman');
        r_values(p) = r;
        p_values(p) = pval;
        fprintf('%-30s %-22s %8.3f %10.4f\n', corr_measure_labels{m}, model_param_labels{p}, r, pval);
    end

    figure('Position', [100 100 800 150]);
    plot_corr_heatmap(r_values, p_values, model_param_labels, corr_measure_labels{m});
    title(corr_measure_labels{m}, 'FontWeight', 'bold');
end

%% =================== MEAN +/- SEM + DOTS BY BEST-FITTING MODEL ============
%  For each condition (Both, Perceptual), subjects are split into three
%  groups by which model best fits them (delta_BIC-based comparison saved
%  in bestModel_*_multiSP.mat by fitReducedModelSpace.m's /
%  fitReducedModelSpace_fixedSigma.m's "Plot proportion of subjects best
%  described by each model" sections -- respects the same fix_sigma
%  toggle used above for the fitted parameters). Each group's mean +/- SEM
%  is shown with individual subject dots overlaid, for each of the four
%  behavioral measures.
%
%  This mirrors the mean+SEM+jittered-dots convention used elsewhere in
%  this repo (Figures/plots/bar_plots_pval.m: bar height, then scatter
%  dots with 'XJitter','randn', then errorbar, using colors_rgb()'s
%  gray_dots/dots_edges for the dots) -- implemented directly here rather
%  than via that helper, since bar_plots_pval.m assumes every group has
%  the same number of data points (n) sharing one x-axis (a within-subject
%  repeated-measures design). Best-fit-model groups are naturally
%  unequal-sized, between-subjects groups, which that helper can't take
%  directly without padding.
group_order = {'basicRL', 'RLsigma', 'BayesianAgent'};
group_labels = {'Basic RL', 'RL modulation', 'Agent'};
conditions = {'Both', 'Perceptual'};

[~,~,~,~,~,~,~,mix_color,perc_color,~,~,~,~,~,~,~,~,~,~,~] = colors_rgb();
condition_colors = {mix_color, perc_color};

if fix_sigma
    bestModel_files = {'bestModel_fixedSigma_Both_multiSP.mat', 'bestModel_fixedSigma_Perceptual_multiSP.mat'};
else
    bestModel_files = {'bestModel_Both_multiSP.mat', 'bestModel_Perceptual_multiSP.mat'};
end

for c = 1:length(conditions)
    condModel_table = importdata(fullfile(save_dir, bestModel_files{c}));
    assert(isequal(condModel_table.SubjectID, indivDiff.SubjectID), ...
        sprintf(['Subject ID mismatch between %s and the individual-' ...
        'differences table -- do not proceed until this is resolved.'], bestModel_files{c}));

    figure('Position', [100 100 1400 350]);
    sgtitle(sprintf('%s condition', conditions{c}), 'FontWeight', 'bold');
    for m = 1:length(behavioral_measures)
        % ecoperf and absEstError have condition-specific columns
        % (indivDiff.ecoperf_Both/.ecoperf_Perceptual,
        % .absEstError_Both/.absEstError_Perceptual); fixedLR/adaptiveLR
        % are still only available as an aggregate-across-conditions
        % value for now (see design notes above). Check for the
        % condition-specific column dynamically rather than hardcoding
        % which measures have one, so this doesn't need to change again
        % once fixedLR/adaptiveLR get their own per-condition columns.
        cond_field = [behavioral_measures{m} '_' conditions{c}];
        has_cond_field = any(strcmp(indivDiff.Properties.VariableNames, cond_field));
        if has_cond_field
            measure_vals = indivDiff.(cond_field);
        else
            measure_vals = indivDiff.(behavioral_measures{m});
        end

        subplot(1, length(behavioral_measures), m);
        plot_mean_sem_dots(measure_vals, condModel_table.best_model, ...
            group_order, group_labels, condition_colors{c}, behavioral_labels{m});

        % One-way ANOVA comparing this measure across the three
        % best-fit-model groups -- run for any measure with a
        % condition-specific column (currently ecoperf, absEstError; see
        % design notes above), since only those are genuinely
        % condition-specific rather than the same aggregate value reused
        % under both conditions' groupings. anova1's vector + group-label
        % form (measure_vals paired with condModel_table.best_model)
        % handles the groups' unequal N directly, with no need to reshape
        % into equal-length columns. 'off' suppresses anova1's own
        % automatic boxplot figure, since plot_mean_sem_dots above already
        % shows this comparison.
        if has_cond_field
            [p_anova, anova_tbl, ~] = anova1(measure_vals, condModel_table.best_model, 'off');
            F_anova = anova_tbl{2, 5};
            df_between = anova_tbl{2, 3};
            df_error = anova_tbl{3, 3};
            fprintf('[%s] One-way ANOVA, %s across best-fit-model groups: F(%d,%d) = %.3f, p = %.4f\n', ...
                conditions{c}, behavioral_measures{m}, df_between, df_error, F_anova, p_anova);

            current_title = get(get(gca, 'Title'), 'String');
            title(sprintf('%s\nANOVA: F(%d,%d) = %.2f, p = %.4f', current_title, df_between, df_error, F_anova, p_anova), ...
                'FontWeight', 'normal');
        end
    end
end

%% ========================================================================
%  Local function: correlation heatmap (r color, r/p text per cell)
%  ------------------------------------------------------------------------
%  One measure (single row) x N model parameters (columns): each cell is
%  colored by Spearman r (diverging blue-white-red, centered at 0) and
%  annotated with its own r and p value. Not a square/symmetric matrix
%  (rows and columns are different variable sets), so there's no
%  self-correlation diagonal to mask -- every cell is a genuine
%  measure-vs-parameter correlation. Plots into whichever axes are
%  already current.
%  ========================================================================
function plot_corr_heatmap(r_values, p_values, col_labels, row_label)
    n_cols = length(r_values);

    imagesc(r_values);
    colormap(gca, diverging_colormap());
    clim([-1 1]);
    colorbar;

    set(gca, 'XTick', 1:n_cols, 'XTickLabel', col_labels, 'XTickLabelRotation', 45, ...
        'YTick', 1, 'YTickLabel', {row_label});

    for c = 1:n_cols
        if p_values(c) < 0.001
            p_str = 'p < .001';
        else
            p_str = sprintf('p = %.3f', p_values(c));
        end
        if abs(r_values(c)) > 0.5
            txt_color = 'w';
        else
            txt_color = 'k';
        end
        text(c, 1, sprintf('r = %.2f\n%s', r_values(c), p_str), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', txt_color, 'FontSize', 9);
    end
    box on;
end

%% ========================================================================
%  Local function: diverging blue-white-red colormap
%  ------------------------------------------------------------------------
%  Simple dependency-free diverging colormap for correlation heatmaps
%  (negative = blue, zero = white, positive = red), built via linear
%  interpolation rather than relying on a toolbox-specific colormap
%  function (e.g. redbluecmap, which needs the Bioinformatics Toolbox).
%  ========================================================================
function cmap = diverging_colormap()
    n = 256;
    cmap = interp1([1 n/2 n], [0.2 0.4 0.8; 1 1 1; 0.8 0.2 0.2], 1:n);
end

%% ========================================================================
%  Local function: group mean +/- SEM with individual subject dots
%  ------------------------------------------------------------------------
%  One behavioral measure split into three best-fitting-model groups
%  (naturally unequal N per group): a bar at each group's mean, its SEM as
%  a black errorbar, and every subject's own value as a jittered dot on
%  top -- same visual convention (dot color/jitter) as
%  Figures/plots/bar_plots_pval.m, built directly here since that helper
%  assumes equal N per group. Plots into whichever axes are already
%  current (a subplot set up by the caller).
%  ========================================================================
function plot_mean_sem_dots(values, groups, group_order, group_labels, bar_color, y_label)
    n_groups = length(group_order);
    group_mean = NaN(n_groups, 1);
    group_sem = NaN(n_groups, 1);
    group_n = NaN(n_groups, 1);
    group_vals = cell(n_groups, 1);

    for g = 1:n_groups
        vals_g = values(strcmp(groups, group_order{g}));
        vals_g = vals_g(~isnan(vals_g));
        group_vals{g} = vals_g;
        group_mean(g) = mean(vals_g);
        group_sem(g) = std(vals_g) / sqrt(length(vals_g));
        group_n(g) = length(vals_g);
    end

    hold on;
    bar(1:n_groups, group_mean, 0.5, 'FaceColor', bar_color, 'FaceAlpha', 0.5, 'EdgeColor', bar_color);

    for g = 1:n_groups
        scatter(repmat(g, group_n(g), 1), group_vals{g}, 40, 'o', ...
            'MarkerEdgeColor', [184 184 184]/255, 'MarkerFaceColor', [220 220 220]/255, ...
            'XJitter', 'randn', 'XJitterWidth', 0.1);
    end

    errorbar(1:n_groups, group_mean, group_sem, 'k', 'linestyle', 'none', 'LineWidth', 1.2);

    set(gca, 'XTick', 1:n_groups, 'XTickLabel', group_labels);
    box off;
    ylabel(y_label);
    title(sprintf('n = %d, %d, %d', group_n(1), group_n(2), group_n(3)), 'FontWeight', 'normal');
end
