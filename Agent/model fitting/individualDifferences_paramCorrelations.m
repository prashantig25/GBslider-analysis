%% ========================================================================
%  Script: Individual Differences -- behavior/LR vs. fitted model parameters
%  ------------------------------------------------------------------------
%  Correlates seven subject-level measures against each subject's fitted
%  parameters from the three reduced-model-space models (basicRL, RLsigma,
%  bayesianAgent; free-sigma fits from fitReducedModelSpace.m, "Both"
%  condition):
%    1) Economic performance (ecoperf)
%    2) Absolute estimation error (absEstError)
%    3) Signed estimation error (signedEstError)
%    4) Fixed LR (signed-PE regression, beta_1)
%    5) Adaptive LR (signed-PE regression, beta_2)
%    6) Signed LR (signed-PE regression, same beta_1 as Fixed LR -- see
%       design notes below)
%    7) Absolute LR (absolute-PE regression, beta_1)
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
%  - subjective_pe below additionally toggles between the objective-PE and
%    subjective-PE definitions of basicRL/RLsigma (see
%    fitReducedModelSpace_fixedSigma.m's "SUBJECTIVE PE VARIANT (BOTH)"
%    design notes). bayesianAgent has no subjective-PE variant, so when
%    subjective_pe is true it's dropped entirely -- no bayesianAgent
%    columns in indivDiff, no bayesianAgent row in the correlation
%    heatmaps, and the best-fit-model groups below become {basicRL_
%    subjectivePE, RLsigma_subjectivePE, Ambiguous} instead of {basicRL,
%    RLsigma, BayesianAgent, Ambiguous}. Both fitReducedModelSpace.m and
%    fitReducedModelSpace_fixedSigma.m now always fit a subjective-PE
%    variant (their "SECTION 2 OF 4"), so this works with either fix_sigma
%    state.
%  - no_integration below additionally toggles between the integrated and
%    no-integration (point-observation) definitions of basicRL/RLsigma --
%    see fitSlider_ALLmodels.m's nll_basicRL_pointObs header, and either
%    fitting script's "SECTION 3/4 OF 4". Combines independently with
%    subjective_pe (4 combinations total). For the subjective-PE
%    combination, bayesianAgent has no no-integration variant either (it
%    has no subjective-PE variant at all), so it's dropped the same way
%    subjective_pe drops it, and the best-fit-model groups become
%    {basicRL_subjectivePE_noIntegration, RLsigma_subjectivePE_
%    noIntegration, Ambiguous}. For the OBJECTIVE-PE, no-integration
%    combination, bayesianAgent DOES have a no-integration variant
%    (nll_bayesianAgent_pointObs), but only in the free-sigma pipeline
%    (fitReducedModelSpace.m's Section 3 fits it; fitReducedModelSpace_
%    fixedSigma.m's Section 3 does not) -- see has_bayesianAgent's
%    definition below, which accounts for this fix_sigma-dependent split.
%    So the best-fit-model groups there are {basicRL_noIntegration,
%    RLsigma_noIntegration, BayesianAgent_noIntegration, Ambiguous} when
%    fix_sigma is false, or {basicRL_noIntegration, RLsigma_noIntegration,
%    Ambiguous} (no agent) when fix_sigma is true.
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
fix_sigma = false;

% subjective_pe toggles which PE (prediction-error) definition's fitted
% parameters get loaded, for basicRL/RLsigma only -- bayesianAgent has no
% subjective-PE variant, so when true it's dropped from everything below
% (indivDiff columns, correlation heatmaps, best-fit-model groups).
%   false (default) - objective PE: the model's own internally-tracked
%                      belief drives each trial's delta-rule update.
%   true             - subjective PE: each trial's delta-rule anchors to
%                      the subject's own actual previous slider report
%                      (mu_hat(n-1)) instead -- see
%                      fitReducedModelSpace_fixedSigma.m's "SUBJECTIVE PE
%                      VARIANT (BOTH)" design notes.
% Works with either fix_sigma state: fitReducedModelSpace_fixedSigma.m
% (fix_sigma=true) and fitReducedModelSpace.m (fix_sigma=false) each have
% their own subjective-PE fitting block, with sigma fixed or free to
% match that script's convention for every other model, and their own
% output filenames (the free-sigma ones have no "fixedSigma" in the name).
subjective_pe = true;

% no_integration toggles which observation-handling definition's fitted
% parameters get loaded, for basicRL/RLsigma only -- bayesianAgent has no
% no-integration variant either (its belief update goes through the
% Agent class's own internal mechanism, not the set_o grid-integration
% this toggle concerns), so when true it's dropped from everything below,
% same as subjective_pe.
%   false (default) - integrated: belief comes from marginalizing over a
%                      discretized grid of hypothetical observations
%                      (nll_basicRL_integrated / nll_RLsigma_VOI, or their
%                      subjectivePE variants).
%   true             - no integration: belief comes from a single point
%                      estimate of the actual observed condiff instead
%                      (nll_basicRL_pointObs / nll_RLsigma, or their
%                      subjectivePE variants) -- see fitSlider_
%                      ALLmodels.m's nll_basicRL_pointObs header for the
%                      full rationale.
% Combines with subjective_pe independently (4 combinations total), and
% works with either fix_sigma state -- both fitting scripts now always
% produce all four combinations' output files every run (see their
% "SECTION 1-4 OF 4" structure), each under its own filename.
no_integration = false;

% bayesianAgent has no subjective-PE variant at all, so subjective_pe
% always excludes it. It DOES now have a no-integration variant, but only
% in the free-sigma pipeline (fitReducedModelSpace.m's Section 3 fits
% nll_bayesianAgent_pointObs for objective PE, no integration) --
% fitReducedModelSpace_fixedSigma.m's Section 3 has no such fit, so
% fix_sigma && no_integration still excludes it.
has_bayesianAgent = ~subjective_pe && (~no_integration || ~fix_sigma);

%% =================== LOAD BEHAVIORAL / LR MEASURES =======================
data = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'preprocessed_data.mat'));
subj_ids = unique(data.ID);

% preprocessed_data.mat mixes conditions 1 ("Both") and 2 ("Perceptual")
% together (only condition 3, "Reward", was dropped upstream) -- see
% conversation from 2026-09-22. ecoperf, absEstError, and signedEstError
% are therefore computed both as the aggregate-across-both-conditions
% value (unchanged, still used by the correlation section below) AND
% separately per condition (used by the MEAN +/- SEM + DOTS section, so
% that section's per-condition subplots show genuinely condition-specific
% values rather than the same aggregate value reused under two different
% best-fit-model groupings). fixedLR/adaptiveLR/signedLR get the same
% treatment just below. signedEstError is calculateEstimationError's 3rd
% output (signed difference from actual mu, i.e. not absolute-valued),
% previously discarded with ~ here.
[ecoperf, absEstError, signedEstError] = calculateEstimationError(data, subj_ids);
[ecoperf_Both, absEstError_Both, signedEstError_Both] = calculateEstimationError(data(data.condition == 1, :), subj_ids);
[ecoperf_Perceptual, absEstError_Perceptual, signedEstError_Perceptual] = calculateEstimationError(data(data.condition == 2, :), subj_ids);

% betas_signed_wo_rewunc_obj.mat is likewise fit on conditions 1+2 mixed.
% betas_signed_wo_rewunc_obj_both.mat / _perceptual.mat (added to
% LR_analysis_preprint.m, see conversation from 2026-09-22) are the same
% signed-PE regression refit separately on each condition's trials, so
% fixedLR/adaptiveLR get the same aggregate + per-condition treatment as
% ecoperf/absEstError above. "Signed LR" is the same column 1 as "Fixed
% LR" (both are the signed-PE model's main pe effect) -- exposed under
% its own label since it's requested alongside "Absolute LR" as a
% signed-vs-absolute pairing, distinct from the Fixed/Adaptive
% decomposition of the signed model into interaction terms.
betas_signed = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'betas_signed_wo_rewunc_obj.mat'));
fixedLR = betas_signed(:, 1);
adaptiveLR = betas_signed(:, 2);
signedLR = betas_signed(:, 1);

betas_signed_Both = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'betas_signed_wo_rewunc_obj_both.mat'));
fixedLR_Both = betas_signed_Both(:, 1);
adaptiveLR_Both = betas_signed_Both(:, 2);
signedLR_Both = betas_signed_Both(:, 1);

betas_signed_Perceptual = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'betas_signed_wo_rewunc_obj_perceptual.mat'));
fixedLR_Perceptual = betas_signed_Perceptual(:, 1);
adaptiveLR_Perceptual = betas_signed_Perceptual(:, 2);
signedLR_Perceptual = betas_signed_Perceptual(:, 1);

% betas_abs_wo_rewunc_obj.mat is the same formula/regression, but fit on
% abs(pe)/abs(up) instead of signed pe/up (see LR_analysis_preprint.m's
% "FIT ALL VERSIONS OF THE ABSOLUTE MODEL" section) -- column 1 is the
% main abs(pe) effect ("Absolute LR"). No condition-specific (_both /
% _perceptual) refit of the absolute model exists yet (only the signed
% model got that treatment above), so absoluteLR is aggregate-only for
% now.
betas_abs = importdata(fullfile(desiredPath, 'Data', 'LR analyses', 'betas_abs_wo_rewunc_obj.mat'));
absoluteLR = betas_abs(:, 1);

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
    if subjective_pe && no_integration
        params_basicRL = importdata(fullfile(save_dir, 'params_basicRL_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat'));
        params_RLsigma = importdata(fullfile(save_dir, 'params_RLsigma_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat'));
    elseif subjective_pe
        params_basicRL = importdata(fullfile(save_dir, 'params_basicRL_subjectivePEv2_fixedSigma_Both_multiSP.mat'));
        params_RLsigma = importdata(fullfile(save_dir, 'params_RLsigma_subjectivePEv2_fixedSigma_Both_multiSP.mat'));
    elseif no_integration
        params_basicRL = importdata(fullfile(save_dir, 'params_basicRL_noIntegration_fixedSigma_Both_multiSP.mat'));
        params_RLsigma = importdata(fullfile(save_dir, 'params_RLsigma_noIntegration_fixedSigma_Both_multiSP.mat'));
    else
        params_basicRL = importdata(fullfile(save_dir, 'params_basicRL_fixedSigma_Both_multiSP.mat'));
        params_RLsigma = importdata(fullfile(save_dir, 'params_RLsigma_fixedSigma_Both_multiSP.mat'));
        params_bayesianAgent = importdata(fullfile(save_dir, 'params_bayesianAgent_fixedSigma_Both_multiSP.mat'));
    end

    perceptualSigma = importdata(fullfile(save_dir, 'sigma_perceptualChoice.mat'));
    fixedSigma = perceptualSigma.sigma;
    assert(length(fixedSigma) == length(uniqueID), ...
        'sigma_perceptualChoice.mat length does not match uniqueID -- do not proceed until this is resolved.');

    sigma_basicRL = fixedSigma;
    sigma_RLsigma = fixedSigma;
    if has_bayesianAgent
        sigma_bayesianAgent = fixedSigma;
    end
else
    if subjective_pe && no_integration
        params_basicRL = importdata(fullfile(save_dir, 'params_basicRL_subjectivePE_noIntegration_Both_multiSP.mat'));
        params_RLsigma = importdata(fullfile(save_dir, 'params_RLsigma_subjectivePE_noIntegration_Both_multiSP.mat'));
    elseif subjective_pe
        params_basicRL = importdata(fullfile(save_dir, 'params_basicRL_subjectivePEv2_Both_multiSP.mat'));
        params_RLsigma = importdata(fullfile(save_dir, 'params_RLsigma_subjectivePEv2_Both_multiSP.mat'));
    elseif no_integration
        params_basicRL = importdata(fullfile(save_dir, 'params_basicRL_noIntegration_Both_multiSP.mat'));
        params_RLsigma = importdata(fullfile(save_dir, 'params_RLsigma_noIntegration_Both_multiSP.mat'));
        % Only the free-sigma pipeline has a no-integration bayesianAgent
        % fit (fitReducedModelSpace.m's Section 3) -- has_bayesianAgent is
        % only true here when ~fix_sigma, so this always resolves safely.
        params_bayesianAgent = importdata(fullfile(save_dir, 'params_bayesianAgent_noIntegration_Both_multiSP.mat'));
    else
        params_basicRL = importdata(fullfile(save_dir, 'params_basicRL_RBVoi_Both_multiSP.mat'));
        params_RLsigma = importdata(fullfile(save_dir, 'params_RLSigma_RBVoi_Both_multiSP.mat'));
        params_bayesianAgent = importdata(fullfile(save_dir, 'params_bayesianAgent_integratedBoth_multiSP.mat'));
    end

    % Sigma is free here regardless of subjective_pe/no_integration
    % (fitReducedModelSpace.m always fits it jointly with alpha/kappa), so
    % it comes straight from each model's own .sigma field -- unlike the
    % fix_sigma branch above, there's no shared sigma_perceptualChoice.mat
    % to read instead.
    sigma_basicRL = params_basicRL.sigma;
    sigma_RLsigma = params_RLsigma.sigma;
    if has_bayesianAgent
        sigma_bayesianAgent = params_bayesianAgent.sigma;
    end
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
indivDiff.signedEstError = signedEstError;
indivDiff.signedEstError_Both = signedEstError_Both;
indivDiff.signedEstError_Perceptual = signedEstError_Perceptual;
indivDiff.fixedLR = fixedLR;
indivDiff.fixedLR_Both = fixedLR_Both;
indivDiff.fixedLR_Perceptual = fixedLR_Perceptual;
indivDiff.adaptiveLR = adaptiveLR;
indivDiff.adaptiveLR_Both = adaptiveLR_Both;
indivDiff.adaptiveLR_Perceptual = adaptiveLR_Perceptual;
indivDiff.signedLR = signedLR;
indivDiff.signedLR_Both = signedLR_Both;
indivDiff.signedLR_Perceptual = signedLR_Perceptual;
indivDiff.absoluteLR = absoluteLR;
indivDiff.basicRL_alpha = params_basicRL.alpha;
indivDiff.basicRL_kappa = params_basicRL.kappa;
indivDiff.basicRL_sigma = sigma_basicRL;
indivDiff.RLsigma_alpha = params_RLsigma.alpha;
indivDiff.RLsigma_kappa = params_RLsigma.kappa;
indivDiff.RLsigma_sigma = sigma_RLsigma;
if has_bayesianAgent
    indivDiff.bayesianAgent_kappa = params_bayesianAgent.kappa;
    indivDiff.bayesianAgent_sigma = sigma_bayesianAgent;
end

% Suffixed by PE variant, integration variant, and sigma pipeline (rather
% than sharing one filename across all eight toggle combinations) so
% toggling subjective_pe/no_integration doesn't trip safe_saveall's "data
% is different" dated-fallback save just because the column set or
% underlying fit changed.
if subjective_pe && no_integration
    variant_suffix = '_subjectivePE_noIntegration';
elseif subjective_pe
    variant_suffix = '_subjectivePE';
elseif no_integration
    variant_suffix = '_noIntegration';
else
    variant_suffix = '';
end
if fix_sigma
    sigma_suffix = '';
else
    sigma_suffix = '_freeSigma';
end
indivDiff_filename = sprintf('individualDifferences_table%s%s.mat', variant_suffix, sigma_suffix);
safe_saveall(fullfile(save_dir, indivDiff_filename), indivDiff);

%% =================== CORRELATE EACH MEASURE AGAINST EACH FITTED PARAM =====
behavioral_measures = {'ecoperf', 'absEstError', 'signedEstError', 'fixedLR', 'adaptiveLR', 'signedLR', 'absoluteLR'};
behavioral_labels = {'Economic performance', 'Abs. estimation error', 'Signed estimation error', ...
    'Fixed LR', 'Adaptive LR', 'Signed LR', 'Absolute LR'};

model_params = {'basicRL_alpha', 'basicRL_kappa', 'basicRL_sigma', ...
    'RLsigma_alpha', 'RLsigma_kappa', 'RLsigma_sigma'};
model_param_labels = {'basicRL: alpha', 'basicRL: kappa', 'basicRL: sigma', ...
    'RLsigma: alpha', 'RLsigma: kappa', 'RLsigma: sigma'};
if has_bayesianAgent
    model_params = [model_params, {'bayesianAgent_kappa', 'bayesianAgent_sigma'}];
    model_param_labels = [model_param_labels, {'bayesianAgent: kappa', 'bayesianAgent: sigma'}];
end

n_params = length(model_params);

% One combined heatmap per condition: all behavioral measures as rows of
% a single imagesc matrix sharing one x-axis (model params), rather than
% stacking separate per-measure axes. Measures with a condition-specific
% column (ecoperf, absEstError, signedEstError, fixedLR, adaptiveLR,
% signedLR) use that column (e.g. ecoperf_Both / ecoperf_Perceptual);
% absoluteLR has no condition-specific refit yet (see design notes
% above), so its single aggregate value is shown in both figures (same
% fallback-to-aggregate convention the MEAN +/- SEM + DOTS section below
% uses). Model parameters on the x-axis are still the "Both"-condition
% fits loaded above either way -- no Perceptual-condition fitted
% parameters are loaded in this script yet.
conditions_heatmap = {'Both', 'Perceptual'};

fprintf('%-30s %-22s %8s %10s\n', 'Behavioral measure', 'Model parameter', 'r', 'p');
fprintf('%s\n', repmat('-', 1, 74));
for c = 1:length(conditions_heatmap)
    cond_label = conditions_heatmap{c};
    r_matrix = NaN(length(behavioral_measures), n_params);
    p_matrix = NaN(length(behavioral_measures), n_params);
    row_labels = cell(1, length(behavioral_measures));

    for m = 1:length(behavioral_measures)
        cond_field = [behavioral_measures{m} '_' cond_label];
        has_cond_field = any(strcmp(indivDiff.Properties.VariableNames, cond_field));
        if has_cond_field
            x_field = cond_field;
            measure_label = sprintf('%s (%s)', behavioral_labels{m}, cond_label);
        else
            x_field = behavioral_measures{m};
            measure_label = behavioral_labels{m};
        end
        % Heatmap row label stays plain (no "(Both)"/"(Perceptual)") since
        % the figure title already states the condition; measure_label
        % keeps the suffix for the printed log below, where rows from both
        % conditions' runs end up interleaved in the same console output.
        row_labels{m} = behavioral_labels{m};

        for p = 1:n_params
            x = indivDiff.(x_field);
            y = indivDiff.(model_params{p});
            [r, pval] = corr(x, y, 'rows', 'complete', 'Type', 'Spearman');
            r_matrix(m, p) = r;
            p_matrix(m, p) = pval;
            fprintf('%-30s %-22s %8.3f %10.4f\n', measure_label, model_param_labels{p}, r, pval);
        end
    end

    figure('Position', [100 100 800 100+80*length(behavioral_measures)]);
    plot_corr_heatmap(r_matrix, p_matrix, model_param_labels, row_labels);
    title(sprintf('%s condition: Spearman correlations', cond_label), 'FontWeight', 'bold');
end

%% =================== MEAN +/- SEM + DOTS BY BEST-FITTING MODEL ============
%  For each condition (Both, Perceptual), subjects are split into four
%  groups: the three models (by which one best fits them, delta_BIC-based
%  comparison saved in bestModel_*_multiSP.mat by fitReducedModelSpace.m's
%  / fitReducedModelSpace_fixedSigma.m's "Plot proportion of subjects best
%  described by each model" sections -- respects the same fix_sigma
%  toggle used above for the fitted parameters), plus a fourth "Ambiguous"
%  group for subjects whose nominal winner doesn't clear the Kass &
%  Raftery (1995) delta-BIC >= 2 decisive threshold (bestModel_table's
%  decisive column) -- same cutoff plot_model_comparison uses for its own
%  "Ambiguous" bar, so a narrow win isn't credited to a model here either.
%  Each group's mean +/- SEM is shown with individual subject dots
%  overlaid, for each of the four behavioral measures.
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
% Group labels match the literal best_model strings saved by
% compute_model_comparison, which differ across the four PE/integration
% combinations: 'basicRL'/'RLsigma'/'BayesianAgent' for the objective-PE,
% integrated 3-way comparison; 'basicRL_subjectivePE'/'RLsigma_
% subjectivePE' for the subjective-PE, integrated 2-way comparison;
% 'basicRL_noIntegration'/'RLsigma_noIntegration'/'BayesianAgent_
% noIntegration' for the objective-PE, no-integration 3-way comparison
% (free-sigma pipeline only -- see has_bayesianAgent's definition above);
% 'basicRL_noIntegration'/'RLsigma_noIntegration' (2-way, no agent) for
% that same combination under the fixed-sigma pipeline;
% 'basicRL_subjectivePE_noIntegration'/'RLsigma_subjectivePE_
% noIntegration' for the subjective-PE, no-integration 2-way comparison
% (see fitReducedModelSpace.m / fitReducedModelSpace_fixedSigma.m's
% model_names / model_names_subjectivePE / model_names_noIntegration /
% model_names_subjectivePE_noIntegration).
if has_bayesianAgent && no_integration
    group_order = {'basicRL_noIntegration', 'RLsigma_noIntegration', 'BayesianAgent_noIntegration', 'Ambiguous'};
    group_labels = {'Basic RL (no integration)', 'RL modulation (no integration)', 'Agent (no integration)', 'Ambiguous'};
elseif has_bayesianAgent
    group_order = {'basicRL', 'RLsigma', 'BayesianAgent', 'Ambiguous'};
    group_labels = {'Basic RL', 'RL modulation', 'Agent', 'Ambiguous'};
elseif subjective_pe && no_integration
    group_order = {'basicRL_subjectivePE_noIntegration', 'RLsigma_subjectivePE_noIntegration', 'Ambiguous'};
    group_labels = {'Basic RL (subj PE, no int)', 'RL modulation (subj PE, no int)', 'Ambiguous'};
elseif subjective_pe
    group_order = {'basicRL_subjectivePE', 'RLsigma_subjectivePE', 'Ambiguous'};
    group_labels = {'Basic RL (subjective PE)', 'RL modulation (subjective PE)', 'Ambiguous'};
else % no_integration, objective PE, fixed-sigma pipeline (no agent variant there)
    group_order = {'basicRL_noIntegration', 'RLsigma_noIntegration', 'Ambiguous'};
    group_labels = {'Basic RL (no integration)', 'RL modulation (no integration)', 'Ambiguous'};
end
conditions = {'Both', 'Perceptual'};

[~,~,~,~,~,~,~,mix_color,perc_color,~,~,~,~,~,~,~,~,~,~,~] = colors_rgb();
condition_colors = {mix_color, perc_color};

% model_names_subjectivePE = {'basicRL_subjectivePE','RLsigma_subjectivePE'}
% (and the no-integration equivalents) are identical in both
% fitReducedModelSpace.m and fitReducedModelSpace_fixedSigma.m, so
% group_order/group_labels above don't need to branch on fix_sigma --
% only the filename (fixed-sigma vs free-sigma pipeline) does.
if subjective_pe && no_integration && fix_sigma
    bestModel_files = {'bestModel_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat', 'bestModel_subjectivePE_noIntegration_fixedSigma_Perceptual_multiSP.mat'};
elseif subjective_pe && no_integration && ~fix_sigma
    bestModel_files = {'bestModel_subjectivePE_noIntegration_Both_multiSP.mat', 'bestModel_subjectivePE_noIntegration_Perceptual_multiSP.mat'};
elseif subjective_pe && fix_sigma
    bestModel_files = {'bestModel_subjectivePEv2_fixedSigma_Both_multiSP.mat', 'bestModel_subjectivePEv2_fixedSigma_Perceptual_multiSP.mat'};
elseif subjective_pe && ~fix_sigma
    bestModel_files = {'bestModel_subjectivePEv2_Both_multiSP.mat', 'bestModel_subjectivePEv2_Perceptual_multiSP.mat'};
elseif no_integration && fix_sigma
    bestModel_files = {'bestModel_noIntegration_fixedSigma_Both_multiSP.mat', 'bestModel_noIntegration_fixedSigma_Perceptual_multiSP.mat'};
elseif no_integration && ~fix_sigma
    bestModel_files = {'bestModel_noIntegration_Both_multiSP.mat', 'bestModel_noIntegration_Perceptual_multiSP.mat'};
elseif fix_sigma
    bestModel_files = {'bestModel_fixedSigma_Both_multiSP.mat', 'bestModel_fixedSigma_Perceptual_multiSP.mat'};
else
    bestModel_files = {'bestModel_Both_multiSP.mat', 'bestModel_Perceptual_multiSP.mat'};
end

for c = 1:length(conditions)
    condModel_table = importdata(fullfile(save_dir, bestModel_files{c}));
    assert(isequal(condModel_table.SubjectID, indivDiff.SubjectID), ...
        sprintf(['Subject ID mismatch between %s and the individual-' ...
        'differences table -- do not proceed until this is resolved.'], bestModel_files{c}));

    % Relabel non-decisive subjects 'Ambiguous' rather than crediting
    % their narrow nominal winner -- see design notes above.
    best_model_grouped = condModel_table.best_model;
    best_model_grouped(~condModel_table.decisive) = {'Ambiguous'};

    figure('Position', [100 100 1400 350]);
    sgtitle(sprintf('%s condition', conditions{c}), 'FontWeight', 'bold');
    for m = 1:length(behavioral_measures)
        % ecoperf, absEstError, signedEstError, fixedLR, adaptiveLR, and
        % signedLR all have condition-specific columns (e.g.
        % indivDiff.ecoperf_Both/.ecoperf_Perceptual); absoluteLR is still
        % only available as an aggregate-across-conditions value for now
        % (see design notes above). Check for the condition-specific
        % column dynamically rather than hardcoding which measures have
        % one, so this doesn't need to change again once absoluteLR gets
        % its own per-condition columns.
        cond_field = [behavioral_measures{m} '_' conditions{c}];
        has_cond_field = any(strcmp(indivDiff.Properties.VariableNames, cond_field));
        if has_cond_field
            measure_vals = indivDiff.(cond_field);
        else
            measure_vals = indivDiff.(behavioral_measures{m});
        end

        subplot(1, length(behavioral_measures), m);
        plot_mean_sem_dots(measure_vals, best_model_grouped, ...
            group_order, group_labels, condition_colors{c}, behavioral_labels{m});

        % One-way ANOVA comparing this measure across the four
        % best-fit-model groups (three models plus Ambiguous) -- run for
        % any measure with a condition-specific column (currently
        % ecoperf, absEstError, signedEstError, fixedLR, adaptiveLR,
        % signedLR; see design notes above), since only those are
        % genuinely condition-specific rather than the same aggregate
        % value reused under both conditions' groupings. anova1's vector +
        % group-label form (measure_vals paired with best_model_grouped)
        % handles the groups' unequal N directly, with no need to reshape
        % into equal-length columns. 'off' suppresses anova1's own
        % automatic boxplot figure, since plot_mean_sem_dots above already
        % shows this comparison.
        if has_cond_field
            [p_anova, anova_tbl, ~] = anova1(measure_vals, best_model_grouped, 'off');
            F_anova = anova_tbl{2, 5};
            df_between = anova_tbl{2, 3};
            df_error = anova_tbl{3, 3};
            fprintf('[%s] One-way ANOVA, %s across best-fit-model groups: F(%d,%d) = %.3f, p = %.4f\n', ...
                conditions{c}, behavioral_measures{m}, df_between, df_error, F_anova, p_anova);

            % Kept short (one line, no df/"ANOVA" label) since 7 subplots
            % share a 1400px-wide figure -- a longer title overflows its
            % own narrow subplot and overlaps the neighboring one. Full
            % stats (including df) are in the fprintf log above.
            current_title = get(get(gca, 'Title'), 'String');
            title(sprintf('%s | F=%.1f, p=%.3f', current_title, F_anova, p_anova), ...
                'FontWeight', 'normal', 'FontSize', 8);
        end
    end
end

%% ========================================================================
%  Local function: correlation heatmap (r color, r/p text per cell)
%  ------------------------------------------------------------------------
%  N measures (rows) x N model parameters (columns), sharing one x-axis:
%  each cell is colored by Spearman r (diverging blue-white-red, centered
%  at 0) and annotated with its own r and p value. Not a square/symmetric
%  matrix (rows and columns are different variable sets), so there's no
%  self-correlation diagonal to mask -- every cell is a genuine
%  measure-vs-parameter correlation. Plots into whichever axes are
%  already current.
%  ========================================================================
function plot_corr_heatmap(r_values, p_values, col_labels, row_labels)
    [n_rows, n_cols] = size(r_values);

    imagesc(r_values);
    colormap(gca, diverging_colormap());
    clim([-1 1]);
    colorbar;

    set(gca, 'XTick', 1:n_cols, 'XTickLabel', col_labels, 'XTickLabelRotation', 45, ...
        'YTick', 1:n_rows, 'YTickLabel', row_labels);

    for r = 1:n_rows
        for c = 1:n_cols
            if p_values(r,c) < 0.001
                p_str = 'p < .001';
            else
                p_str = sprintf('p = %.3f', p_values(r,c));
            end
            if abs(r_values(r,c)) > 0.5
                txt_color = 'w';
            else
                txt_color = 'k';
            end
            text(c, r, sprintf('r = %.2f\n%s', r_values(r,c), p_str), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'Color', txt_color, 'FontSize', 9);
        end
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
    title(sprintf('n=%s', strjoin(string(group_n), ',')), 'FontWeight', 'normal', 'FontSize', 8);
end
