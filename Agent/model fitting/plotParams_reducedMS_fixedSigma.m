%% ========================================================================
%  Script: Plot Fitted Parameters (Reduced Model Space, Sigma Fixed)
%  ------------------------------------------------------------------------
%  Plots alpha/kappa (fitted) and sigma (fixed, from each subject's own
%  perceptual-choice fit -- see fitReducedModelSpace_fixedSigma.m) plus
%  AIC/BIC/delta-BIC summaries for the basicRL, RLsigma, and BayesianAgent
%  model fits produced by fitReducedModelSpace_fixedSigma.m, for both the
%  "both" and "perceptual" condition fits.
%
%  Unlike plotParams_reducedMS.m's sigma plot (independently estimated per
%  model, so the three columns can differ), sigma here is pinned to the
%  same per-subject value for all three models (fitReducedModelSpace_
%  fixedSigma.m sets sigmaParameter(n) = fixedSigma(n) identically for
%  basicRL/RLsigma/bayesianAgent), so the three columns are expected to be
%  identical -- this plot is really just a sanity check that they are, and
%  a look at the fixed-sigma distribution across subjects.
%  ========================================================================
clc; clearvars;

% PATH STUFF -- anchor reads to Agent/model fitting/, where
% fitReducedModelSpace_fixedSigma.m saves its output, regardless of
% MATLAB's current folder.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)'; % to which directory one must save in
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');

% Filenames match fitReducedModelSpace_fixedSigma.m's multi-start
% (multiSP) outputs.
plot_fitted_params_fixedSigma('Both', ...
    fullfile(save_dir, 'params_basicRL_fixedSigma_Both_multiSP.mat'), ...
    fullfile(save_dir, 'params_RLsigma_fixedSigma_Both_multiSP.mat'), ...
    fullfile(save_dir, 'params_bayesianAgent_fixedSigma_Both_multiSP.mat'), ...
    fullfile(save_dir, 'AICBIC_fixedSigma_Both_reducedMS_multiSP.mat'));

plot_fitted_params_fixedSigma('Perceptual', ...
    fullfile(save_dir, 'params_basicRL_fixedSigma_Perceptual_multiSP.mat'), ...
    fullfile(save_dir, 'params_RLsigma_fixedSigma_Perceptual_multiSP.mat'), ...
    fullfile(save_dir, 'params_bayesianAgent_fixedSigma_Perceptual_multiSP.mat'), ...
    fullfile(save_dir, 'AICBIC_fixedSigma_Perceptual_reducedMS_multiSP.mat'));

%% ========================================================================
function plot_fitted_params_fixedSigma(condition_label, basicRL_file, RLsigma_file, bayesianAgent_file, AICBIC_file)
    basicRL = importdata(basicRL_file);
    RLsigma = importdata(RLsigma_file);
    bayesianAgent = importdata(bayesianAgent_file);

    n = length(basicRL.alpha); % number of subjects
    model_names = {'basicRL', 'RLsigma', 'BayesianAgent'};
    all_facecolors = lines(3);

    % bayesianAgent has no alpha parameter (fitted)
    alpha_all = [basicRL.alpha, RLsigma.alpha, NaN(n,1)];
    kappa_all = [basicRL.kappa, RLsigma.kappa, bayesianAgent.kappa];
    % sigma is fixed (not fitted) and identical across all three models
    % by construction -- see design notes above.
    sigma_all = [basicRL.sigma, RLsigma.sigma, bayesianAgent.sigma];

    param_names = {'alpha', 'kappa', 'sigma'};
    param_titles = {'alpha (fitted)', 'kappa (fitted)', 'sigma (fixed, from perceptual-choice fit)'};
    for p = 1:length(param_names)
        switch param_names{p}
            case 'alpha'
                y = alpha_all;
            case 'kappa'
                y = kappa_all;
            case 'sigma'
                y = sigma_all;
        end

        % Remove columns that are all NaN (e.g. alpha for BayesianAgent)
        valid_cols = ~all(isnan(y),1);
        y = y(:,valid_cols);
        this_model_names = model_names(valid_cols);
        facecolors = all_facecolors(valid_cols, :);

        mean_all = nanmean(y);
        SEM_all = nanstd(y)./sqrt(sum(~isnan(y)));

        title_name = sprintf('%s across models (%s condition, sigma fixed)', param_titles{p}, condition_label);

        figure;
        bar_plots_pval(y, mean_all, SEM_all, n, 1, size(y,2), this_model_names, 1, ...
            {param_names{p}}, title_name, '', param_names{p}, 0, 1, 20, 1, 12, 1, ...
            'Arial', 1, facecolors);
    end

    %% AIC/BIC/delta-BIC summary
    AICBIC_table = importdata(AICBIC_file);

    AIC = [AICBIC_table.AIC_basicRL, AICBIC_table.AIC_RLsigma, AICBIC_table.AIC_BayesianAgent];
    BIC = [AICBIC_table.BIC_basicRL, AICBIC_table.BIC_RLsigma, AICBIC_table.BIC_BayesianAgent];
    delta_BIC = [AICBIC_table.delta_BIC_basicRL, AICBIC_table.delta_BIC_RLsigma, AICBIC_table.delta_BIC_BayesianAgent];

    sum_AIC = nansum(AIC, 1);
    sum_BIC = nansum(BIC, 1);

    figure;
    h_aic = bar(1:3, sum_AIC, 'FaceColor', 'flat');
    set(gca, 'XTickLabel', model_names);
    title(sprintf('Sum of AIC across all subjects (%s condition, sigma fixed)', condition_label));
    ylabel('Sum AIC');
    xlabel('Model');
    box off;
    set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1);
    for i = 1:3
        h_aic.CData(i,:) = all_facecolors(i,:);
    end
    h_aic.FaceAlpha = 0.5;

    figure;
    h_bic = bar(1:3, sum_BIC, 'FaceColor', 'flat');
    set(gca, 'XTickLabel', model_names);
    title(sprintf('Sum of BIC across all subjects (%s condition, sigma fixed)', condition_label));
    ylabel('Sum BIC');
    xlabel('Model');
    box off;
    set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1);
    for i = 1:3
        h_bic.CData(i,:) = all_facecolors(i,:);
    end
    h_bic.FaceAlpha = 0.5;

    mean_delta_BIC = nanmean(delta_BIC, 1);
    SEM_delta_BIC = nanstd(delta_BIC, 1) ./ sqrt(sum(~isnan(delta_BIC), 1));
    n_subjects = size(delta_BIC, 1);

    figure;
    bar_plots_pval(delta_BIC, mean_delta_BIC, SEM_delta_BIC, n_subjects, 1, 3, model_names, 1, ...
        {'Delta BIC'}, sprintf('Mean Delta BIC across models (%s condition, sigma fixed)', condition_label), ...
        '', 'Delta BIC', 0, 1, 20, 1, 12, 1, 'Arial', 1, all_facecolors);
end
