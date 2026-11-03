%% ========================================================================
%  Script: mu_congruence learning curves for one best-fit-model group
%  ------------------------------------------------------------------------
%  Plots the group-average mu_congruence curve (slider response, averaged
%  first across each subject's own blocks, then across subjects) for the
%  Both condition and the Perceptual condition, restricted to subjects in
%  a single best-fitting-model group.
%
%  target_group below selects which group to plot:
%    'Ambiguous'     - subjects whose nominal best-fitting model (from
%                      fitSlider_ALLmodels.compute_model_comparison) does
%                      not beat the runner-up by the Kass & Raftery (1995)
%                      delta-BIC >= 2 "decisive" cutoff -- same definition
%                      plot_model_comparison's own "Ambiguous" bar uses,
%                      and the same relabeling individualDifferences_
%                      paramCorrelations.m applies before its mean+SEM+
%                      dots-by-group plot. (default)
%    'basicRL'       - subjects decisively best fit by the Basic RL model
%    'RLsigma'       - subjects decisively best fit by the RL + Est.
%                      Sensitivity model
%    'BayesianAgent' - subjects decisively best fit by the Bayesian Agent
%                      model
%  Group membership is recomputed separately for each condition (Both /
%  Perceptual), since a subject can be decisive-for-one-model in one
%  condition and ambiguous in the other.
%
%  Design notes:
%  - mu_congruence is indexed by its trials column (trial position within
%    a block, 1-25, resetting at the start of every block -- confirmed
%    against Data/descriptive data/main study/study2.txt) rather than by
%    row position, so a subject/block with a missing trial still lines up
%    correctly against every other subject/block's trial-25 x-axis instead
%    of silently shifting.
%  - Averaging is nanmean/nanstd throughout (matching Raw data
%    preprocessing/preprocess_mainstudy.m's mix_curve/perc_curve
%    convention), so occasional missing trials don't drop a whole
%    subject's curve.
%  - fix_sigma mirrors individualDifferences_paramCorrelations.m's toggle:
%    which bestModel_*_multiSP.mat files define the best-fit-model groups
%    (fixed-sigma fits from fitReducedModelSpace_fixedSigma.m, or
%    free-sigma fits from fitReducedModelSpace.m).
%  ========================================================================
clc; clearvars;

% PATH STUFF -- anchor to the repo root so relative loads below work
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

target_group = 'RLsigma'; % 'Ambiguous' / 'basicRL' / 'RLsigma' / 'BayesianAgent'

fix_sigma = true; % true = fixedSigma bestModel files, false = free-sigma bestModel files

%% =================== LOAD FITTING DATA AND BEST-MODEL GROUPS ============
[dataBoth, dataPerceptual, uniqueID, ~] = fitSlider_ALLmodels.load_fitting_data();

conditions = {'Both', 'Perceptual'};
condition_data = {dataBoth, dataPerceptual};

if fix_sigma
    bestModel_files = {'bestModel_fixedSigma_Both_multiSP.mat', 'bestModel_fixedSigma_Perceptual_multiSP.mat'};
else
    bestModel_files = {'bestModel_Both_multiSP.mat', 'bestModel_Perceptual_multiSP.mat'};
end

[~,~,~,~,~,~,~,mix_color,perc_color,~,~,~,~,~,~,~,~,~,~,~] = colors_rgb();
condition_colors = {mix_color, perc_color};

%% =================== BUILD PER-CONDITION GROUP CURVES ====================
max_trial = max([dataBoth.trials; dataPerceptual.trials]);
mean_curves = NaN(length(conditions), max_trial);
sem_curves = NaN(length(conditions), max_trial);
legend_names = cell(1, length(conditions));

for c = 1:length(conditions)
    condModel_table = importdata(fullfile(save_dir, bestModel_files{c}));
    assert(isequal(condModel_table.SubjectID, uniqueID), ...
        sprintf(['Subject ID mismatch between %s and uniqueID -- do not ' ...
        'proceed until this is resolved.'], bestModel_files{c}));

    % Relabel non-decisive subjects 'Ambiguous' rather than crediting
    % their narrow nominal winner -- see header notes above.
    best_model_grouped = condModel_table.best_model;
    best_model_grouped(~condModel_table.decisive) = {'Ambiguous'};

    target_subjects = uniqueID(strcmp(best_model_grouped, target_group));
    fprintf('[%s condition] %d/%d subjects in group "%s"\n', ...
        conditions{c}, length(target_subjects), length(uniqueID), target_group);

    data_c = condition_data{c};
    subj_curves = NaN(length(target_subjects), max_trial);
    for s = 1:length(target_subjects)
        dataSubj = data_c(data_c.ID == target_subjects(s), :);
        blockIDs = unique(dataSubj.blocks);
        block_curves = NaN(length(blockIDs), max_trial);
        for b = 1:length(blockIDs)
            dataBlock = dataSubj(dataSubj.blocks == blockIDs(b), :);
            block_curves(b, dataBlock.trials) = dataBlock.mu_congruence;
        end
        subj_curves(s, :) = nanmean(block_curves, 1);
    end

    mean_curves(c, :) = nanmean(subj_curves, 1);
    sem_curves(c, :) = nanstd(subj_curves, 0, 1) ./ sqrt(sum(~isnan(subj_curves), 1));
    legend_names{c} = sprintf('%s condition (n = %d)', conditions{c}, length(target_subjects));
end

%% =================== PLOT ================================================
figure('Position', [100 100 700 500]);
colors_name = [condition_colors{1}; condition_colors{2}];
lg_curves(1:max_trial, mean_curves, sem_curves, colors_name, legend_names, ...
    sprintf('mu_{congruence} learning curve -- %s group', target_group), ...
    'Trial within block', 'mu_{congruence}', 12, 1.5, 'Arial');
