%% ========================================================================
%  Script: Fit RL Models with Sigma Fixed from the Perceptual-Choice Fit (+AIC/BIC)
%  ------------------------------------------------------------------------
%  Same pipeline as fitReducedModelSpace.m, except sigma (perceptual
%  sensitivity) is not fit jointly with each model's other parameters.
%  Instead, each subject's sigma comes from their perceptual-choice fit
%  (fitPerceptualChoice.m, saved to sigma_perceptualChoice.mat) and is held
%  fixed while only the remaining parameters (alpha/kappa) are estimated
%  from the slider data. This reduces each model's free-parameter count by
%  one relative to fitReducedModelSpace.m, which is reflected in the
%  AIC/BIC computation below.
%  ========================================================================
clc; clearvars;
rng(123); % for reproducability
%% =================== LOAD AND PREPARE DATA =============================
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

% Load each subject's fixed sigma from the perceptual-choice fit. uniqueID
% here and in fitPerceptualChoice.m are both derived as unique(data.ID)
% from the same raw preprocessed_dataFitting.mat, so subject order matches.
perceptualSigma = importdata("sigma_perceptualChoice.mat");
fixedSigma = perceptualSigma.sigma;

%% =================== BASIC RL MODEL ====================================
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_basicRL = NaN(numSubjs, 1);
init_params = [0.1, 5]; % [alpha, kappa]; sigma is fixed, not fit
lb = [0, 1];
ub = [1, 100];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    sigma_n = fixedSigma(n);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated( ...
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    alphaParameter(n) = params(1);
    kappaParameter(n) = params(2);
    sigmaParameter(n) = sigma_n;
    nll_basicRL(n) = nll;

    fprintf('Subject number: %d\n', n);

end
params_basicRL.alpha = alphaParameter;
params_basicRL.kappa = kappaParameter;
params_basicRL.sigma = sigmaParameter;
safe_saveall('params_basicRL_fixedSigma_Both.mat', params_basicRL);
safe_saveall('nll_basicRL_fixedSigma_Both.mat', nll_basicRL);
%% =================== BAYESIAN AGENT MODEL ==============================
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_bayesianAgent = NaN(numSubjs, 1);
init_params = 5; % [kappa]; sigma is fixed, not fit
lb = 1;
ub = 100;
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.rewards, subj.contrast);
    sigma_n = fixedSigma(n);
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent([params(1), sigma_n], subj.mu_hat, subj.dataTable, ...
        length(unique(subj.blocks)), 25, unique(subj.blocks), rewards);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    kappaParameter(n) = params(1);
    sigmaParameter(n) = sigma_n;
    nll_bayesianAgent(n) = nll;

    fprintf('Subject number: %d\n', n);
end
params_bayesianAgent.sigma = sigmaParameter;
params_bayesianAgent.kappa = kappaParameter;
safe_saveall('params_bayesianAgent_fixedSigma_Both.mat', params_bayesianAgent);
safe_saveall('nll_bayesianAgent_fixedSigma_Both.mat', nll_bayesianAgent);

%% =================== RL + EST SENSITIVITY MODEL ========================
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_RLsigma = NaN(numSubjs, 1);
init_params = [0.1, 5]; % [alpha, kappa]; sigma is fixed, not fit
lb = [0, 1];
ub = [1, 100];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    sigma_n = fixedSigma(n);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI( ...
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    alphaParameter(n) = params(1);
    kappaParameter(n) = params(2);
    sigmaParameter(n) = sigma_n;
    nll_RLsigma(n) = nll;

    fprintf('Subject number: %d\n', n);
end
params_RLsigma.alpha = alphaParameter;
params_RLsigma.sigma = sigmaParameter;
params_RLsigma.kappa = kappaParameter;
safe_saveall('params_RLsigma_fixedSigma_Both.mat', params_RLsigma);
safe_saveall('nll_RLsigma_fixedSigma_Both.mat', nll_RLsigma);

%% =================== COMPUTE AND SAVE AIC/BIC ==========================
nll_basicRL = importdata("nll_basicRL_fixedSigma_Both.mat");
nll_RLsigma = importdata("nll_RLsigma_fixedSigma_Both.mat");
nll_bayesianAgent = importdata("nll_bayesianAgent_fixedSigma_Both.mat");

disp('Computing AIC and BIC for all models (sigma fixed)...');
% Number of trials per subject (assuming all subjects have same number)
num_trials = repelem(200,numSubjs);
% Model parameter counts -- one fewer than fitReducedModelSpace.m, since
% sigma is fixed rather than estimated. [basicRL, RLsigma, BayesianAgent]
num_params = [2, 2, 1];
AIC = NaN(numSubjs, 3);
BIC = NaN(numSubjs, 3);
for n = 1:numSubjs
    nlls = [nll_basicRL(n), nll_RLsigma(n), nll_bayesianAgent(n)];
    [AIC(n,:), BIC(n,:)] = fitSlider_ALLmodels.compute_aic_bic(nlls, num_params, num_trials(n));
end
model_names = {'basicRL','RLsigma','BayesianAgent'};
AICBIC_table = array2table([AIC, BIC], 'VariableNames', ...
    {'AIC_basicRL','AIC_RLsigma','AIC_BayesianAgent', ...
     'BIC_basicRL','BIC_RLsigma','BIC_BayesianAgent'});
AICBIC_table.SubjectID = uniqueID;
AICBIC_table = movevars(AICBIC_table, 'SubjectID', 'Before', 1);
% Compute delta BIC for each subject and model
delta_BIC = BIC - min(BIC,[],2);
AICBIC_table.delta_BIC_basicRL = delta_BIC(:,1);
AICBIC_table.delta_BIC_RLsigma = delta_BIC(:,2);
AICBIC_table.delta_BIC_BayesianAgent = delta_BIC(:,3);
% Save updated table
safe_saveall('AICBIC_fixedSigma_Both_reducedMS.mat', AICBIC_table);
disp('All model parameters and AIC/BIC estimated and saved (sigma fixed).');

%% Plot proportion of subjects best described by each model
% Extract delta BIC values for each model
delta_BIC_data = [AICBIC_table.delta_BIC_basicRL, ...
                  AICBIC_table.delta_BIC_RLsigma, ...
                  AICBIC_table.delta_BIC_BayesianAgent, ...
                  ];

% Find best model for each subject (deltaBIC = 0, or closest to 0 due to floating point)
[~, best_model_idx] = min(abs(delta_BIC_data), [], 2);

% Count proportion of subjects best described by each model
model_counts = histcounts(best_model_idx, 1:4);
model_proportions = model_counts / length(best_model_idx);

% Model names and colors
model_names = {'basicRL', 'RLsigma', 'BayesianAgent'};
colors = lines(3);

% Create the bar plot
figure;
h = bar(1:3, model_proportions, 'FaceColor', 'flat');

% Set individual bar colors and transparency
for i = 1:3
    h.CData(i,:) = colors(i,:);
end
h.FaceAlpha = 0.4;

% Customize the plot
set(gca, 'XTickLabel', model_names);
xlabel('Model');
ylabel('Proportion of Subjects');
title('Proportion of Subjects Best Described by Each Model (Both condition, Sigma Fixed)');
ylim([0, 1]);

% Add value labels on top of bars
for i = 1:3
    text(i, model_proportions(i) + 0.02, sprintf('%.2f', model_proportions(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Add a horizontal grid for easier reading
grid on;
set(gca, 'GridAlpha', 0.3);

% Display results in command window
fprintf('\nModel Selection Results (Sigma Fixed):\n');
fprintf('------------------------\n');
for i = 1:3
    fprintf('%s: %.1f%% of subjects (%d/%d)\n', ...
            model_names{i}, model_proportions(i)*100, model_counts(i), length(best_model_idx));
end

%%

% Prepare log model evidence matrix (LME = -NLL)
% Each row = subject, each column = model
lme = -0.5 * [BIC];

% Run Bayesian Model Selection
fprintf('Running Bayesian Model Selection (sigma fixed)...\n');
[alpha, exp_r, xp] = spm_BMS(lme, 1e6, 1, 0, 1, []);

% Display results
fprintf('\n=== BAYESIAN MODEL SELECTION RESULTS (SIGMA FIXED) ===\n\n');

% Model probabilities (expected posterior)
fprintf('Model Probabilities (exp_r):\n');
for i = 1:length(model_names)
    fprintf('  %s: %.4f\n', model_names{i}, exp_r(i));
end

% Exceedance probabilities
fprintf('\nExceedance Probabilities (xp):\n');
for i = 1:length(model_names)
    fprintf('  %s: %.4f\n', model_names{i}, xp(i));
end

% Determine winning model
[~, winning_model_idx] = max(exp_r);
fprintf('\nWinning Model: %s (exp_r = %.4f, xp = %.4f)\n', ...
    model_names{winning_model_idx}, exp_r(winning_model_idx), xp(winning_model_idx));

% Create summary table
BMS_results = table(model_names', exp_r', xp', ...
    'VariableNames', {'Model', 'ModelProbability', 'ExceedanceProbability'});
disp(BMS_results);

%% Bayesian Model Selection Results Visualization
% Assumes you have: exp_r, xp, model_names from spm_BMS output

% Model names for plotting (shorter labels)
model_labels = {'Basic RL', 'RL Sigma', 'Bayesian'};
colors = lines(3); % Custom colors

%% Main Figure with Two Subplots
fig = figure('Position', [100, 100, 1000, 500]);

% Subplot 1: Model Probabilities
subplot(1, 2, 1);
b1 = bar(exp_r, 'FaceColor', 'flat');
b1.CData = colors;
b1.FaceAlpha = 0.5;
set(gca, 'XTickLabel', model_labels, 'XTickLabelRotation', 45);
ylabel('Model Probability (exp_r)', 'FontSize', 12, 'FontWeight', 'bold');
title('Model Probabilities', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0, max(exp_r) * 1.1]);
grid on;
grid minor;

% Add value labels on bars
for i = 1:length(exp_r)
    text(i, exp_r(i) + 0.01, sprintf('%.3f', exp_r(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

% Subplot 2: Exceedance Probabilities
subplot(1, 2, 2);
b2 = bar(xp, 'FaceColor', 'flat');
b2.CData = colors;
b2.FaceAlpha = 0.5;
set(gca, 'XTickLabel', model_labels, 'XTickLabelRotation', 45);
ylabel('Exceedance Probability (xp)', 'FontSize', 12, 'FontWeight', 'bold');
title('Exceedance Probabilities', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0, 1.05]);
grid on;
grid minor;

% Add value labels on bars
for i = 1:length(xp)
    if xp(i) < 0.001
        text(i, xp(i) + 0.02, '< 0.001', ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    else
        text(i, xp(i) + 0.02, sprintf('%.3f', xp(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
end

% Add overall title
sgtitle('Bayesian Model Selection Results (Both condition, Sigma Fixed)', 'FontSize', 16, 'FontWeight', 'bold');

%% Detailed Figure: Combined Plot with Winning Model Highlighted
fig2 = figure('Position', [150, 150, 800, 600]);

% Find winning model
[~, winner_idx] = max(xp);

% Create bar plot
x_pos = 1:length(model_names);
b = bar(x_pos, [exp_r; xp]', 'grouped');

% Set colors
b(1).FaceColor = [0.3 0.5 0.8]; % Model probabilities
b(2).FaceColor = [0.8 0.4 0.3]; % Exceedance probabilities

% Highlight winning model
b(1).CData(winner_idx, :) = [0.1 0.7 0.1]; % Green for winner
b(2).CData(winner_idx, :) = [0.1 0.7 0.1]; % Green for winner

set(gca, 'XTickLabel', model_labels, 'XTickLabelRotation', 45);
ylabel('Probability', 'FontSize', 12, 'FontWeight', 'bold');
title('Model Comparison: Probabilities and Exceedance (Sigma Fixed)', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Model Probability (exp_r)', 'Exceedance Probability (xp)'}, ...
    'Location', 'northeast', 'FontSize', 11);
grid on;
ylim([0, 1.05]);

% Add text annotation for winner
text(winner_idx, max([exp_r(winner_idx), xp(winner_idx)]) + 0.1, ...
    sprintf('WINNER\n%s', model_labels{winner_idx}), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
    'Color', [0.1 0.7 0.1]);


%% Family-wise Model Comparison using VBA Toolbox

% 1. Prepare Log Model Evidence Matrix
% Convert from information criteria to log model evidence
BIC_matrix = [AICBIC_table.BIC_basicRL, ...
                  AICBIC_table.BIC_RLsigma, ...
                  AICBIC_table.BIC_BayesianAgent];
lme = -0.5 * BIC_matrix;  % Convert BIC to log model evidence

% Check dimensions
[n_subjects, n_models] = size(lme);
fprintf('Data: %d subjects, %d models\n', n_subjects, n_models);

% 2. Define Model Families
family_names = {'Basic_RL', 'LR_Modulation'};
% Define which models belong to which family
families{1} = [1];           % BasicRL
families{2} = [2, 3];    % LR modulation models

%% 3. Run Family-wise Bayesian Model Selection
fprintf('Running family-wise model comparison (sigma fixed)...\n');
% Set up VBA options structure
options_vba = struct();
options_vba.families = families;
options_vba.verbose = 1;
options_vba.DisplayWin = 0;

try
    % VBA_groupBMC with proper options structure
    [posterior, out] = VBA_groupBMC(lme', options_vba);
    fprintf('VBA_groupBMC completed successfully!\n');
    vba_success = true;
catch ME
    fprintf('VBA_groupBMC failed with error: %s\n', ME.message);
    fprintf('Using manual implementation instead...\n');
    [posterior, out] = manual_family_BMS(lme, families, family_names);
    vba_success = false;
end

%% 4. Extract and Display Results
if vba_success
    % VBA succeeded - extract results correctly
    family_prob = out.families.Ef;        % Family probabilities
    family_xp = out.families.ep;          % Family exceedance probabilities
    model_prob = out.Ef;                  % Individual model probabilities
    model_xp = out.ep;                    % Individual model exceedance probabilities

    fprintf('\n=== VBA FAMILY-WISE RESULTS (SIGMA FIXED) ===\n');
else
    % Manual implementation results
    family_prob = posterior.r;            % Family probabilities
    family_xp = out.families.ep;          % Family exceedance probabilities
    model_prob = posterior.r_model;       % Individual model probabilities
    model_xp = out.ep;                    % Individual model exceedance probabilities

    fprintf('\n=== MANUAL FAMILY-WISE RESULTS (SIGMA FIXED) ===\n');
end

% Display family results
for i = 1:length(families)
    fprintf('Family %d (%s):\n', i, family_names{i});
    fprintf('  Probability: %.3f\n', family_prob(i));
    fprintf('  Exceedance Probability: %.3f\n', family_xp(i));
    fprintf('  Models included: %s\n', mat2str(families{i}));
    fprintf('\n');
end

% Display model results
fprintf('=== MODEL-LEVEL RESULTS (SIGMA FIXED) ===\n');
model_names = {'BasicRL', 'RLSigma', 'RLSigma + CS', 'Agent', 'Agent + CS', 'BasicRL + CS'};
for i = 1:n_models
    fprintf('Model %d (%s): Prob=%.3f, XP=%.3f\n', i, model_names{i}, model_prob(i), model_xp(i));
end

%% 5. Enhanced Visualization with Copper Colormap
% Create copper colormap colors with alpha
copper_colors = copper(length(families) + n_models);
dark_gray = [0.3, 0.3, 0.3];  % Dark gray for text

% Create main figure
fig = figure('Position', [100, 100, 1400, 500], 'Color', 'white');

%% Plot 1: Family Probabilities
subplot(1, 3, 1);
h1 = bar(family_prob, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6, 'FaceAlpha', 0.3);

% Set copper colormap colors for families
h1.CData(1,:) = copper_colors(1, 1:3);  % First copper color
h1.CData(2,:) = copper_colors(2, 1:3);  % Second copper color

% Styling - remove grid and use normal font
ylim([0, 1]);
ylabel('Family Probability', 'FontSize', 12, 'FontWeight', 'bold');
title('Family Probabilities', 'FontSize', 14, 'FontWeight', 'normal', 'Color', dark_gray);
set(gca, 'XTickLabel', {'Basic RL', 'LR Modulation'}, 'FontSize', 11, ...
    'Box', 'off', 'LineWidth', 1.2);
grid off;  % Explicitly remove grid

% Add value labels on bars
for i = 1:length(family_prob)
    text(i, family_prob(i) + 0.03, sprintf('%.3f', family_prob(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
         'FontSize', 11, 'Color', dark_gray);
end

ax1 = gca;
ax1.Color = [0.98, 0.98, 0.98];

%% Plot 2: Family Exceedance Probabilities
subplot(1, 3, 2);
h2 = bar(family_xp, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6, 'FaceAlpha', 0.3);

% Set copper colormap colors for families (slightly darker)
h2.CData(1,:) = copper_colors(1, 1:3) * 0.8;
h2.CData(2,:) = copper_colors(2, 1:3) * 0.8;

% Styling - remove grid and use normal font
ylim([0, 1]);
ylabel('Exceedance Probability', 'FontSize', 12, 'FontWeight', 'bold');
title('Family Exceedance Probabilities', 'FontSize', 14, 'FontWeight', 'normal', 'Color', dark_gray);
set(gca, 'XTickLabel', {'Basic RL', 'LR Modulation'}, 'FontSize', 11, ...
    'Box', 'off', 'LineWidth', 1.2);
grid off;  % Explicitly remove grid

% Add value labels
for i = 1:length(family_xp)
    text(i, family_xp(i) + 0.03, sprintf('%.3f', family_xp(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
         'FontSize', 11, 'Color', dark_gray);
end

ax2 = gca;
ax2.Color = [0.98, 0.98, 0.98];

%% Plot 3: Individual Model Probabilities
subplot(1, 3, 3);

% Create color array for each model using copper colormap
model_copper_colors = copper(n_models);
bar_colors = model_copper_colors(:, 1:3);  % Remove alpha channel

% Create bar plot with copper colors and alpha
h3 = bar(1:n_models, model_prob, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.7, 'FaceAlpha', 0.3);
h3.CData = bar_colors;

% Styling - remove grid and use normal font
ylim([0, max(model_prob) * 1.15]);
xlabel('Model', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Model Probability', 'FontSize', 12, 'FontWeight', 'bold');
title('Individual Model Probabilities', 'FontSize', 14, 'FontWeight', 'normal', 'Color', dark_gray);

% Better x-axis labels
set(gca, 'XTick', 1:n_models, 'XTickLabel', model_names, ...
    'XTickLabelRotation', 45, 'FontSize', 10, ...
    'Box', 'off', 'LineWidth', 1.2);
grid off;  % Explicitly remove grid

% Add value labels on bars
for i = 1:n_models
    if model_prob(i) > 0.01  % Only label bars with meaningful height
        text(i, model_prob(i) + max(model_prob) * 0.02, sprintf('%.3f', model_prob(i)), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
             'FontSize', 9, 'Color', dark_gray);
    end
end

ax3 = gca;
ax3.Color = [0.98, 0.98, 0.98];

%% Overall figure improvements
% Adjust subplot spacing
set(fig, 'Units', 'normalized');
subplot(1,3,1); pos1 = get(gca, 'Position'); pos1(1) = 0.08; set(gca, 'Position', pos1);
subplot(1,3,2); pos2 = get(gca, 'Position'); pos2(1) = 0.38; set(gca, 'Position', pos2);
subplot(1,3,3); pos3 = get(gca, 'Position'); pos3(1) = 0.68; pos3(3) = 0.28; set(gca, 'Position', pos3);

%% 6. Save Results
results_summary = struct();
results_summary.family_names = family_names;
results_summary.families = families;
results_summary.family_probabilities = family_prob;
results_summary.family_exceedance_probabilities = family_xp;
results_summary.model_names = model_names;
results_summary.model_probabilities = model_prob;
results_summary.model_exceedance_probabilities = model_xp;

% Save to file
save('family_wise_BMS_results_fixedSigma_both.mat', 'results_summary', 'posterior', 'out');
fprintf('\nResults saved to: family_wise_BMS_results_fixedSigma_both.mat\n');

%% 7. Additional Analysis: Model Contributions within Families
fprintf('\n=== WITHIN-FAMILY MODEL CONTRIBUTIONS (SIGMA FIXED) ===\n');
for f = 1:length(families)
    fprintf('\nFamily %d (%s):\n', f, family_names{f});
    models_in_family = families{f};
    family_model_probs = model_prob(models_in_family);

    % Normalize to get relative contributions within family
    if sum(family_model_probs) > 0
        relative_contrib = family_model_probs / sum(family_model_probs);
        for m = 1:length(models_in_family)
            model_idx = models_in_family(m);
            fprintf('  %s (Model %d): %.3f (%.1f%% of family)\n', ...
                model_names{model_idx}, model_idx, family_model_probs(m), ...
                relative_contrib(m) * 100);
        end
    end
end

% ---------- Stacked bars per-family (no hard-coded probs) ----------
num_families = length(families);

% Choose base colors for each family (customize if you want)
if num_families == 1
    family_colors = [0.2 0.6 0.9];
elseif num_families == 2
    family_colors = [
        0.0000 0.4470 0.7410;   % blue-ish
        0.9 0.3250 0.0980    % orange-ish
    ];
else
    family_colors = parula(num_families); % fallback for >2 families
end

figure;
hold on;
handles = gobjects(0);   % for legend handles
labels  = {};            % for legend labels

% Plot each family as a single stacked bar (so colors can be set per-family)
for f = 1:num_families
    models_in_family = families{f};
    family_model_probs = model_prob(models_in_family);
    if sum(family_model_probs) > 0
        rel = family_model_probs / sum(family_model_probs);  % relative contributions
    else
        rel = zeros(size(family_model_probs));
    end

    % Create shades for models within the family (darker -> lighter)
    n = numel(rel);
    % shades from 0.55 (darker) to 1.0 (base color)
    shades = linspace(0.35, 1.00, n)';
    base = family_colors(min(f,size(family_colors,1)), :);
    model_colors = bsxfun(@times, shades, base); % n x 3 matrix

    % Draw a single stacked bar at x = f
    b = bar(f, rel, 0.6, 'stacked');   % b is an array of Bar objects (one per segment)
    for i = 1:numel(b)
        % set color for this segment (this family only)
        b(i).FaceColor = model_colors(i,:);
        b(i).EdgeColor = 'none';

        % store handle & label for legend (label with model name + family)
        model_idx = models_in_family(i);
        handles(end+1) = b(i); %#ok<SAGROW>
        labels{end+1} = sprintf('%s (Family: %s)', model_names{model_idx}, family_names{f});
    end

    % optional: add text labels inside segments (percent)
    cum = 0;
    for i = 1:n
        h = rel(i);
        if h > 0
            yc = cum + h/2;
            % choose text color for contrast
            rgb = model_colors(i,:);
            lum = 0.299*rgb(1) + 0.587*rgb(2) + 0.114*rgb(3);
            txtc = 'w';
            if lum > 0.7, txtc = 'k'; end
            text(f, yc, sprintf('%.1f%%', h*100), ...
                 'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
                 'FontSize',9, 'Color', txtc);
        end
        cum = cum + h;
    end
end

% Format axes & legend
xlim([0.5, num_families+0.5]);
set(gca, 'XTick', 1:num_families, 'XTickLabel', family_names, 'FontSize', 12);
ylabel('Relative Model Contribution (within family)');
title('Within-Family Model Contributions (stacked) (Sigma Fixed)');
ylim([0 1]);
grid on; box on;

% Legend (one entry per model segment)
legend(handles, labels, 'Location', 'eastoutside');

hold off;

%% 8. Statistical Summary
fprintf('\n=== STATISTICAL SUMMARY (SIGMA FIXED) ===\n');
fprintf('Winning Family: %s (Prob = %.3f, XP = %.3f)\n', ...
    family_names{find(family_prob == max(family_prob))}, ...
    max(family_prob), max(family_xp));
fprintf('Winning Model: %s (Prob = %.3f, XP = %.3f)\n', ...
    model_names{find(model_prob == max(model_prob))}, ...
    max(model_prob), max(model_xp));

% Evidence strength interpretation
if max(family_xp) > 0.95
    evidence_strength = 'Very Strong';
elseif max(family_xp) > 0.90
    evidence_strength = 'Strong';
elseif max(family_xp) > 0.75
    evidence_strength = 'Moderate';
else
    evidence_strength = 'Weak';
end
fprintf('Evidence Strength: %s (XP = %.3f)\n', evidence_strength, max(family_xp));

%% ========================================================================
%  PERCEPTUAL CONDITION
%  ------------------------------------------------------------------------
%  Same fitting + model-selection pipeline as above, run on the
%  perceptual-condition data (dataPerceptual) instead of the both-condition
%  data (dataBoth). Sigma is still fixed per subject from the same
%  perceptual-choice fit (fixedSigma).
%  ========================================================================
%% =================== BASIC RL MODEL (PERCEPTUAL) ========================
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_basicRL = NaN(numSubjs, 1);
init_params = [0.1, 5]; % [alpha, kappa]; sigma is fixed, not fit
lb = [0, 1];
ub = [1, 100];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    sigma_n = fixedSigma(n);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated( ...
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    alphaParameter(n) = params(1);
    kappaParameter(n) = params(2);
    sigmaParameter(n) = sigma_n;
    nll_basicRL(n) = nll;

    fprintf('Subject number: %d\n', n);

end
params_basicRL.alpha = alphaParameter;
params_basicRL.kappa = kappaParameter;
params_basicRL.sigma = sigmaParameter;
safe_saveall('params_basicRL_fixedSigma_Perceptual.mat', params_basicRL);
safe_saveall('nll_basicRL_fixedSigma_Perceptual.mat', nll_basicRL);
%% =================== BAYESIAN AGENT MODEL (PERCEPTUAL) ==================
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_bayesianAgent = NaN(numSubjs, 1);
init_params = 5; % [kappa]; sigma is fixed, not fit
lb = 1;
ub = 100;
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.rewards, subj.contrast);
    sigma_n = fixedSigma(n);
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent([params(1), sigma_n], subj.mu_hat, subj.dataTable, ...
        length(unique(subj.blocks)), 25, unique(subj.blocks), rewards);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    kappaParameter(n) = params(1);
    sigmaParameter(n) = sigma_n;
    nll_bayesianAgent(n) = nll;

    fprintf('Subject number: %d\n', n);
end
params_bayesianAgent.sigma = sigmaParameter;
params_bayesianAgent.kappa = kappaParameter;
safe_saveall('params_bayesianAgent_fixedSigma_Perceptual.mat', params_bayesianAgent);
safe_saveall('nll_bayesianAgent_fixedSigma_Perceptual.mat', nll_bayesianAgent);

%% =================== RL + EST SENSITIVITY MODEL (PERCEPTUAL) ============
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_RLsigma = NaN(numSubjs, 1);
init_params = [0.1, 5]; % [alpha, kappa]; sigma is fixed, not fit
lb = [0, 1];
ub = [1, 100];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    sigma_n = fixedSigma(n);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI( ...
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    alphaParameter(n) = params(1);
    kappaParameter(n) = params(2);
    sigmaParameter(n) = sigma_n;
    nll_RLsigma(n) = nll;

    fprintf('Subject number: %d\n', n);
end
params_RLsigma.alpha = alphaParameter;
params_RLsigma.sigma = sigmaParameter;
params_RLsigma.kappa = kappaParameter;
safe_saveall('params_RLsigma_fixedSigma_Perceptual.mat', params_RLsigma);
safe_saveall('nll_RLsigma_fixedSigma_Perceptual.mat', nll_RLsigma);

%% =================== COMPUTE AND SAVE AIC/BIC (PERCEPTUAL) ==============
nll_basicRL = importdata("nll_basicRL_fixedSigma_Perceptual.mat");
nll_RLsigma = importdata("nll_RLsigma_fixedSigma_Perceptual.mat");
nll_bayesianAgent = importdata("nll_bayesianAgent_fixedSigma_Perceptual.mat");

disp('Computing AIC and BIC for all models (perceptual condition, sigma fixed)...');
% Number of trials per subject (assuming all subjects have same number)
num_trials = repelem(200,numSubjs);
% Model parameter counts -- one fewer than fitReducedModelSpace.m, since
% sigma is fixed rather than estimated. [basicRL, RLsigma, BayesianAgent]
num_params = [2, 2, 1];
AIC = NaN(numSubjs, 3);
BIC = NaN(numSubjs, 3);
for n = 1:numSubjs
    nlls = [nll_basicRL(n), nll_RLsigma(n), nll_bayesianAgent(n)];
    [AIC(n,:), BIC(n,:)] = fitSlider_ALLmodels.compute_aic_bic(nlls, num_params, num_trials(n));
end
model_names = {'basicRL','RLsigma','BayesianAgent'};
AICBIC_table = array2table([AIC, BIC], 'VariableNames', ...
    {'AIC_basicRL','AIC_RLsigma','AIC_BayesianAgent', ...
     'BIC_basicRL','BIC_RLsigma','BIC_BayesianAgent'});
AICBIC_table.SubjectID = uniqueID;
AICBIC_table = movevars(AICBIC_table, 'SubjectID', 'Before', 1);
% Compute delta BIC for each subject and model
delta_BIC = BIC - min(BIC,[],2);
AICBIC_table.delta_BIC_basicRL = delta_BIC(:,1);
AICBIC_table.delta_BIC_RLsigma = delta_BIC(:,2);
AICBIC_table.delta_BIC_BayesianAgent = delta_BIC(:,3);
% Save updated table
safe_saveall('AICBIC_fixedSigma_Perceptual_reducedMS.mat', AICBIC_table);
disp('All model parameters and AIC/BIC estimated and saved (perceptual condition, sigma fixed).');

%% Plot proportion of subjects best described by each model (Perceptual)
delta_BIC_data = [AICBIC_table.delta_BIC_basicRL, ...
                  AICBIC_table.delta_BIC_RLsigma, ...
                  AICBIC_table.delta_BIC_BayesianAgent, ...
                  ];

[~, best_model_idx] = min(abs(delta_BIC_data), [], 2);

model_counts = histcounts(best_model_idx, 1:4);
model_proportions = model_counts / length(best_model_idx);

model_names = {'basicRL', 'RLsigma', 'BayesianAgent'};
colors = lines(3);

figure;
h = bar(1:3, model_proportions, 'FaceColor', 'flat');

for i = 1:3
    h.CData(i,:) = colors(i,:);
end
h.FaceAlpha = 0.4;

set(gca, 'XTickLabel', model_names);
xlabel('Model');
ylabel('Proportion of Subjects');
title('Proportion of Subjects Best Described by Each Model (Perceptual condition, Sigma Fixed)');
ylim([0, 1]);

for i = 1:3
    text(i, model_proportions(i) + 0.02, sprintf('%.2f', model_proportions(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

grid on;
set(gca, 'GridAlpha', 0.3);

fprintf('\nModel Selection Results (Perceptual condition, Sigma Fixed):\n');
fprintf('------------------------\n');
for i = 1:3
    fprintf('%s: %.1f%% of subjects (%d/%d)\n', ...
            model_names{i}, model_proportions(i)*100, model_counts(i), length(best_model_idx));
end

%%

lme = -0.5 * [BIC];

fprintf('Running Bayesian Model Selection (perceptual condition, sigma fixed)...\n');
[alpha, exp_r, xp] = spm_BMS(lme, 1e6, 1, 0, 1, []);

fprintf('\n=== BAYESIAN MODEL SELECTION RESULTS (PERCEPTUAL, SIGMA FIXED) ===\n\n');

fprintf('Model Probabilities (exp_r):\n');
for i = 1:length(model_names)
    fprintf('  %s: %.4f\n', model_names{i}, exp_r(i));
end

fprintf('\nExceedance Probabilities (xp):\n');
for i = 1:length(model_names)
    fprintf('  %s: %.4f\n', model_names{i}, xp(i));
end

[~, winning_model_idx] = max(exp_r);
fprintf('\nWinning Model: %s (exp_r = %.4f, xp = %.4f)\n', ...
    model_names{winning_model_idx}, exp_r(winning_model_idx), xp(winning_model_idx));

BMS_results = table(model_names', exp_r', xp', ...
    'VariableNames', {'Model', 'ModelProbability', 'ExceedanceProbability'});
disp(BMS_results);

%% Bayesian Model Selection Results Visualization (Perceptual)
model_labels = {'Basic RL', 'RL Sigma', 'Bayesian'};
colors = lines(3); % Custom colors

%% Main Figure with Two Subplots (Perceptual)
fig = figure('Position', [100, 100, 1000, 500]);

subplot(1, 2, 1);
b1 = bar(exp_r, 'FaceColor', 'flat');
b1.CData = colors;
b1.FaceAlpha = 0.5;
set(gca, 'XTickLabel', model_labels, 'XTickLabelRotation', 45);
ylabel('Model Probability (exp_r)', 'FontSize', 12, 'FontWeight', 'bold');
title('Model Probabilities', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0, max(exp_r) * 1.1]);
grid on;
grid minor;

for i = 1:length(exp_r)
    text(i, exp_r(i) + 0.01, sprintf('%.3f', exp_r(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

subplot(1, 2, 2);
b2 = bar(xp, 'FaceColor', 'flat');
b2.CData = colors;
b2.FaceAlpha = 0.5;
set(gca, 'XTickLabel', model_labels, 'XTickLabelRotation', 45);
ylabel('Exceedance Probability (xp)', 'FontSize', 12, 'FontWeight', 'bold');
title('Exceedance Probabilities', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0, 1.05]);
grid on;
grid minor;

for i = 1:length(xp)
    if xp(i) < 0.001
        text(i, xp(i) + 0.02, '< 0.001', ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    else
        text(i, xp(i) + 0.02, sprintf('%.3f', xp(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
end

sgtitle('Bayesian Model Selection Results (Perceptual condition, Sigma Fixed)', 'FontSize', 16, 'FontWeight', 'bold');

%% Detailed Figure: Combined Plot with Winning Model Highlighted (Perceptual)
fig2 = figure('Position', [150, 150, 800, 600]);

[~, winner_idx] = max(xp);

x_pos = 1:length(model_names);
b = bar(x_pos, [exp_r; xp]', 'grouped');

b(1).FaceColor = [0.3 0.5 0.8]; % Model probabilities
b(2).FaceColor = [0.8 0.4 0.3]; % Exceedance probabilities

b(1).CData(winner_idx, :) = [0.1 0.7 0.1]; % Green for winner
b(2).CData(winner_idx, :) = [0.1 0.7 0.1]; % Green for winner

set(gca, 'XTickLabel', model_labels, 'XTickLabelRotation', 45);
ylabel('Probability', 'FontSize', 12, 'FontWeight', 'bold');
title('Model Comparison: Probabilities and Exceedance (Perceptual condition, Sigma Fixed)', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Model Probability (exp_r)', 'Exceedance Probability (xp)'}, ...
    'Location', 'northeast', 'FontSize', 11);
grid on;
ylim([0, 1.05]);

text(winner_idx, max([exp_r(winner_idx), xp(winner_idx)]) + 0.1, ...
    sprintf('WINNER\n%s', model_labels{winner_idx}), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
    'Color', [0.1 0.7 0.1]);


%% Family-wise Model Comparison using VBA Toolbox (Perceptual)

BIC_matrix = [AICBIC_table.BIC_basicRL, ...
                  AICBIC_table.BIC_RLsigma, ...
                  AICBIC_table.BIC_BayesianAgent];
lme = -0.5 * BIC_matrix;  % Convert BIC to log model evidence

[n_subjects, n_models] = size(lme);
fprintf('Data: %d subjects, %d models\n', n_subjects, n_models);

family_names = {'Basic_RL', 'LR_Modulation'};
families{1} = [1];           % BasicRL
families{2} = [2, 3];    % LR modulation models

fprintf('Running family-wise model comparison (perceptual condition, sigma fixed)...\n');
options_vba = struct();
options_vba.families = families;
options_vba.verbose = 1;
options_vba.DisplayWin = 0;

try
    [posterior, out] = VBA_groupBMC(lme', options_vba);
    fprintf('VBA_groupBMC completed successfully!\n');
    vba_success = true;
catch ME
    fprintf('VBA_groupBMC failed with error: %s\n', ME.message);
    fprintf('Using manual implementation instead...\n');
    [posterior, out] = manual_family_BMS(lme, families, family_names);
    vba_success = false;
end

if vba_success
    family_prob = out.families.Ef;        % Family probabilities
    family_xp = out.families.ep;          % Family exceedance probabilities
    model_prob = out.Ef;                  % Individual model probabilities
    model_xp = out.ep;                    % Individual model exceedance probabilities

    fprintf('\n=== VBA FAMILY-WISE RESULTS (PERCEPTUAL, SIGMA FIXED) ===\n');
else
    family_prob = posterior.r;            % Family probabilities
    family_xp = out.families.ep;          % Family exceedance probabilities
    model_prob = posterior.r_model;       % Individual model probabilities
    model_xp = out.ep;                    % Individual model exceedance probabilities

    fprintf('\n=== MANUAL FAMILY-WISE RESULTS (PERCEPTUAL, SIGMA FIXED) ===\n');
end

for i = 1:length(families)
    fprintf('Family %d (%s):\n', i, family_names{i});
    fprintf('  Probability: %.3f\n', family_prob(i));
    fprintf('  Exceedance Probability: %.3f\n', family_xp(i));
    fprintf('  Models included: %s\n', mat2str(families{i}));
    fprintf('\n');
end

fprintf('=== MODEL-LEVEL RESULTS (PERCEPTUAL, SIGMA FIXED) ===\n');
model_names = {'BasicRL', 'RLSigma', 'RLSigma + CS', 'Agent', 'Agent + CS', 'BasicRL + CS'};
for i = 1:n_models
    fprintf('Model %d (%s): Prob=%.3f, XP=%.3f\n', i, model_names{i}, model_prob(i), model_xp(i));
end

%% Enhanced Visualization with Copper Colormap (Perceptual)
copper_colors = copper(length(families) + n_models);
dark_gray = [0.3, 0.3, 0.3];  % Dark gray for text

fig = figure('Position', [100, 100, 1400, 500], 'Color', 'white');

subplot(1, 3, 1);
h1 = bar(family_prob, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6, 'FaceAlpha', 0.3);

h1.CData(1,:) = copper_colors(1, 1:3);
h1.CData(2,:) = copper_colors(2, 1:3);

ylim([0, 1]);
ylabel('Family Probability', 'FontSize', 12, 'FontWeight', 'bold');
title('Family Probabilities', 'FontSize', 14, 'FontWeight', 'normal', 'Color', dark_gray);
set(gca, 'XTickLabel', {'Basic RL', 'LR Modulation'}, 'FontSize', 11, ...
    'Box', 'off', 'LineWidth', 1.2);
grid off;

for i = 1:length(family_prob)
    text(i, family_prob(i) + 0.03, sprintf('%.3f', family_prob(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
         'FontSize', 11, 'Color', dark_gray);
end

ax1 = gca;
ax1.Color = [0.98, 0.98, 0.98];

subplot(1, 3, 2);
h2 = bar(family_xp, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6, 'FaceAlpha', 0.3);

h2.CData(1,:) = copper_colors(1, 1:3) * 0.8;
h2.CData(2,:) = copper_colors(2, 1:3) * 0.8;

ylim([0, 1]);
ylabel('Exceedance Probability', 'FontSize', 12, 'FontWeight', 'bold');
title('Family Exceedance Probabilities', 'FontSize', 14, 'FontWeight', 'normal', 'Color', dark_gray);
set(gca, 'XTickLabel', {'Basic RL', 'LR Modulation'}, 'FontSize', 11, ...
    'Box', 'off', 'LineWidth', 1.2);
grid off;

for i = 1:length(family_xp)
    text(i, family_xp(i) + 0.03, sprintf('%.3f', family_xp(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
         'FontSize', 11, 'Color', dark_gray);
end

ax2 = gca;
ax2.Color = [0.98, 0.98, 0.98];

subplot(1, 3, 3);

model_copper_colors = copper(n_models);
bar_colors = model_copper_colors(:, 1:3);

h3 = bar(1:n_models, model_prob, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.7, 'FaceAlpha', 0.3);
h3.CData = bar_colors;

ylim([0, max(model_prob) * 1.15]);
xlabel('Model', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Model Probability', 'FontSize', 12, 'FontWeight', 'bold');
title('Individual Model Probabilities', 'FontSize', 14, 'FontWeight', 'normal', 'Color', dark_gray);

set(gca, 'XTick', 1:n_models, 'XTickLabel', model_names, ...
    'XTickLabelRotation', 45, 'FontSize', 10, ...
    'Box', 'off', 'LineWidth', 1.2);
grid off;

for i = 1:n_models
    if model_prob(i) > 0.01
        text(i, model_prob(i) + max(model_prob) * 0.02, sprintf('%.3f', model_prob(i)), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
             'FontSize', 9, 'Color', dark_gray);
    end
end

ax3 = gca;
ax3.Color = [0.98, 0.98, 0.98];

sgtitle('Family-wise Bayesian Model Selection Results (Perceptual condition, Sigma Fixed)', 'FontSize', 16, 'FontWeight', 'normal', 'Color', dark_gray);

set(fig, 'Units', 'normalized');
subplot(1,3,1); pos1 = get(gca, 'Position'); pos1(1) = 0.08; set(gca, 'Position', pos1);
subplot(1,3,2); pos2 = get(gca, 'Position'); pos2(1) = 0.38; set(gca, 'Position', pos2);
subplot(1,3,3); pos3 = get(gca, 'Position'); pos3(1) = 0.68; pos3(3) = 0.28; set(gca, 'Position', pos3);

%% Save Results (Perceptual)
results_summary = struct();
results_summary.family_names = family_names;
results_summary.families = families;
results_summary.family_probabilities = family_prob;
results_summary.family_exceedance_probabilities = family_xp;
results_summary.model_names = model_names;
results_summary.model_probabilities = model_prob;
results_summary.model_exceedance_probabilities = model_xp;

save('family_wise_BMS_results_fixedSigma_perceptual.mat', 'results_summary', 'posterior', 'out');
fprintf('\nResults saved to: family_wise_BMS_results_fixedSigma_perceptual.mat\n');

%% Additional Analysis: Model Contributions within Families (Perceptual)
fprintf('\n=== WITHIN-FAMILY MODEL CONTRIBUTIONS (PERCEPTUAL, SIGMA FIXED) ===\n');
for f = 1:length(families)
    fprintf('\nFamily %d (%s):\n', f, family_names{f});
    models_in_family = families{f};
    family_model_probs = model_prob(models_in_family);

    if sum(family_model_probs) > 0
        relative_contrib = family_model_probs / sum(family_model_probs);
        for m = 1:length(models_in_family)
            model_idx = models_in_family(m);
            fprintf('  %s (Model %d): %.3f (%.1f%% of family)\n', ...
                model_names{model_idx}, model_idx, family_model_probs(m), ...
                relative_contrib(m) * 100);
        end
    end
end

% ---------- Stacked bars per-family (no hard-coded probs) ----------
num_families = length(families);

if num_families == 1
    family_colors = [0.2 0.6 0.9];
elseif num_families == 2
    family_colors = [
        0.0000 0.4470 0.7410;   % blue-ish
        0.9 0.3250 0.0980    % orange-ish
    ];
else
    family_colors = parula(num_families);
end

figure;
hold on;
handles = gobjects(0);
labels  = {};

for f = 1:num_families
    models_in_family = families{f};
    family_model_probs = model_prob(models_in_family);
    if sum(family_model_probs) > 0
        rel = family_model_probs / sum(family_model_probs);
    else
        rel = zeros(size(family_model_probs));
    end

    n = numel(rel);
    shades = linspace(0.35, 1.00, n)';
    base = family_colors(min(f,size(family_colors,1)), :);
    model_colors = bsxfun(@times, shades, base);

    b = bar(f, rel, 0.6, 'stacked');
    for i = 1:numel(b)
        b(i).FaceColor = model_colors(i,:);
        b(i).EdgeColor = 'none';

        model_idx = models_in_family(i);
        handles(end+1) = b(i); %#ok<SAGROW>
        labels{end+1} = sprintf('%s (Family: %s)', model_names{model_idx}, family_names{f});
    end

    cum = 0;
    for i = 1:n
        h = rel(i);
        if h > 0
            yc = cum + h/2;
            rgb = model_colors(i,:);
            lum = 0.299*rgb(1) + 0.587*rgb(2) + 0.114*rgb(3);
            txtc = 'w';
            if lum > 0.7, txtc = 'k'; end
            text(f, yc, sprintf('%.1f%%', h*100), ...
                 'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
                 'FontSize',9, 'Color', txtc);
        end
        cum = cum + h;
    end
end

xlim([0.5, num_families+0.5]);
set(gca, 'XTick', 1:num_families, 'XTickLabel', family_names, 'FontSize', 12);
ylabel('Relative Model Contribution (within family)');
title('Within-Family Model Contributions (stacked) (Perceptual condition, Sigma Fixed)');
ylim([0 1]);
grid on; box on;

legend(handles, labels, 'Location', 'eastoutside');

hold off;

%% Statistical Summary (Perceptual)
fprintf('\n=== STATISTICAL SUMMARY (PERCEPTUAL, SIGMA FIXED) ===\n');
fprintf('Winning Family: %s (Prob = %.3f, XP = %.3f)\n', ...
    family_names{find(family_prob == max(family_prob))}, ...
    max(family_prob), max(family_xp));
fprintf('Winning Model: %s (Prob = %.3f, XP = %.3f)\n', ...
    model_names{find(model_prob == max(model_prob))}, ...
    max(model_prob), max(model_xp));

if max(family_xp) > 0.95
    evidence_strength = 'Very Strong';
elseif max(family_xp) > 0.90
    evidence_strength = 'Strong';
elseif max(family_xp) > 0.75
    evidence_strength = 'Moderate';
else
    evidence_strength = 'Weak';
end
fprintf('Evidence Strength: %s (XP = %.3f)\n', evidence_strength, max(family_xp));

%% Manual Family BMS Implementation (fallback function)
function [posterior, out] = manual_family_BMS(lme, families, family_names)
    fprintf('Implementing family-wise BMS manually...\n');

    [n_subjects, n_models] = size(lme);
    n_families = length(families);

    % Step 1: Compute family-level log evidence
    family_lme = zeros(n_subjects, n_families);

    for f = 1:n_families
        models_in_family = families{f};
        % For each subject, compute log sum exp of models in family
        for s = 1:n_subjects
            family_evidence = lme(s, models_in_family);
            % Numerical stable log-sum-exp
            max_ev = max(family_evidence);
            family_lme(s, f) = max_ev + log(sum(exp(family_evidence - max_ev)));
        end
    end

    % Step 2: Run BMS on family-level evidence using SPM functions
    try
        [alpha_fam, exp_r_fam, xp_fam] = spm_BMS(family_lme, 1e6, 0, 0, 1, ones(1, n_families));
    catch
        % Fallback if SPM not available
        fprintf('SPM not available, using simplified BMS...\n');
        % Simple softmax over mean log evidence
        mean_family_lme = mean(family_lme, 1);
        exp_r_fam = exp(mean_family_lme - max(mean_family_lme));
        exp_r_fam = exp_r_fam / sum(exp_r_fam);
        xp_fam = exp_r_fam;  % Simplified exceedance probabilities
    end

    % Step 3: Compute model-level probabilities within families
    model_prob = zeros(1, n_models);
    for f = 1:n_families
        models_in_family = families{f};
        if length(models_in_family) == 1
            % Single model in family
            model_prob(models_in_family) = exp_r_fam(f);
        else
            % Multiple models in family - run BMS within family
            family_models_lme = lme(:, models_in_family);
            try
                [~, exp_r_within, ~] = spm_BMS(family_models_lme, 1e6, 0, 0, 1, []);
            catch
                % Fallback method
                mean_model_lme = mean(family_models_lme, 1);
                exp_r_within = exp(mean_model_lme - max(mean_model_lme));
                exp_r_within = exp_r_within / sum(exp_r_within);
            end
            % Weight by family probability
            model_prob(models_in_family) = exp_r_fam(f) * exp_r_within;
        end
    end

    % Step 4: Format output
    posterior.r = exp_r_fam;           % Family probabilities
    posterior.r_model = model_prob;     % Model probabilities

    out.families.ep = xp_fam;          % Family exceedance probabilities
    out.ep = zeros(1, n_models);       % Model exceedance probabilities

    % Approximate model exceedance probabilities
    for m = 1:n_models
        out.ep(m) = model_prob(m);  % Simplified approximation
    end

    fprintf('Manual family BMS completed.\n');
end
