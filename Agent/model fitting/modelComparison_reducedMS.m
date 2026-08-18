%% ========================================================================
%  Script: Fit RL Models to Participant Data and Save Parameters (+AIC/BIC)
%  ------------------------------------------------------------------------
clc; clearvars;
%% =================== LOAD AND PREPARE DATA =============================
pupil = 0; % fit models to the pupil dataset
if pupil == 1
    data = readtable("/Users/prashantig/Brown Dropbox/Prashanti Ganesh/PhD/" + ...
        "Semester 8/pupil_manuscript/Perceptual_unc_aug_task_pupil-main/data/" + ...
        "GB data peak corrected/behavior/model fitting/preprocessed_lr_pupil_no_zerope.xlsx");
    uniqueID = unique(data.id);
    data.ID = data.id;
    for h = 1:height(data)
        if data.congruence(h) == 0
            data.mu_congruence(h) = 1-data.mu(h);
        else
            data.mu_congruence(h) = data.mu(h);
        end
    end
else
    data = importdata("preprocessed_dataFitting.mat");
    uniqueID = unique(data.ID);
    data = data(data.choice_cond ~= 3,:);
end
numSubjs = length(uniqueID);
% Precompute contrast difference
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;
data(data.condition == 2,:) = [];
%% =================== REDUCED COMPUTE AND SAVE AIC/BIC ==========================
nll_basicRL = importdata("nll_basicRL_sigma_integratedPerceptual.mat");
nll_RLsigma = importdata("nll_RLsigma_integratedBoth.mat");
nll_bayesianAgent = importdata("nll_bayesianAgent_integratedBoth.mat");
% nll_RLsigma = importdata("nll_RLsigma_confirmBias_integratedPerceptual.mat");
% nll_bayesianAgent = importdata("nll_basicRL_confirmBias_integratedPerceptual.mat");

disp('Computing AIC and BIC for all models...');
% Number of trials per subject (assuming all subjects have same number)
num_trials = repelem(200,numSubjs);%arrayfun(@(n) length(preprocess_fitSlider(data, uniqueID(n)).mu_hat,pupil), 1:numSubjs)';
% Model parameter counts (updated to include confirmation bias model)
num_params = [3, 4, 4]; % [basicRL, RLsigma, PWRL, BayesianAgent, BayesianAgent_confirmBias]
% num_params = num_params - 1;
% Preallocate (updated dimensions)
AIC = NaN(numSubjs, length(num_params));
BIC = NaN(numSubjs, length(num_params));
for n = 1:numSubjs
    % nlls = [nll_basicRL(n), nll_RLsigma(n), nll_PWRL(n), nll_bayesianAgent(n), nll_bayesianAgent_confirmBias(n)];
    nlls = [nll_basicRL(n), nll_RLsigma(n), nll_bayesianAgent(n)];
    [AIC(n,:), BIC(n,:)] = fitSlider_ALLmodels.compute_aic_bic(nlls, num_params, num_trials(n));
end
model_names = {'basicRL','RLsigma','BayesianAgent'};
% model_names = {'basicRL','RLsigma + CB','basicRL + CB'};

AICBIC_table = array2table([AIC, BIC], 'VariableNames', ...
    {'AIC_basicRL','AIC_RLsigma','AIC_BayesianAgent', ...
     'BIC_basicRL','BIC_RLsigma','BIC_BayesianAgent'});
AICBIC_table.SubjectID = uniqueID;
AICBIC_table = movevars(AICBIC_table, 'SubjectID', 'Before', 1);
% Compute delta BIC for each subject and model
% delta_BIC = BIC - min(BIC,[],2)
delta_BIC = BIC - min(BIC,[],2);
% Add delta_BIC to the table (updated to include confirmation bias model)
AICBIC_table.delta_BIC_basicRL = delta_BIC(:,1);
AICBIC_table.delta_BIC_RLsigma = delta_BIC(:,2);
AICBIC_table.delta_BIC_BayesianAgent = delta_BIC(:,3);
% Save updated table
safe_saveall('AICBIC_integratedBoth_sigma_reducedMS.mat', AICBIC_table);
disp('All model parameters and AIC/BIC estimated and saved.');

%% Plot proportion of subjects best described by each model
% Extract delta BIC values for each model
delta_BIC_data = [AICBIC_table.delta_BIC_basicRL, ...
                  AICBIC_table.delta_BIC_RLsigma, ...
                  AICBIC_table.delta_BIC_BayesianAgent];

% Find best model for each subject (deltaBIC = 0, or closest to 0 due to floating point)
[~, best_model_idx] = min(abs(delta_BIC_data), [], 2);

% Count proportion of subjects best described by each model
model_counts = histcounts(best_model_idx, 1:4);
model_proportions = model_counts / length(best_model_idx);

% Model names and colors
% model_names = {'basicRL', 'RLsigma', 'BayesianAgent'};
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
title('Proportion of Subjects Best Described by Each Model');
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
fprintf('\nModel Selection Results:\n');
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
fprintf('Running Bayesian Model Selection...\n');
[alpha, exp_r, xp] = spm_BMS(lme, 1e6, 1, 0, 1, []);

% Display results
fprintf('\n=== BAYESIAN MODEL SELECTION RESULTS ===\n\n');

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
model_labels = model_names;
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
sgtitle('Bayesian Model Selection Results', 'FontSize', 16, 'FontWeight', 'bold');

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
title('Model Comparison: Probabilities and Exceedance', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Model Probability (exp_r)', 'Exceedance Probability (xp)'}, ...
    'Location', 'northeast', 'FontSize', 11);
grid on;
ylim([0, 1.05]);

% Add text annotation for winner
text(winner_idx, max([exp_r(winner_idx), xp(winner_idx)]) + 0.1, ...
    sprintf('WINNER\n%s', model_labels{winner_idx}), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
    'Color', [0.1 0.7 0.1]);

