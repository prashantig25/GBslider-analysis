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
%% =================== BASIC RL MODEL ====================================
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
% betaParameter = NaN(numSubjs, 1);
nll_basicRL = NaN(numSubjs, 1);
init_params = [0.1, 5, 0.05]; % , 0.5]; % [alpha, kappa]
lb = [0, 1, 0]; %, 0];
ub = [1, 100, 0.1]; %, 1];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n), pupil);
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    alphaParameter(n) = params(1);
    kappaParameter(n) = params(2);
    sigmaParameter(n) = params(3);
    nll_basicRL(n) = nll;

    fprintf('Subject number: %d\n', n);

end
params_basicRL.alpha = alphaParameter;
params_basicRL.kappa = kappaParameter;
params_basicRL.sigma = sigmaParameter;
safe_saveall('params_basicRL_sigma_integratedBoth.mat', params_basicRL);
safe_saveall('nll_basicRL_sigma_integratedBoth.mat', nll_basicRL);
%% =================== BAYESIAN AGENT MODEL ==============================
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
% betaParameter = NaN(numSubjs, 1);
nll_bayesianAgent = NaN(numSubjs, 1);
init_params = [5, 0.05]; %, 0.5]; % [kappa, sigma]
lb = [1, 0]; %, 0];
ub = [100, 0.1]; %,1];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n), pupil);
    rewards = arrayfun(@(h) ...
        subj.rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent(params, subj.mu_hat, subj.dataTable, ...
        length(unique(subj.blocks)), 25, unique(subj.blocks), rewards.');
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    sigmaParameter(n) = params(2);
    kappaParameter(n) = params(1);
    % betaParameter(n) = params(3);
    nll_bayesianAgent(n) = nll;

    fprintf('Subject number: %d\n', n);
end
params_bayesianAgent.sigma = sigmaParameter;
params_bayesianAgent.kappa = kappaParameter;
% params_bayesianAgent.beta = betaParameter;
safe_saveall('params_bayesianAgent_integratedBoth.mat', params_bayesianAgent);
safe_saveall('nll_bayesianAgent_integratedBoth.mat', nll_bayesianAgent);
%% =================== BAYESIAN AGENT CONFIRMATION BIAS MODEL ============
sigmaParameter_CB = NaN(numSubjs, 1);
kappaParameter_CB = NaN(numSubjs, 1);
confirmBiasParameter_CB = NaN(numSubjs, 1);
% betaParameter = NaN(numSubjs, 1);
nll_bayesianAgent_confirmBias = NaN(numSubjs, 1);
init_params = [5, 0.05, 0.75]; %,0.5]; % [kappa, sigma]
lb = [1, 0, 0.5]; %,0];
ub = [100, 0.1, 1]; %,1];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n),pupil);
    rewards = arrayfun(@(h) ...
        subj.rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent_confirmBias(params, subj.mu_hat, subj.dataTable, ...
        length(unique(subj.blocks)), 25, unique(subj.blocks), rewards.');
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    sigmaParameter_CB(n) = params(2);
    kappaParameter_CB(n) = params(1);
    confirmBiasParameter_CB(n) = params(3);
    % betaParameter(n) = params(4);
    nll_bayesianAgent_confirmBias(n) = nll;

    fprintf('Subject number: %d\n', n);

end
params_bayesianAgent_confirmBias.sigma = sigmaParameter_CB;
params_bayesianAgent_confirmBias.kappa = kappaParameter_CB;
params_bayesianAgent_confirmBias.confirmBias = confirmBiasParameter_CB;
% params_bayesianAgent_confirmBias.beta = betaParameter;
safe_saveall('nll_bayesianAgent_confirmBias_integratedBoth.mat', nll_bayesianAgent_confirmBias);
safe_saveall('params_bayesianAgent_confirmBias_integratedBoth.mat', params_bayesianAgent_confirmBias);

%% =================== RL + EST SENSITIVITY MODEL ========================
alphaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
% betaParameter = NaN(numSubjs, 1);
nll_RLsigma = NaN(numSubjs, 1);
init_params = [0.1, 5, 0.01]; %, 0.5]; % [alpha, kappa, sigma]
lb = [0, 1, 0]; %, 0];
ub = [1, 100, 0.1]; %, 1];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n),pupil);
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    alphaParameter(n) = params(1);
    sigmaParameter(n) = params(3);
    kappaParameter(n) = params(2);
    % betaParameter(n) = params(4); 
    nll_RLsigma(n) = nll;

    fprintf('Subject number: %d\n', n);
end
params_RLsigma.alpha = alphaParameter;
params_RLsigma.sigma = sigmaParameter;
params_RLsigma.kappa = kappaParameter;
% params_RLsigma.beta = betaParameter;
safe_saveall('params_RLsigma_integratedBoth.mat', params_RLsigma);
safe_saveall('nll_RLsigma_integratedBoth.mat', nll_RLsigma);

%% =================== RL + EST SENSITIVITY + CONFIRMBIAS MODEL ========================
alphaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
confirmBiasParameter = NaN(numSubjs, 1);
noconfirmBiasParameter = NaN(numSubjs, 1);
% betaParameter = NaN(numSubjs, 1);
nll_RLsigma = NaN(numSubjs, 1);
init_params = [5, 0.01, 0.1, 0.1]; %, 0.5]; % [alpha, kappa, sigma]
lb = [1, 0, 0, 0]; %, 0];
ub = [100, 0.1, 1, 1];%, 1];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n),pupil);
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_confirmBias_VOI(params, subj.mu_hat, subj.blocks, rewards, subj.condiff, subj.dataTable.confirm_rew);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    % alphaParameter(n) = params(1);
    sigmaParameter(n) = params(2);
    kappaParameter(n) = params(1);
    confirmBiasParameter(n) = params(3);
    noconfirmBiasParameter(n) = params(4);
    % betaParameter(n) = params(5);
    nll_RLsigma(n) = nll;

    fprintf('Subject number: %d\n', n);
end
% params_RLsigma.alpha = alphaParameter;
params_RLsigma_CB.sigma = sigmaParameter;
params_RLsigma_CB.kappa = kappaParameter;
params_RLsigma_CB.confirmBias = confirmBiasParameter;
params_RLsigma_CB.noconfirmBias = noconfirmBiasParameter;
% params_RLsigma.beta = betaParameter;
safe_saveall('params_RLsigma_CB_RBVoi_integratedBoth.mat', params_RLsigma_CB);
safe_saveall('nll_RLsigma_CB_RBVoi_integratedBoth.mat', nll_RLsigma);

%% =================== basicRL + CONFIRMBIAS MODEL ========================
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
confirmBiasParameter = NaN(numSubjs, 1);
noconfirmBiasParameter = NaN(numSubjs, 1);
% betaParameter = NaN(numSubjs, 1);
nll_RLsigma = NaN(numSubjs, 1);
init_params = [5, 0.01, 0.1, 0.1]; %, 0.5]; % [alpha, kappa, sigma]
lb = [1, 0, 0, 0]; %, 0];
ub = [100, 0.1, 1, 1];%, 1];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n),pupil);
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_confirmBias_VOI(params, subj.mu_hat, subj.blocks, rewards, subj.condiff, subj.dataTable.confirm_rew);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    sigmaParameter(n) = params(2);
    kappaParameter(n) = params(1);
    confirmBiasParameter(n) = params(3);
    noconfirmBiasParameter(n) = params(4);
    % betaParameter(n) = params(5);
    nll_RLsigma(n) = nll;

    fprintf('Subject number: %d\n', n);
end
% params_RLsigma.alpha = alphaParameter;
params_basicRL_CB.sigma = sigmaParameter;
params_basicRL_CB.kappa = kappaParameter;
params_basicRL_CB.confirmBias = confirmBiasParameter;
params_RLsigparams_basicRL_CBma.noconfirmBias = noconfirmBiasParameter;
% params_RLsigma.beta = betaParameter;
safe_saveall('params_basicRL_CB_RBVoi_integratedBoth.mat', params_basicRL_CB);
safe_saveall('nll_basicRL_CB_RBVoi_integratedBoth.mat', nll_RLsigma);

%% =================== COMPUTE AND SAVE AIC/BIC ==========================
nll_basicRL = importdata("nll_basicRL_sigma_RBVoi_Both.mat");
nll_RLsigma = importdata("nll_RLSigma_RBVoi_Both.mat");
nll_RLsigma_confirmBias = importdata("nll_RLsigma_CB_RBVoi_integratedBoth.mat");
nll_bayesianAgent = importdata("nll_bayesianAgent_integratedBoth.mat");
nll_bayesianAgent_confirmBias = importdata("nll_bayesianAgent_confirmBias_integratedBoth.mat");
nll_basicRL_confirmBias = importdata("nll_basicRL_CB_RBVoi_integratedBoth.mat");

disp('Computing AIC and BIC for all models...');
% Number of trials per subject (assuming all subjects have same number)
num_trials = repelem(200,numSubjs);%arrayfun(@(n) length(preprocess_fitSlider(data, uniqueID(n)).mu_hat,pupil), 1:numSubjs)';
% Model parameter counts (updated to include confirmation bias model)
num_params = [3, 3, 4, 2, 2, 4]; % [basicRL, RLsigma, PWRL, BayesianAgent, BayesianAgent_confirmBias]
% num_params = num_params - 1;
% Preallocate (updated dimensions)
AIC = NaN(numSubjs, 6);
BIC = NaN(numSubjs, 6);
for n = 1:numSubjs
    % nlls = [nll_basicRL(n), nll_RLsigma(n), nll_PWRL(n), nll_bayesianAgent(n), nll_bayesianAgent_confirmBias(n)];
    nlls = [nll_basicRL(n), nll_RLsigma(n), nll_RLsigma_confirmBias(n), nll_bayesianAgent(n), nll_bayesianAgent_confirmBias(n), nll_basicRL_confirmBias(n)];
    [AIC(n,:), BIC(n,:)] = fitSlider_ALLmodels.compute_aic_bic(nlls, num_params, num_trials(n));
end
model_names = {'basicRL','RLsigma','RLSigma_confirmBias','BayesianAgent','BayesianAgent_confirmBias'};
AICBIC_table = array2table([AIC, BIC], 'VariableNames', ...
    {'AIC_basicRL','AIC_RLsigma','AIC_RLSigma_confirmBias','AIC_BayesianAgent','AIC_BayesianAgent_confirmBias','AIC_basicRL_confirmBias', ...
     'BIC_basicRL','BIC_RLsigma','BIC_RLSigma_confirmBias','BIC_BayesianAgent','BIC_BayesianAgent_confirmBias','BIC_basicRL_confirmBias'});
AICBIC_table.SubjectID = uniqueID;
AICBIC_table = movevars(AICBIC_table, 'SubjectID', 'Before', 1);
% Compute delta BIC for each subject and model
% delta_BIC = BIC - min(BIC,[],2)
delta_BIC = BIC - min(BIC,[],2);
% Add delta_BIC to the table (updated to include confirmation bias model)
AICBIC_table.delta_BIC_basicRL = delta_BIC(:,1);
AICBIC_table.delta_BIC_RLsigma = delta_BIC(:,2);
AICBIC_table.delta_BIC_RLsigma_confirmBias = delta_BIC(:,3);
AICBIC_table.delta_BIC_BayesianAgent = delta_BIC(:,4);
AICBIC_table.delta_BIC_BayesianAgent_confirmBias = delta_BIC(:,5);
AICBIC_table.delta_BIC_basicRL_confirmBias = delta_BIC(:,6);
% Save updated table
safe_saveall('AICBIC_RBVoi_integratedBoth.mat', AICBIC_table);
disp('All model parameters and AIC/BIC estimated and saved.');

%% Plot proportion of subjects best described by each model
% Extract delta BIC values for each model
delta_BIC_data = [AICBIC_table.delta_BIC_basicRL, ...
                  AICBIC_table.delta_BIC_RLsigma, ...
                  AICBIC_table.delta_BIC_RLsigma_confirmBias, ...
                  AICBIC_table.delta_BIC_BayesianAgent, ...
                  AICBIC_table.delta_BIC_BayesianAgent_confirmBias,...
                  AICBIC_table.delta_BIC_basicRL_confirmBias];

% Find best model for each subject (deltaBIC = 0, or closest to 0 due to floating point)
[~, best_model_idx] = min(abs(delta_BIC_data), [], 2);

% Count proportion of subjects best described by each model
model_counts = histcounts(best_model_idx, 1:7);
model_proportions = model_counts / length(best_model_idx);

% Model names and colors
model_names = {'basicRL', 'RLsigma', 'RLSigma_confirmBias', 'BayesianAgent', 'BayesianAgent_confirmBias','basicRL_confirmBias'};
colors = lines(6);

% Create the bar plot
figure;
h = bar(1:6, model_proportions, 'FaceColor', 'flat');

% Set individual bar colors and transparency
for i = 1:6
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
for i = 1:6
    text(i, model_proportions(i) + 0.02, sprintf('%.2f', model_proportions(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Add a horizontal grid for easier reading
grid on;
set(gca, 'GridAlpha', 0.3);

% Display results in command window
fprintf('\nModel Selection Results:\n');
fprintf('------------------------\n');
for i = 1:6
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
model_labels = {'Basic RL', 'RL Sigma', 'RL Sigma + CB', 'Bayesian', 'Bayesian + CB','BasicRL + CB'};
colors = lines(6); % Custom colors

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

