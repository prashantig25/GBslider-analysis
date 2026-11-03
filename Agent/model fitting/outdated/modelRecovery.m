clc
clearvars

% parameters
alpha = 0.2;
kappa = 20;
sigma = 0.04;
confirmBias = 0.7;
confirmBias_LR = 0.3;
noconfirmBias_LR = 0.25;

data = readtable("data_agent_condition1.txt");
data.choice = data.action;
data.confirm_rew = NaN(height(data),1); % Initialize result
data.obtained_reward = data.reward;
recoding = data.action .* ((-1) .^ (2 + data.obtained_reward)); % recoding for action = 0
data.recoded_reward = data.obtained_reward + recoding; % storing recoded values

% Define high contrast trials (μ < 0.5), where mismatch is better
contrast_one_idx = data.contrast == 1;

% Case: match (state == action) → chosen pair is better → use reward directly
data.confirm_rew(contrast_one_idx & (data.state == data.action)) = ...
    data.obtained_reward(contrast_one_idx & (data.state == data.action));

% Case: mismatch (state ~= action) → chosen pair is worse → use 1 - reward
data.confirm_rew(contrast_one_idx & (data.state ~= data.action)) = ...
    1 - data.obtained_reward(contrast_one_idx & (data.state ~= data.action));

low_contrast_idx = ~contrast_one_idx; % where μ > 0.5

% Case: match (state == action) → chosen pair is better → use reward directly
data.confirm_rew(low_contrast_idx & (data.state == data.action)) = ...
    data.obtained_reward(low_contrast_idx & (data.state == data.action));

% Case: mismatch (state ~= action) → chosen pair is worse → use 1 - reward
data.confirm_rew(low_contrast_idx & (data.state ~= data.action)) = ...
    1 - data.obtained_reward(low_contrast_idx & (data.state ~= data.action));

% Calculate total number of trials
n_trials = size(data, 1);

% Every 100 trials is a new subject
trials_per_subject = 100;
n_subjects = floor(n_trials / trials_per_subject); % Ensure integer

% Every 25 trials within a subject is a new block
trials_per_block = 25;
blocks_per_subject = trials_per_subject / trials_per_block; % Should be 4 blocks per subject

% Check if data length is compatible
if mod(n_trials, trials_per_subject) ~= 0
    warning('Number of trials (%d) is not evenly divisible by trials_per_subject (%d)', n_trials, trials_per_subject);
end

% Create subject ID column (only for complete subjects)
complete_trials = n_subjects * trials_per_subject;
subject_ids = repelem(1:n_subjects, trials_per_subject)';

% Create block column
% For each subject, blocks go 1, 2, 3, 4, then repeat for next subject
block_within_subject = repelem(1:blocks_per_subject, trials_per_block); % [1,1,1,...,2,2,2,...,3,3,3,...,4,4,4,...]
blocks = repmat(block_within_subject, 1, n_subjects)';

% Handle any remaining trials (incomplete last subject)
if n_trials > complete_trials
    remaining_trials = n_trials - complete_trials;
    remaining_subject_id = n_subjects + 1;
    remaining_blocks = ceil((1:remaining_trials) / trials_per_block);

    % Extend arrays for remaining trials
    subject_ids = [subject_ids; repmat(remaining_subject_id, remaining_trials, 1)];
    blocks = [blocks; remaining_blocks'];
end

% Add columns to data
data.subject_id = subject_ids;
data.blocks = blocks;
data.choices = data.action;
data.condiff_relative = abs(data.contrast_diff);

% Generate synthetic data using a known model (e.g., Model A with specific parameters)

% basicRL
[predictedUp, mu_hat] = predict_allModels.predict_basicRL(...
    [alpha, kappa, sigma], ...
    data.blocks, data.recoded_reward, data.state, data.contrast_diff,'sample');
data.mu_basicRL = mu_hat;

% RLSigma
[~, mu_hat] = predict_allModels.predict_RLsigma(...
    [alpha, kappa, sigma], ...
    data.blocks, data.recoded_reward, data.state, data.contrast_diff, 'sample');
data.mu_RLSigma = mu_hat;

% RLSigma + CB
[predictedUp, mu_hat] = predict_allModels.predict_RLsigma_confirmBias(...
    [kappa, sigma, confirmBias_LR, noconfirmBias_LR], ...
    data.blocks, data.recoded_reward, data.state, data.contrast_diff, data.confirm_rew, 'sample');
data.mu_RLSigmaCB = mu_hat;

% Agent
[~, mu_hat] = predict_allModels.predict_bayesianAgent(...
    [kappa, sigma], ...
    data.blocks, data.reward, data.contrast_diff, data.action, data.choice_cond, data.contrast, 'sample');
data.mu_Agent = mu_hat;

% Agent + CB
[~, mu_hat] = predict_allModels.predict_bayesianAgent_confirmBias(...
    [kappa, sigma, confirmBias], ...
    data.blocks, data.reward, data.contrast_diff, data.action, data.choice_cond, data.contrast, data.confirm_rew, 'sample');
data.mu_AgentCB = mu_hat;

% safe_saveall('data_modelRecovery.mat',data);

% basicRL + CB
% [~, mu_hat] = predict_allModels.predict_RLsigma(...
%     [alpha, kappa, sigma], ...
%     data.blocks, data.rewards, data.state, abs(data.contrast_diff), 'sample');
% data.mu_RLSigma = mu_hat;

%% Fit multiple candidate models to this synthetic data

data.mu_recovered = data.mu_RLSigmaCB;
% =================== BASIC RL MODEL ====================================

numSubjs = length(unique(data.subject_id));
% betaParameter = NaN(numSubjs, 1);
nll_basicRL = NaN(numSubjs, 1);
init_params = [0.1, 5, 0.05]; % , 0.5]; % [alpha, kappa]
lb = [0, 1, 0]; %, 0];
ub = [1, 100, 0.1]; %, 1];
parfor n = 1:numSubjs
    dataSubj = data(data.subject_id == n,:);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated(params, dataSubj.mu_recovered, dataSubj.blocks, dataSubj.recoded_reward, data.contrast_diff);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    nll_basicRL(n) = nll;

    fprintf('Subject number: %d\n', n);

end

safe_saveall('nll_basicRL_RLSigmaCB.mat', nll_basicRL);

% =================== BAYESIAN AGENT MODEL ==============================

nll_bayesianAgent = NaN(numSubjs, 1);
init_params = [5, 0.05]; %, 0.5]; % [kappa, sigma]
lb = [1, 0]; %, 0];
ub = [100, 0.1]; %,1];
parfor n = 1:numSubjs
    dataSubj = data(data.subject_id == n,:);
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent(params, dataSubj.mu_recovered, dataSubj, ...
        length(unique(dataSubj.blocks)), 25, unique(dataSubj.blocks), dataSubj.reward);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);

    nll_bayesianAgent(n) = nll;

    fprintf('Subject number: %d\n', n);
end

safe_saveall('nll_bayesianAgent_RLSigmaCB.mat', nll_bayesianAgent);

% =================== BAYESIAN AGENT CONFIRMATION BIAS MODEL ============

nll_bayesianAgent_confirmBias = NaN(numSubjs, 1);
init_params = [5, 0.05, 0.75]; %,0.5]; % [kappa, sigma]
lb = [1, 0, 0.5]; %,0];
ub = [100, 0.1, 1]; %,1];
parfor n = 1:numSubjs
    dataSubj = data(data.subject_id == n,:);
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent_confirmBias(params, dataSubj.mu_recovered, dataSubj, ...
        length(unique(dataSubj.blocks)), 25, unique(dataSubj.blocks), dataSubj.reward);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    nll_bayesianAgent_confirmBias(n) = nll;

    fprintf('Subject number: %d\n', n);

end

safe_saveall('nll_bayesianAgentCB_RLSigmaCB.mat', nll_bayesianAgent_confirmBias);

% =================== RL + EST SENSITIVITY MODEL ========================

nll_RLsigma = NaN(numSubjs, 1);
init_params = [0.1, 5, 0.01]; %, 0.5]; % [alpha, kappa, sigma]
lb = [0, 1, 0]; %, 0];
ub = [1, 100, 0.1]; %, 1];
parfor n = 1:numSubjs
    dataSubj = data(data.subject_id(n) == n,:);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI(params, dataSubj.mu_recovered, dataSubj.blocks, dataSubj.recoded_reward ...
        , dataSubj.contrast_diff);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);

    nll_RLsigma(n) = nll;

    fprintf('Subject number: %d\n', n);
end

safe_saveall('nll_RLsigma_RLSigmaCB.mat', nll_RLsigma);

% =================== RL + EST SENSITIVITY + CONFIRMBIAS MODEL ========================

nll_RLsigma = NaN(numSubjs, 1);
init_params = [5, 0.01, 0.1, 0.1]; %, 0.5]; % [alpha, kappa, sigma]
lb = [1, 0, 0, 0]; %, 0];
ub = [100, 0.1, 1, 1];%, 1];
parfor n = 1:numSubjs
    dataSubj = data(data.subject_id(n),:);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_confirmBias_VOI(params, dataSubj.mu_recovered, dataSubj.blocks, ...
        dataSubj.recoded_reward, dataSubj.contrast_diff, dataSubj.confirm_rew);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);

    nll_RLsigma(n) = nll;

    fprintf('Subject number: %d\n', n);
end

safe_saveall('nll_RLsigmaCB_RLSigmaCB.mat', nll_RLsigma);


% Check if your model comparison procedure correctly identifies Model A as the best fit

%% =================== COMPUTE AND SAVE AIC/BIC ==========================
nll_basicRL = importdata("nll_basicRL_RLSigmaCB.mat");
nll_RLsigma = importdata("nll_RLsigma_RLSigmaCB.mat");
nll_RLsigma_confirmBias = importdata("nll_RLsigmaCB_RLSigmaCB.mat");
nll_bayesianAgent = importdata("nll_bayesianAgent_RLSigmaCB.mat");
nll_bayesianAgent_confirmBias = importdata("nll_bayesianAgentCB_RLSigmaCB.mat");

disp('Computing AIC and BIC for all models...');
% Number of trials per subject (assuming all subjects have same number)
num_trials = repelem(100,numSubjs);%arrayfun(@(n) length(preprocess_fitSlider(data, uniqueID(n)).mu_hat,pupil), 1:numSubjs)';
% Model parameter counts (updated to include confirmation bias model)
num_params = [3, 3, 4, 2, 2]; % [basicRL, RLsigma, PWRL, BayesianAgent, BayesianAgent_confirmBias]
% num_params = num_params - 1;
% Preallocate (updated dimensions)
AIC = NaN(numSubjs, length(num_params));
BIC = NaN(numSubjs, length(num_params));
for n = 1:numSubjs
    % nlls = [nll_basicRL(n), nll_RLsigma(n), nll_PWRL(n), nll_bayesianAgent(n), nll_bayesianAgent_confirmBias(n)];
    nlls = [nll_basicRL(n), nll_RLsigma(n), nll_RLsigma_confirmBias(n), nll_bayesianAgent(n), nll_bayesianAgent_confirmBias(n)];
    [AIC(n,:), BIC(n,:)] = fitSlider_ALLmodels.compute_aic_bic(nlls, num_params, num_trials(n));
end
model_names = {'basicRL','RLsigma','RLSigma_confirmBias','BayesianAgent','BayesianAgent_confirmBias'};
AICBIC_table = array2table([AIC, BIC], 'VariableNames', ...
    {'AIC_basicRL','AIC_RLsigma','AIC_RLSigma_confirmBias','AIC_BayesianAgent','AIC_BayesianAgent_confirmBias', ...
     'BIC_basicRL','BIC_RLsigma','BIC_RLSigma_confirmBias','BIC_BayesianAgent','BIC_BayesianAgent_confirmBias'});
AICBIC_table.SubjectID = unique(data.subject_id);
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
% Save updated table
safe_saveall('AICBIC_basicRL_RLSigmaCB.mat', AICBIC_table);
disp('All model parameters and AIC/BIC estimated and saved.');

%% Plot proportion of subjects best described by each model
% Extract delta BIC values for each model
delta_BIC_data = [AICBIC_table.delta_BIC_basicRL, ...
                  AICBIC_table.delta_BIC_RLsigma, ...
                  AICBIC_table.delta_BIC_RLsigma_confirmBias, ...
                  AICBIC_table.delta_BIC_BayesianAgent, ...
                  AICBIC_table.delta_BIC_BayesianAgent_confirmBias];

% Find best model for each subject (deltaBIC = 0, or closest to 0 due to floating point)
[~, best_model_idx] = min(abs(delta_BIC_data), [], 2);

% Count proportion of subjects best described by each model
model_counts = histcounts(best_model_idx, 1:6);
model_proportions = model_counts / length(best_model_idx);

% Model names and colors
model_names = {'basicRL', 'RLsigma', 'RLSigma_confirmBias', 'BayesianAgent', 'BayesianAgent_confirmBias'};
colors = lines(6);

% Create the bar plot
figure;
h = bar(1:5, model_proportions, 'FaceColor', 'flat');

% Set individual bar colors and transparency
for i = 1:5
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
for i = 1:5
    text(i, model_proportions(i) + 0.02, sprintf('%.2f', model_proportions(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Add a horizontal grid for easier reading
grid on;
set(gca, 'GridAlpha', 0.3);

% Create the bar plot
figure;
h = bar(1:5, mean(delta_BIC_data), 'FaceColor', 'flat');

% Set individual bar colors and transparency
for i = 1:5
    h.CData(i,:) = colors(i,:);
end
h.FaceAlpha = 0.4;

% Customize the plot
set(gca, 'XTickLabel', model_names);
xlabel('Model');
ylabel('');
title('Mean delta BIC');
% ylim([0, 1]);

meanDelta = mean(delta_BIC_data);

% Add a horizontal grid for easier reading
grid on;
set(gca, 'GridAlpha', 0.3);

% Display results in command window
fprintf('\nModel Selection Results:\n');
fprintf('------------------------\n');
for i = 1:5
    fprintf('%s: %.1f%% of subjects (%d/%d)\n', ...
            model_names{i}, model_proportions(i)*100, model_counts(i), length(best_model_idx));
end