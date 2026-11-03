clc
clearvars
close all

%% =================== LOAD AND PREPARE DATA =============================

% Load fitted parameters for all models
params_basicRL      = importdata("params_basicRL.mat");
params_RLsigma      = importdata("params_RLsigma.mat");
params_PWRL         = importdata("params_PWRL.mat");
params_bayesianAgent= importdata("params_bayesianAgent.mat");

data = importdata("preprocessed_dataFitting.mat");
data = data(data.choice_cond ~= 3,:);
data = data(data.pe ~= 0, :);
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

% Precompute contrast difference
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;

model_names = {'basicRL','RL+EstSens','PWRL','BayesianAgent'};
nModels = numel(model_names);

actual_updates_all = {};
predicted_updates_all = cell(1, nModels);

for m = 1:nModels
    predicted_updates_all{m} = {};
end

for n = 1:98
    % Extract subject-specific data
    cond = importdata("preprocessed_dataFitting.mat");
    cond = cond(cond.ID == uniqueID(n),:);
    cond = cond(cond.choice_cond ~= 3, :);
    condition = cond.condition(cond.trials./1 == 1);

    subj = preprocess_fitSlider(data, uniqueID(n));
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    actual_updates = [NaN; diff(subj.dataTable.up)];
    actual_updates_all{end+1} = actual_updates;

    % --- Compute predicted updates for each model using RLModelPredictor ---
    predicted_updates = cell(1, nModels);

    % 1. basicRL
    [predicted_updates{1}, ~] = predict_allModels.predict_basicRL( ...
        [params_basicRL.alpha(n), params_basicRL.kappa(n)], ...
        subj.blocks, rewards, subj.state, 'sample');

    % 2. RL+EstSens
    [predicted_updates{2}, ~] = predict_allModels.predict_RLsigma( ...
        [params_RLsigma.alpha(n), params_RLsigma.kappa(n), params_RLsigma.sigma(n)], ...
        subj.blocks, rewards, subj.state, subj.condiff, 'sample');

    % 3. PWRL
    [predicted_updates{3}, ~] = predict_allModels.predict_PWRL( ...
        [params_PWRL.alpha(n), params_PWRL.kappa(n), params_PWRL.sigma(n)], ...
        subj.blocks, rewards, subj.condiff, subj.choices, ...
        subj.recoded_rewards, subj.state, 'sample',subj.contrast);

    % 4. BayesianAgent
    [predicted_updates{4}, ~] = predict_allModels.predict_bayesianAgent( ...
        [params_bayesianAgent.kappa(n), params_bayesianAgent.sigma(n)], ...
        subj.blocks, subj.rewards, subj.condiff, subj.choices, condition, 'sample');

    for m = 1:nModels
        predicted_updates_all{m}{end+1} = predicted_updates{m};
    end

    %% ----------- SIMULATIONS (Example for basicRL) -----------
    % nSimulations = 500;
    % simulated_updates = zeros(length(subj.state), nSimulations);
    % for s = 1:nSimulations
    %     [sim_mu_hat, ~] = RLModelPredictor.predict_basicRL( ...
    %         [params_basicRL.alpha(n), params_basicRL.kappa(n)], ...
    %         subj.blocks, rewards, subj.state, 'sample');
    %     simulated_updates(:, s) = sim_mu_hat;
    % end
    % 
    % figure
    % scatter(1:size(simulated_updates,1), nanmean(simulated_updates, 2)); hold on;
    % scatter(1:length(actual_updates), actual_updates, 'k');
    % legend('Model mean update', 'Actual update');
    % title(['Subject ' num2str(uniqueID(n))]);
end

% Concatenate all subjects' updates for summary plots
actual_all = vertcat(actual_updates_all{:});
predicted_all = cell(1, nModels);
for m = 1:nModels
    predicted_all{m} = vertcat(predicted_updates_all{m}{:});
end

%% ----------- HISTOGRAMS OF ACTUAL VS PREDICTED UPDATES -----------
figure;
for m = 1:nModels
    subplot(2,2,m)
    % Remove NaNs
    actual_flat = actual_all(~isnan(actual_all));
    predicted_flat = predicted_all{m}(~isnan(predicted_all{m}));
    % Common bin edges
    minval = min([actual_flat; predicted_flat]);
    maxval = max([actual_flat; predicted_flat]);
    bins = linspace(minval, maxval, 75);
    histogram(actual_flat, bins, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Actual');
    hold on;
    histogram(predicted_flat, bins, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Predicted');
    hold off;
    xlabel('Update Value');
    ylabel('Frequency');
    title(['Actual vs Predicted Updates - ' model_names{m}]);
    legend();
end
sgtitle('Histogram of Actual vs Predicted Updates Across All Subjects');
