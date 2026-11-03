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
% data = data(data.pe ~= 0, :);
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

% Precompute contrast difference
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;

model_names = {'basicRL','RLEstSens','PWRL','BayesianAgent'};
nModels = numel(model_names);

% Clean model names for struct field usage
model_fieldnames = regexprep(model_names, '[^a-zA-Z0-9]', '_');

actual_updates_all = {};
predicted_updates_all = cell(1, nModels);

% Use cleaned field names for struct
simulated_data = struct();
for m = 1:nModels
    simulated_data.(model_fieldnames{m}) = cell(numSubjs, 1);
end
nSimulations = 5;  % Number of simulations per model/subject

%% =================== RUN SIMULATIONS =================================
for n = 1:numSubjs

    cond = importdata("preprocessed_dataFitting.mat");
    cond = cond(cond.ID == uniqueID(n),:);
    cond = cond(cond.choice_cond ~= 3, :);
    condition = cond.condition(cond.trials./1 == 1);

    % --- Subject-specific setup ---
    subj = preprocess_fitSlider(data, uniqueID(n));
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    
    % --- Run simulations for each model ---
    for m = 1:nModels
        model_name = model_names{m};
        nTrials = length(subj.blocks);
        simulations = zeros(nTrials, nSimulations);
        
        % Get appropriate parameters
        switch model_name
            case 'basicRL'
                params = [params_basicRL.alpha(n), params_basicRL.kappa(n)];
            case 'RLEstSens'
                params = [params_RLsigma.alpha(n), params_RLsigma.kappa(n), params_RLsigma.sigma(n)];
            case 'PWRL'
                params = [params_PWRL.alpha(n), params_PWRL.kappa(n), params_PWRL.sigma(n)];
            case 'BayesianAgent'
                params = [params_bayesianAgent.kappa(n), params_bayesianAgent.sigma(n)];
        end
        
        % Run simulations
        for s = 1:nSimulations
            switch model_name
                case 'basicRL'
                    [predictedUp, mu_hat] = predict_allModels.predict_basicRL(...
                        params, subj.blocks, rewards, subj.state, 'sample');
                case 'RLEstSens'
                    [predictedUp, mu_hat] = predict_allModels.predict_RLsigma(...
                        params, subj.blocks, rewards, subj.state, subj.condiff, 'sample');
                case 'PWRL'
                    [predictedUp, mu_hat] = predict_allModels.predict_PWRL(...
                        params, subj.blocks, subj.rewards, subj.condiff, subj.choices, ...
                        subj.recoded_rewards, subj.state, 'sample');
                case 'BayesianAgent'
                    [predictedUp, mu_hat] = predict_allModels.predict_bayesianAgent(...
                        params, subj.blocks, subj.rewards, subj.condiff, ...
                        subj.choices, condition, 'sample');
            end
            simulations(:, s) = predictedUp;
        end
        
        simulated_data.(model_name){n} = simulations;
    end
end

%% =================== ORGANIZE SIMULATION DATA ========================
% Create trial-wise structure for analysis
analysis_data = struct();

for n = 1:numSubjs
    subj_id = uniqueID(n);
    subj_data = preprocess_fitSlider(data, subj_id);
    nTrials = length(subj_data.blocks);
    
    % Initialize subject structure
    analysis_data(n).SubjectID = subj_id;
    analysis_data(n).Trials = struct();
    
    % Populate trial data
    for t = 1:nTrials
        analysis_data(n).Trials(t).TrialNumber = t;
        analysis_data(n).Trials(t).PE = subj_data.dataTable.pe(t);
        analysis_data(n).Trials(t).EstimationError = subj_data.dataTable.est_error(t);
        analysis_data(n).Trials(t).ActualUpdate = subj_data.dataTable.up(t);
        analysis_data(n).Trials(t).condiff = subj_data.dataTable.condiff_relative(t);
        analysis_data(n).Trials(t).mu = subj_data.dataTable.mu_congruence(t);

        % Add simulated updates from all models
        for m = 1:nModels
            model_name = model_names{m};
            analysis_data(n).Trials(t).(model_name).SimUpdates = ...
                squeeze(simulated_data.(model_name){n}(t, :))';
        end
    end
end

%% =================== ANALYSIS AND PLOTTING SCRIPT ====================
% Load simulation results
% load('simulation_results.mat');

% Setup
nModels = length(model_names);
colors = [0.8 0.2 0.2; 0.2 0.6 0.8; 0.4 0.8 0.2; 0.8 0.6 0.2; 0.6 0.2 0.8]; % Colors for actual + 4 models
figure_size = [100, 100, 1200, 800];

%% =================== PREPARE DATA FOR ANALYSIS =======================
% Initialize data containers
all_pe = [];
all_condiff = [];
all_actual_up = [];
all_mu_actual = [];
all_trial_in_block = [];
all_condition = [];
all_subject_id = [];

% Store simulation data with proper indexing
trial_counter = 0;
sim_data_struct = struct();

% First pass: collect basic data and count total trials
for n = 1:length(analysis_data)
    nTrials = length(analysis_data(n).Trials);
    
    % Get subject-specific data for condition assignment
    subj_id = analysis_data(n).SubjectID;
    try
        cond = importdata("preprocessed_dataFitting.mat");
        cond = cond(cond.ID == subj_id,:);
        cond = cond(cond.choice_cond ~= 3, :);
        condition_vec = cond.condition;
    catch
        condition_vec = ones(nTrials, 1); % Default condition if data not found
    end
    
    for t = 1:nTrials
        trial_counter = trial_counter + 1;
        trial_data = analysis_data(n).Trials(t);
        
        % Basic trial data
        all_pe = [all_pe; trial_data.PE];
        all_condiff = [all_condiff; trial_data.condiff];
        all_actual_up = [all_actual_up; trial_data.ActualUpdate];
        all_mu_actual = [all_mu_actual; trial_data.mu];
        all_subject_id = [all_subject_id; subj_id];
        
        % Condition and trial number
        if t <= length(condition_vec)
            all_condition = [all_condition; condition_vec(t)];
        else
            all_condition = [all_condition; 1]; % Default condition
        end
        
        % Calculate trial within block (assuming blocks of 40 trials)
        trial_in_block = mod(t-1, 40) + 1;
        all_trial_in_block = [all_trial_in_block; trial_in_block];
        
        % Store simulation data with trial index
        for m = 1:nModels
            model_name = model_names{m};
            sim_data = trial_data.(model_name).SimUpdates;
            
            % Ensure sim_data is a row vector
            if size(sim_data, 1) > size(sim_data, 2)
                sim_data = sim_data';
            end
            
            sim_data_struct.(model_name)(trial_counter, :) = sim_data;
        end
    end
end

% Now calculate means and standard errors with consistent dimensions
total_trials = trial_counter;
sim_means = zeros(total_trials, nModels);
sim_ses = zeros(total_trials, nModels);

for m = 1:nModels
    model_name = model_names{m};
    model_data = sim_data_struct.(model_name);
    
    % Calculate mean and SE across simulations (columns)
    sim_means(:, m) = mean(model_data, 2);
    sim_ses(:, m) = std(model_data, 0, 2) / sqrt(size(model_data, 2));
end

fprintf('Data preparation complete: %d total trials, %d models\n', total_trials, nModels);

%% =================== PLOT 1: PE vs UPDATES ===========================
figure('Position', figure_size);

subplot(2, 2, 1);
hold on;

% Bin data for cleaner visualization
pe_bins = linspace(min(all_pe), max(all_pe), 20);
bin_centers = (pe_bins(1:end-1) + pe_bins(2:end)) / 2;

% Bin actual data
[~, bin_idx] = histc(all_pe, pe_bins);
actual_binned = zeros(length(bin_centers), 1);
actual_se = zeros(length(bin_centers), 1);

for i = 1:length(bin_centers)
    mask = bin_idx == i;
    if sum(mask) > 0
        actual_binned(i) = mean(all_actual_up(mask));
        actual_se(i) = std(all_actual_up(mask)) / sqrt(sum(mask));
    end
end

% Plot actual data
errorbar(bin_centers, actual_binned, actual_se, 'o-', 'Color', colors(1,:), ...
    'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Actual');

% Plot simulated data for each model
for m = 1:nModels
    sim_binned = zeros(length(bin_centers), 1);
    sim_binned_se = zeros(length(bin_centers), 1);
    
    for i = 1:length(bin_centers)
        mask = bin_idx == i;
        if sum(mask) > 0
            sim_binned(i) = mean(sim_means(mask, m));
            sim_binned_se(i) = mean(sim_ses(mask, m));
        end
    end
    
    errorbar(bin_centers, sim_binned, sim_binned_se, 's-', 'Color', colors(m+1,:), ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', model_names{m});
end

% Add regression lines
lsline;
xlabel('Prediction Error (PE)');
ylabel('Update');
title('Updates vs Prediction Error');
legend('Location', 'best');
grid on;

%% =================== PLOT 2: CONDIFF vs UPDATES ======================
subplot(2, 2, 2);
hold on;

% Bin contrast difference data
condiff_bins = linspace(min(all_condiff), max(all_condiff), 15);
bin_centers_condiff = (condiff_bins(1:end-1) + condiff_bins(2:end)) / 2;

% Bin actual data
[~, bin_idx_condiff] = histc(all_condiff, condiff_bins);
actual_binned_condiff = zeros(length(bin_centers_condiff), 1);
actual_se_condiff = zeros(length(bin_centers_condiff), 1);

for i = 1:length(bin_centers_condiff)
    mask = bin_idx_condiff == i;
    if sum(mask) > 0
        actual_binned_condiff(i) = mean(all_actual_up(mask));
        actual_se_condiff(i) = std(all_actual_up(mask)) / sqrt(sum(mask));
    end
end

% Plot actual data
errorbar(bin_centers_condiff, actual_binned_condiff, actual_se_condiff, 'o-', ...
    'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Actual');

% Plot simulated data for each model
for m = 1:nModels
    sim_binned_condiff = zeros(length(bin_centers_condiff), 1);
    sim_binned_se_condiff = zeros(length(bin_centers_condiff), 1);
    
    for i = 1:length(bin_centers_condiff)
        mask = bin_idx_condiff == i;
        if sum(mask) > 0
            sim_binned_condiff(i) = mean(sim_means(mask, m));
            sim_binned_se_condiff(i) = mean(sim_ses(mask, m));
        end
    end
    
    errorbar(bin_centers_condiff, sim_binned_condiff, sim_binned_se_condiff, 's-', ...
        'Color', colors(m+1,:), 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', model_names{m});
end

% Add regression lines
lsline;
xlabel('Contrast Difference');
ylabel('Update');
title('Updates vs Contrast Difference');
legend('Location', 'best');
grid on;

%% =================== PLOT 3: LEARNING CURVES =======================
% Create learning curves for different conditions
unique_conditions = unique(all_condition(~isnan(all_condition)));
n_conditions = min(2, length(unique_conditions)); % Plot max 2 conditions

for cond_idx = 1:n_conditions
    subplot(2, 2, 2 + cond_idx);
    hold on;
    
    condition_val = unique_conditions(cond_idx);
    
    % Initialize arrays for learning curves
    max_trials = 40; % Assuming 40 trials per block
    actual_curve = zeros(max_trials, 1);
    actual_curve_se = zeros(max_trials, 1);
    sim_curves = zeros(max_trials, nModels);
    sim_curves_se = zeros(max_trials, nModels);
    
    % Calculate actual learning curve
    for trial = 1:max_trials
        mask = (all_condition == condition_val) & (all_trial_in_block == trial);
        if sum(mask) > 0
            actual_curve(trial) = mean(all_mu_actual(mask));
            actual_curve_se(trial) = std(all_mu_actual(mask)) / sqrt(sum(mask));
        end
    end
    
    % For simulated learning curves, we need to reconstruct belief evolution
    % Since we have simulated updates, we can show the average simulated update pattern
    for m = 1:nModels
        for trial = 1:max_trials
            mask = (all_condition == condition_val) & (all_trial_in_block == trial);
            if sum(mask) > 0
                % Show simulated updates scaled to match the belief range
                sim_curves(trial, m) = mean(sim_means(mask, m));
                sim_curves_se(trial, m) = mean(sim_ses(mask, m));
            end
        end
    end
    
    % Plot actual learning curve
    errorbar(1:max_trials, actual_curve, actual_curve_se, 'o-', ...
        'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 4, 'DisplayName', 'Actual μ');
    
    % Plot simulated update patterns (secondary y-axis might be needed)
    yyaxis right;
    for m = 1:nModels
        errorbar(1:max_trials, sim_curves(:, m), sim_curves_se(:, m), 's-', ...
            'Color', colors(m+1,:), 'LineWidth', 1.5, 'MarkerSize', 3, ...
            'DisplayName', [model_names{m} ' Updates']);
    end
    ylabel('Simulated Updates');
    
    yyaxis left;
    xlabel('Trial in Block');
    ylabel('Actual Belief State (μ)');
    title(sprintf('Learning Pattern - Condition %d', condition_val));
    legend('Location', 'best');
    grid on;
end

% Add overall title
sgtitle('Model Simulation Analysis', 'FontSize', 16, 'FontWeight', 'bold');

%%

%% =================== PLOT 3: LEARNING CURVES =======================
% Create learning curves for different conditions
unique_conditions = unique(all_condition(~isnan(all_condition)));
n_conditions = min(2, length(unique_conditions)); % Plot max 2 conditions

for cond_idx = 1:n_conditions
    subplot(2, 2, 2 + cond_idx);
    hold on;
    condition_val = unique_conditions(cond_idx);
    
    % Initialize arrays for learning curves
    max_trials = 40; % Assuming 40 trials per block
    actual_curve = zeros(max_trials, 1);
    actual_curve_se = zeros(max_trials, 1);
    sim_curves = zeros(max_trials, nModels);
    sim_curves_se = zeros(max_trials, nModels);
    
    % Calculate actual learning curve
    for trial = 1:max_trials
        mask = (all_condition == condition_val) & (all_trial_in_block == trial);
        if sum(mask) > 0
            actual_curve(trial) = mean(all_mu_actual(mask));
            actual_curve_se(trial) = std(all_mu_actual(mask)) / sqrt(sum(mask));
        end
    end
    
    % For simulated learning curves, we need to reconstruct belief evolution
    % Since we have simulated updates, we can show the average simulated update pattern
    for m = 1:nModels
        for trial = 1:max_trials
            mask = (all_condition == condition_val) & (all_trial_in_block == trial);
            if sum(mask) > 0
                % Show simulated updates scaled to match the belief range
                sim_curves(trial, m) = mean(sim_means(mask, m));
                sim_curves_se(trial, m) = mean(sim_ses(mask, m));
            end
        end
    end
    
    % Plot actual learning curve with shaded error bar
    x_vals = 1:max_trials;
    actual_upper = actual_curve + actual_curve_se;
    actual_lower = actual_curve - actual_curve_se;
    
    % Create shaded area for actual data
    fill([x_vals, fliplr(x_vals)], [actual_upper', fliplr(actual_lower')], ...
         colors(1,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % Plot actual line
    plot(x_vals, actual_curve, 'o-', 'Color', colors(1,:), ...
         'LineWidth', 2, 'MarkerSize', 4, 'DisplayName', 'Actual μ');
    
    % Plot simulated update patterns with shaded error bars (secondary y-axis)
    yyaxis right;
    for m = 1:nModels
        sim_upper = sim_curves(:, m) + sim_curves_se(:, m);
        sim_lower = sim_curves(:, m) - sim_curves_se(:, m);
        
        % Create shaded area for simulated data
        fill([x_vals, fliplr(x_vals)], [sim_upper', fliplr(sim_lower')], ...
             colors(m+1,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        
        % Plot simulated line
        plot(x_vals, sim_curves(:, m), 's-', 'Color', colors(m+1,:), ...
             'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', [model_names{m} ' Updates']);
    end
    
    ylabel('Simulated Updates');
    yyaxis left;
    xlabel('Trial in Block');
    ylabel('Actual Belief State (μ)');
    title(sprintf('Learning Pattern - Condition %d', condition_val));
    legend('Location', 'best');
    grid on;
end

% Add overall title
sgtitle('Model Simulation Analysis', 'FontSize', 16, 'FontWeight', 'bold');
%% =================== STATISTICAL SUMMARY ==========================
fprintf('\n=== SIMULATION ANALYSIS SUMMARY ===\n');
fprintf('Number of subjects: %d\n', length(analysis_data));
fprintf('Total trials analyzed: %d\n', length(all_pe));
fprintf('Models compared: %s\n', strjoin(model_names, ', '));

% Correlation analysis
fprintf('\n=== CORRELATIONS WITH ACTUAL UPDATES ===\n');
actual_pe_corr = corr(all_pe, all_actual_up, 'rows', 'complete');
actual_condiff_corr = corr(all_condiff, all_actual_up, 'rows', 'complete');

fprintf('Actual PE vs Updates: r = %.3f\n', actual_pe_corr);
fprintf('Actual ConDiff vs Updates: r = %.3f\n', actual_condiff_corr);

fprintf('\nSimulated correlations:\n');
for m = 1:nModels
    pe_corr = corr(all_pe, sim_means(:, m), 'rows', 'complete');
    condiff_corr = corr(all_condiff, sim_means(:, m), 'rows', 'complete');
    fprintf('%s - PE: r = %.3f, ConDiff: r = %.3f\n', model_names{m}, pe_corr, condiff_corr);
end

% Model fit comparison (RMSE)
fprintf('\n=== MODEL FIT (RMSE) ===\n');
for m = 1:nModels
    rmse = sqrt(mean((all_actual_up - sim_means(:, m)).^2, 'omitnan'));
    fprintf('%s: RMSE = %.4f\n', model_names{m}, rmse);
end

fprintf('\nAnalysis complete. Figures saved.\n');