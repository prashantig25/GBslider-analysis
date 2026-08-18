clc
clearvars

%% =================== LOAD AND PREPARE DATA =============================
rng(54)
% Load fitted parameters for all models
params_basicRL      = importdata("params_basicRL.mat");
params_RLsigma      = importdata("params_RLsigma.mat");
params_PWRL         = importdata("params_PWRL.mat");
params_bayesianAgent= importdata("params_bayesianAgent.mat");

data = importdata("preprocessed_dataFitting.mat");
data = data(data.choice_cond ~= 3,:);
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

% Precompute contrast difference
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;

% LR
data.lr = data.up./data.pe;
data.absLR = abs(data.lr);

model_names = {'basicRL','RLEstSens','PWRL','BayesianAgent'};
nModels = numel(model_names);

% Clean model names for struct field usage
model_fieldnames = regexprep(model_names, '[^a-zA-Z0-9]', '_');

% Store both predicted updates and mu_hat values
simulated_data = struct();
predicted_mu_hat = struct();
simulated_pe = struct();
for m = 1:nModels
    simulated_data.(model_fieldnames{m}) = cell(numSubjs, 1);
    predicted_mu_hat.(model_fieldnames{m}) = cell(numSubjs, 1);
    simulated_pe.(model_fieldnames{m}) = cell(numSubjs, 1);
end
nSimulations = 200;  % Number of simulations per model/subject

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
        peSimulated = NaN(nTrials, nSimulations);
        mu_hat_simulations = zeros(nTrials, nSimulations);
        
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
                    [predictedUp, mu_hat, pe] = predict_allModels.predict_basicRL(...
                        params, subj.blocks, rewards, subj.state, 'sample');
                    peSimulated(:,s) = pe;
                    simulated_pe.('basicRL'){n} = peSimulated;
                case 'RLEstSens'
                    [predictedUp, mu_hat] = predict_allModels.predict_RLsigma(...
                        params, subj.blocks, rewards, subj.state, subj.condiff, 'sample');
                case 'PWRL'
                    [predictedUp, mu_hat] = predict_allModels.predict_PWRL(...
                        params, subj.blocks, subj.recoded_rewards, subj.condiff, subj.choices, ...
                        subj.recoded_rewards, subj.state, 'sample',subj.contrast);
                case 'BayesianAgent'
                    [predictedUp, mu_hat] = predict_allModels.predict_bayesianAgent(...
                        params, subj.blocks, subj.rewards, subj.condiff, ...
                        subj.choices, condition, 'sample');
                    for h = 1:height(predictedUp)

                        if subj.contrast(h) == 1
                            mu_hat(h) = 1-mu_hat(h);
                        end
                    end
            end
            simulations(:, s) = predictedUp;
            mu_hat_simulations(:, s) = mu_hat;
        end
        
        % Store both updates and mu_hat
        simulated_data.(model_name){n} = simulations;
        predicted_mu_hat.(model_name){n} = mu_hat_simulations;
    end 
end

safe_saveall('simulatedUP_analytical1.mat',simulated_data)
safe_saveall('simulatedMU_analytical1.mat',predicted_mu_hat)
safe_saveall('simulatedPE_analytical1.mat',simulated_pe)

%% =================== ORGANIZE SIMULATION DATA ========================
% Create trial-wise structure for analysis

simulated_data = importdata('simulatedUP_analytical1.mat');
predicted_mu_hat = importdata('simulatedMU_analytical1.mat');
simulated_pe = importdata('simulatedPE_analytical1.mat');

%%
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
        analysis_data(n).Trials(t).TrialNumber = subj_data.dataTable.trials(t);
        analysis_data(n).Trials(t).PE = subj_data.dataTable.pe(t);
        analysis_data(n).Trials(t).EstimationError = subj_data.dataTable.est_error(t);
        analysis_data(n).Trials(t).ActualUpdate = subj_data.dataTable.up(t);
        analysis_data(n).Trials(t).condiff = subj_data.dataTable.condiff_relative(t);
        analysis_data(n).Trials(t).mu = subj_data.dataTable.mu_congruence(t);
        analysis_data(n).Trials(t).lr = subj_data.dataTable.lr(t);
        analysis_data(n).Trials(t).absLR = subj_data.dataTable.absLR(t);
        analysis_data(n).Trials(t).contrast = subj_data.dataTable.contrast(t);

        % Add simulated updates and mu_hat from all models
        for m = 1:nModels
            model_name = model_names{m};
            analysis_data(n).Trials(t).(model_name).SimUpdates = ...
                squeeze(simulated_data.(model_name){n}(t, :))';
            analysis_data(n).Trials(t).(model_name).SimMuHat = ...
                squeeze(predicted_mu_hat.(model_name){n}(t, :))';
            analysis_data(n).Trials(t).(model_name).SimPE = ...
                nanmean(simulated_pe.('basicRL'){n}(t, :))';
        end
    end
end

% safe_saveall("fullSims_allmodels.mat",analysis_data);

%% =================== PREPARE DATA FOR ANALYSIS =======================
% Initialize data containers
all_pe = [];
all_condiff = [];
all_actual_up = [];
all_mu_actual = [];
all_trial_in_block = [];
all_condition = [];
all_subject_id = [];
all_LR = [];
all_absLR = [];
all_contrast = [];
all_absEE = [];

% Store simulation data with proper indexing
trial_counter = 0;
sim_data_struct = struct();
sim_mu_hat_struct = struct();
sim_pe_struct = struct();

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
        all_LR = [all_LR; trial_data.lr];
        all_absLR = [all_absLR; trial_data.absLR];
        all_absEE = [all_absEE; trial_data.EstimationError];
        all_contrast = [all_contrast; trial_data.contrast];
        
        % Condition and trial number
        if t <= length(condition_vec)
            all_condition = [all_condition; condition_vec(t)];
        else
            all_condition = [all_condition; 1]; % Default condition
        end
        
        % Calculate trial within block (assuming blocks of 40 trials)
        trial_in_block = mod(t-1, 25) + 1;
        all_trial_in_block = [all_trial_in_block; trial_in_block];
        
        % Store simulation data with trial index
        for m = 1:nModels
            model_name = model_names{m};
            sim_data = trial_data.(model_name).SimUpdates;
            sim_mu_data = trial_data.(model_name).SimMuHat;
            sim_pe_data = trial_data.('basicRL').SimPE;
            
            % Ensure data is a row vector
            if size(sim_data, 1) > size(sim_data, 2)
                sim_data = sim_data';
            end
            if size(sim_mu_data, 1) > size(sim_mu_data, 2)
                sim_mu_data = sim_mu_data';
            end
            
            sim_data_struct.(model_name)(trial_counter, :) = sim_data;
            sim_mu_hat_struct.(model_name)(trial_counter, :) = sim_mu_data;
            sim_pe_struct.('basicRL')(trial_counter, :) = sim_pe_data;
        end
    end
end

% Now calculate means and standard errors with consistent dimensions
total_trials = trial_counter;
sim_means = zeros(total_trials, nModels);
sim_ses = zeros(total_trials, nModels);
sim_mu_means = zeros(total_trials, nModels);
sim_mu_ses = zeros(total_trials, nModels);
sim_pe_means = NaN(total_trials, nModels);
sim_pe_ses = NaN(total_trials, nModels);
sim_lr_means = NaN(total_trials, nModels);
sim_lr_ses = NaN(total_trials, nModels);
sim_absLR_means = NaN(total_trials,nModels);
sim_absLR_ses = NaN(total_trials,nModels);
sim_EE_means = NaN(total_trials,nModels);

for m = 1:nModels
    model_name = model_names{m};
    model_data = sim_data_struct.(model_name);
    model_mu_data = sim_mu_hat_struct.(model_name);
    model_pe = sim_pe_struct.basicRL;
    model_lr = model_data./all_pe;
    model_absLR = abs(model_data)./abs(model_pe);

    idx = abs(model_lr(:,1)) > 1;     % Logical index for rows where abs(model_lr) > 1 in the first column
    model_lr(idx,1) = NaN;            % Set those values in model_lr(:,1) to NaN
    model_absLR(idx,1) = NaN;         % Set those values in model_absLR(:,1) to NaN

    % Calculate mean and SE across simulations (columns)
    sim_lr_means(:,m) = nanmean(model_lr,2);    
    sim_absLR_means(:,m) = nanmean(model_absLR,2);

    idx = abs(sim_lr_means(:,m)) > 1;     % Logical index for rows where abs(model_lr) > 1 in the first column
    sim_lr_means(idx,m) = NaN;            % Set those values in model_lr(:,1) to NaN
    idx = abs(sim_absLR_means(:,m)) > 1;     % Logical index for rows where abs(model_lr) > 1 in the first column
    sim_absLR_means(idx,m) = NaN;         % Set those values in model_absLR(:,1) to NaN

    sim_lr_ses(:,m) = nanstd(model_lr,0,2)./sqrt(nSimulations);
    sim_absLR_ses(:,m) = nanstd(model_absLR,0,2)./sqrt(nSimulations);
    sim_means(:, m) = nanmean(model_data, 2);
    sim_ses(:, m) = std(model_data, 0, 2) / sqrt(size(model_data, 2));
    sim_mu_means(:, m) = nanmean(model_mu_data, 2);
    sim_mu_ses(:, m) = std(model_mu_data, 0, 2) / sqrt(size(model_mu_data, 2));
    sim_pe_means(:,m) = nanmean(model_pe,2);
    sim_pe_ses(:,m) = std(model_pe,0,2)/sqrt(size(model_mu_data, 2));

    % add estimation error
    % sim_EE_means = nanmean(model_mu_data, 2);
end

fprintf('Data preparation complete: %d total trials, %d models\n', total_trials, nModels);

%% =================== SETUP PLOTTING PARAMETERS =======================

% Function to create binned plots
function [bin_centers, binned_data, binned_se] = create_binned_data(x_data, y_data, n_bins)
    x_bins = linspace(min(x_data), max(x_data), n_bins + 1);
    bin_centers = (x_bins(1:end-1) + x_bins(2:end)) / 2;
    [~, bin_idx] = histc(x_data, x_bins);
    
    binned_data = zeros(length(bin_centers), 1);
    binned_se = zeros(length(bin_centers), 1);
    
    for i = 1:length(bin_centers)
        mask = bin_idx == i;
        if sum(mask) > 0
            binned_data(i) = mean(y_data(mask));
            binned_se(i) = std(y_data(mask)) / sqrt(sum(mask));
        end
    end
end


%%

%% =================== DEBUG PE BINNING ISSUES =========================
% Script to diagnose and fix the large simulated updates in first PE bin

% First, let's examine the data distributions
figure('Position', [100, 100, 1400, 1000]);

% 1. Check the distribution of PE values
subplot(3,3,1);
histogram(all_pe(:,1), 50);
title('Distribution of PE values');
xlabel('PE');
ylabel('Frequency');
grid on;

% 2. Check the distribution of actual updates
subplot(3,3,2);
histogram(all_actual_up, 50);
title('Distribution of Actual Updates');
xlabel('Actual Updates');
ylabel('Frequency');
grid on;

% 3. Check simulated updates for each model
for m = 1:4
    subplot(3,3,2+m);
    histogram(sim_lr_means(:,m), 50);
    title(sprintf('Sim Updates - %s', model_names{m}));
    xlabel('Simulated Updates');
    ylabel('Frequency');
    grid on;
end



% 5. Check bin assignments
subplot(3,3,8);
histogram(bins, 1:11);
title('Bin Assignments');
xlabel('Bin Number');
ylabel('Count');
grid on;

% 6. Summary statistics for first bin
fprintf('\n=================== FIRST BIN DIAGNOSTICS ===================\n');
fprintf('First bin PE range: 0 to 0.1\n');
fprintf('Number of points in first bin: %d\n', sum(bins == 1));
fprintf('PE range in first bin: [%.4f, %.4f]\n', min(abs(all_pe(first_bin_indices,1))), max(abs(all_pe(first_bin_indices,1))));
fprintf('Actual updates in first bin - Mean: %.4f, Std: %.4f\n', ...
        nanmean(abs(all_actual_up(first_bin_indices))), nanstd(abs(all_actual_up(first_bin_indices))));

for m = 1:4
    fprintf('%s updates in first bin - Mean: %.4f, Std: %.4f\n', ...
            model_names{m}, nanmean(abs(sim_lr_means(first_bin_indices,m))), ...
            nanstd(abs(sim_lr_means(first_bin_indices,m))));
end

% 4. Diagnostic: Check what's happening in the first bin
figure
hold on 
numBins = 10;
quantiles = linspace(0, 1, numBins + 1);
binEdges = quantile(abs(all_pe(:,1)), quantiles);
[bins, ~] = discretize(abs(all_pe(:,1)), binEdges);
first_bin_indices = find(bins == 1);
scatter(all_pe(first_bin_indices,1), all_actual_up(first_bin_indices), 'o');
hold on;
for m = 1:4
    subplot(2,2,m)
    scatter(all_pe(first_bin_indices,1), sim_lr_means(first_bin_indices,m), 's');
end
title('First Bin Data Points');
xlabel('PE');
ylabel('Updates');
legend(['Actual', model_names]);
grid on;
%% =================== PROPOSED FIX =================================
% Fixed plotting code with better binning and error handling

figure('Position', [100, 100, 1200, 800]);
colors = [[0.5,0.5,0.5]; lines(4)];
markerSizeActual = 8;
markerSizeModels = 8;

% Create proper PE bins (centered approach)
pe_edges = -1:0.2:1;  % Use wider bins for more stable estimates
n_bins = length(pe_edges) - 1;

for m = 1:5
    subplot(2,3,m);
    hold on;
    
    % Bin the data using PE values (not absolute PE for binning)
    [~, bin_indices] = histcounts(all_pe(:,1), pe_edges);
    
    % Calculate bin centers
    bin_centers = pe_edges(1:end-1) + diff(pe_edges)/2;
    
    % Always plot actual data
    if m == 1
        [binnedUp, binnedUp_SEM] = deal(NaN(n_bins,1));
        for b = 1:n_bins
            bin_mask = bin_indices == b;
            if sum(bin_mask) >= 5  % Require at least 5 points per bin
                binnedUp(b) = nanmean(abs(all_actual_up(bin_mask)));
                binnedUp_SEM(b) = nanstd(abs(all_actual_up(bin_mask)))/sqrt(sum(bin_mask));
            end
        end
        
        errorbar(bin_centers, binnedUp, binnedUp_SEM, 'o', ...
                'Color', colors(1,:), 'LineWidth', 1, 'MarkerSize', markerSizeActual, ...
                'MarkerFaceColor', colors(1,:), 'DisplayName', 'Actual');
        title('Actual Data Only');
    else
        % Plot both actual and model data for comparison
        model_idx = m-1;
        
        % Actual data (for reference)
        [binnedUp_actual, binnedUp_SEM_actual] = deal(NaN(n_bins,1));
        [binnedUp_model, binnedUp_SEM_model] = deal(NaN(n_bins,1));
        
        for b = 1:n_bins
            bin_mask = bin_indices == b;
            if sum(bin_mask) >= 5
                % Actual data
                binnedUp_actual(b) = nanmean(abs(all_actual_up(bin_mask)));
                binnedUp_SEM_actual(b) = nanstd(abs(all_actual_up(bin_mask)))/sqrt(sum(bin_mask));
                
                % Model data
                binnedUp_model(b) = nanmean(abs(sim_lr_means(bin_mask, model_idx)));
                binnedUp_SEM_model(b) = nanstd(abs(sim_lr_means(bin_mask, model_idx)))/sqrt(sum(bin_mask));
            end
        end
        
        % Plot actual (gray)
        errorbar(bin_centers, binnedUp_actual, binnedUp_SEM_actual, 'o', ...
                'Color', colors(1,:), 'LineWidth', 1, 'MarkerSize', markerSizeActual, ...
                'MarkerFaceColor', colors(1,:), 'DisplayName', 'Actual');
        
        % Plot model
        errorbar(bin_centers, binnedUp_model, binnedUp_SEM_model, 's', ...
                'Color', colors(m,:), 'LineWidth', 1, 'MarkerSize', markerSizeModels, ...
                'MarkerFaceColor', colors(m,:), 'DisplayName', model_names{model_idx});
        
        title(sprintf('%s vs Actual', model_names{model_idx}));
        legend('Location', 'best');
    end
    
    xlabel('PE');
    ylabel('Abs LR');
    xlim([-1 1]);
    ylim([0 inf]);  % Ensure y-axis starts at 0
    grid on;
end

%% =================== ADDITIONAL CHECKS =============================
% Check for potential data issues

fprintf('\n=================== DATA QUALITY CHECKS ===================\n');

% Check for infinite or extremely large values
fprintf('Infinite values in PE: %d\n', sum(isinf(all_pe(:,1))));
fprintf('Infinite values in actual updates: %d\n', sum(isinf(all_actual_up)));

for m = 1:4
    fprintf('Infinite values in %s simulated updates: %d\n', model_names{m}, sum(isinf(sim_lr_means(:,m))));
    fprintf('Extremely large values (>10) in %s: %d\n', model_names{m}, sum(abs(sim_lr_means(:,m)) > 10));
end

% Check for NaN values
fprintf('\nNaN values in PE: %d\n', sum(isnan(all_pe(:,1))));
fprintf('NaN values in actual updates: %d\n', sum(isnan(all_actual_up)));

for m = 1:4
    fprintf('NaN values in %s simulated updates: %d\n', model_names{m}, sum(isnan(sim_lr_means(:,m))));
end

% Check data ranges
fprintf('\nData ranges:\n');
fprintf('PE range: [%.4f, %.4f]\n', min(all_pe(:,1)), max(all_pe(:,1)));
fprintf('Actual updates range: [%.4f, %.4f]\n', min(all_actual_up), max(all_actual_up));

for m = 1:4
    fprintf('%s updates range: [%.4f, %.4f]\n', model_names{m}, ...
            min(sim_lr_means(:,m)), max(sim_lr_means(:,m)));
end

%% PLOT 1: PE vs UPDATES

% all_pe(all_pe == 0) = [];
figure;
set(gcf, 'Position', [100 100 1200 800]); % Enlarge figure window

% Define colors and markers
colors = [[0.5,0.5,0.5,];lines(nModels)];
markerSizeActual = 8;
markerSizeModels = 8;

% Create 5 subplots in 2x3 grid (use positions 1-5)
for m = 1:5 % Assuming 5 total plots (1 actual + 4 models)
    subplot(2,3,m);
    hold on;
    
    % Always plot actual data in each subplot for reference
    bins = discretize(all_pe, 0.1:0.1:1);
    [bin_centers, binnedUp, binnedUp_SEM] = deal(NaN(10,1));
    
    for b = 1:10
        bin_centers = 0.1:0.1:1;
        binnedUp(b) = nanmean(abs(all_LR(bins == b)));
        binnedUp_SEM(b) = nanstd(abs(all_LR(bins == b)))/sqrt(sum(bins == b));
    end
    
    if m == 1
        errorbar(bin_centers, binnedUp, binnedUp_SEM, 'o', ...
            'Color', colors(1,:), 'LineWidth', 1, 'MarkerSize', markerSizeActual, ...
            'MarkerFaceColor', colors(1,:), 'DisplayName', 'Actual');
    end
    title("Empirical updates");  % Subplot 1 = "Model 0" (Actual)

    % Plot model data for current subplot
    if m > 1  % Subplot 1 is actual-only
        model_idx = m-1;
    bins = discretize(all_pe, 0.1:0.1:1);
        [binnedPe, modelUp, modelSEM] = deal(NaN(10,1));
        
        for b = 1:10
            binnedPe(b) = nanmean(abs(all_pe(bins == b, 1)));
            modelUp(b) = nanmean(abs(sim_lr_means(bins == b, model_idx)));
            modelSEM(b) = nanstd(abs(sim_lr_means(bins == b, model_idx)))/sqrt(sum(bins == b));
        end
        
        errorbar(bin_centers, flip(modelUp), flip(modelSEM), 's', ...
            'Color', colors(m,:), 'LineWidth', 1, 'MarkerSize', markerSizeModels, ...
            'MarkerFaceColor', colors(m,:), 'DisplayName', model_names{model_idx});
        hold on
        title(sprintf(string(model_names(m-1))));  % Subplot 1 = "Model 0" (Actual)
    end

    xlabel('PE');
    ylabel('Absolute LR');
    % xlim([-1 1]);
    grid on;
    
    if m == 1
        legend('Location', 'best');
    end
end


%% PLOT 2: CONDIFF vs UPDATES

% all_lr_nozeroPE = all_LR(abs(all_pe) > 0,1);
figure;
set(gcf, 'Position', [100 100 1200 800]); % Enlarge figure window

% Define colors and markers
colors = [[0.5,0.5,0.5,];lines(nModels)];
markerSizeActual = 8;
markerSizeModels = 8;

% Create 5 subplots in 2x3 grid (use positions 1-5)
for m = 1:5  % Assuming 5 total plots (1 actual + 4 models)
    subplot(2,3,m);
    hold on;
    
    % Always plot actual data in each subplot for reference
    all_condiffModel = all_condiff;
    all_condiffModel(all_pe == 0) = [];
    all_LRModel = all_LR;
    all_LRModel(all_pe == 0) = [];
    all_up = all_actual_up;
    all_up(all_pe == 0) = [];
    bins = discretize(abs(all_condiffModel), 0:0.01:0.1);
    [bin_centers, binnedUp, binnedUp_SEM] = deal(NaN(10,1));
    bin_centers = 0:0.01:0.09;

    for b = 1:10
        binnedUp(b) = nanmean(all_LRModel(bins == b));
        binnedUp_SEM(b) = nanstd(all_LRModel(bins == b))/sqrt(sum(bins == b));
    end
    
    if m == 1
        errorbar(bin_centers, binnedUp, binnedUp_SEM, 'o', ...
            'Color', colors(1,:), 'LineWidth', 1, 'MarkerSize', markerSizeActual, ...
            'MarkerFaceColor', colors(1,:), 'DisplayName', 'Actual');
    end

    % Plot model data for current subplot
    if m > 1  % Subplot 1 is actual-only
        model_idx = m-1;
        bins = discretize(abs(all_condiff), 0:0.01:0.1);
        [binnedPe, modelUp, modelSEM] = deal(NaN(10,1));
        
        for b = 1:10
            binnedPe(b) = nanmean(all_condiff(bins == b, 1));
            modelUp(b) = nanmean(sim_lr_means(bins == b, model_idx));
            modelSEM(b) = nanstd(sim_lr_means(bins == b, model_idx))/sqrt(sum(bins == b));
        end
        
        errorbar(bin_centers.', modelUp, modelSEM, 's', ...
            'Color', colors(m,:), 'LineWidth', 1, 'MarkerSize', markerSizeModels, ...
            'MarkerFaceColor', colors(m,:), 'DisplayName', model_names{model_idx});
        hold on
        title(sprintf(string(model_names(m-1))));  % Subplot 1 = "Model 0" (Actual)
    end

    xlabel('Contrast difference');
    ylabel('Signed LR');
    % xlim([-1 1]);
    grid on;
    
    if m == 1
        legend('Location', 'best');
    end
end

%% PLOT 3 & 4: LEARNING CURVES

unique_conditions = unique(all_condition(~isnan(all_condition)));
n_conditions = min(2, length(unique_conditions));

figure
hold on
% pwrl_mu = sim_mu_means(:,3);
% for h = 1:height(sim_mu_means)
%     if all_contrast(h) == 1
%         pwrl_mu(h,1) = 1-sim_mu_means(h,3);
%     end
% end
% sim_mu_means(:,3) = pwrl_mu;

for cond_idx = 1:n_conditions
    subplot(1, 2, cond_idx);
    hold on;
    condition_val = unique_conditions(cond_idx);
    
    max_trials = 25;
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
    
    % Calculate simulated curves
    for m = 1:nModels
        for trial = 1:max_trials
            mask = (all_condition == condition_val) & (all_trial_in_block == trial);
            if sum(mask) > 0
                sim_curves(trial, m) = mean(sim_mu_means(mask, m));
                sim_curves_se(trial, m) = std(sim_mu_means(mask, m)) / sqrt(sum(mask));
            end
        end
    end
    
    % Plot actual data with shaded error
    x = 1:25;
    % Plot actual data with shaded error (hidden from legend)
    fill([x fliplr(x)], [actual_curve'+actual_curve_se' fliplr(actual_curve'-actual_curve_se')], ...
        colors(1,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, actual_curve, 'Color', colors(1,:), 'LineWidth', 2, 'DisplayName', 'Actual μ');

    % Plot simulated curves with shaded error (hidden from legend)
    for m = 1:nModels
        fill([x fliplr(x)], [sim_curves(:,m)'+sim_curves_se(:,m)' fliplr(sim_curves(:,m)'-sim_curves_se(:,m)')], ...
            colors(m+1,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(x, sim_curves(:,m), 'Color', colors(m+1,:), 'LineWidth', 2, ...
            'DisplayName', [model_names{m} ' μ']);
    end

    xlabel('Trial');
    ylabel('Learned contingency parameter (mu)');
    title(sprintf('Learning Curves - Condition %d', condition_val));
    legend('Location', 'best');
    % grid on;
    xlim([1 max_trials]);
    ylim([0.4,1])
end

%%

unique_conditions = unique(all_condition(~isnan(all_condition)));
n_conditions = min(2, length(unique_conditions));

% Define true mu values for each condition
true_mu_values = [0.7, 0.9]; % condition 1 = 0.7, condition 2 = 0.9

% Create figure with 2 rows: top for mu curves, bottom for estimation error curves
figure('Position', [100, 100, 1200, 800]);

% Define colors
colors = [[0.5,0.5,0.5]; lines(nModels)];

for cond_idx = 1:n_conditions
    condition_val = unique_conditions(cond_idx);
    true_mu = true_mu_values(cond_idx);
    max_trials = 25;
    
    % Initialize arrays
    actual_curve = zeros(max_trials, 1);
    actual_curve_se = zeros(max_trials, 1);
    actual_error_curve = zeros(max_trials, 1);
    actual_error_se = zeros(max_trials, 1);
    
    sim_curves = zeros(max_trials, nModels);
    sim_curves_se = zeros(max_trials, nModels);
    sim_error_curves = zeros(max_trials, nModels);
    sim_error_se = zeros(max_trials, nModels);
    
    % Calculate actual learning and error curves
    for trial = 1:max_trials
        mask = (all_condition == condition_val) & (all_trial_in_block == trial);
        if sum(mask) > 0
            % Mu curves
            actual_curve(trial) = mean(all_mu_actual(mask));
            actual_curve_se(trial) = std(all_mu_actual(mask)) / sqrt(sum(mask));
            
            % Error curves
            actual_error_curve(trial) = mean(all_absEE(mask));
            actual_error_se(trial) = std(all_absEE(mask)) / sqrt(sum(mask));

            % actual_errors = abs(true_mu - all_mu_actual(mask));
            % actual_error_curve(trial) = mean(actual_errors);
            % actual_error_se(trial) = std(actual_errors) / sqrt(sum(mask));
        end
    end
    
    % Calculate simulated curves
    for m = 1:nModels
        for trial = 1:max_trials
            mask = (all_condition == condition_val) & (all_trial_in_block == trial);
            if sum(mask) > 0
                % Mu curves
                sim_curves(trial, m) = mean(sim_mu_means(mask, m));
                sim_curves_se(trial, m) = std(sim_mu_means(mask, m)) / sqrt(sum(mask));
                
                % Error curves
                sim_errors = abs(true_mu - sim_mu_means(mask, m));
                sim_error_curves(trial, m) = mean(sim_errors);
                sim_error_se(trial, m) = std(sim_errors) / sqrt(sum(mask));
            end
        end
    end
    
    x = 1:25;
    
    % SUBPLOT 1: Mu Learning Curves (Top Row)
    subplot(2, 2, cond_idx);
    hold on;
    
    % Plot actual mu with shaded error
    fill([x fliplr(x)], [actual_curve'+actual_curve_se' fliplr(actual_curve'-actual_curve_se')], ...
         colors(1,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, actual_curve, 'Color', colors(1,:), 'LineWidth', 2, 'DisplayName', 'Actual μ');
    
    % Plot simulated mu curves with shaded error
    for m = 1:nModels
        fill([x fliplr(x)], [sim_curves(:,m)'+sim_curves_se(:,m)' fliplr(sim_curves(:,m)'-sim_curves_se(:,m)')], ...
             colors(m+1,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(x, sim_curves(:,m), 'Color', colors(m+1,:), 'LineWidth', 2, ...
             'DisplayName', [model_names{m} ' μ']);
    end
    
    % Add horizontal line for true mu value
    yline(true_mu, '--k', 'LineWidth', 1.5, 'DisplayName', sprintf('True μ = %.1f', true_mu));
    
    xlabel('Trial');
    ylabel('Learned contingency parameter (μ)');
    title(sprintf('Learning Curves - Condition %d (True μ = %.1f)', condition_val, true_mu));
    legend('Location', 'best');
    xlim([1 max_trials]);
    ylim([0.5, 1]);
    grid on;
    
    % SUBPLOT 2: Estimation Error Curves (Bottom Row)
    subplot(2, 2, cond_idx + 2);
    hold on;
    
    % Plot actual estimation error with shaded error
    fill([x fliplr(x)], [actual_error_curve'+actual_error_se' fliplr(actual_error_curve'-actual_error_se')], ...
         colors(1,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, actual_error_curve, 'Color', colors(1,:), 'LineWidth', 2, 'DisplayName', 'Actual Error');
    
    % Plot simulated estimation error curves with shaded error
    for m = 1:nModels
        fill([x fliplr(x)], [sim_error_curves(:,m)'+sim_error_se(:,m)' fliplr(sim_error_curves(:,m)'-sim_error_se(:,m)')], ...
             colors(m+1,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(x, sim_error_curves(:,m), 'Color', colors(m+1,:), 'LineWidth', 2, ...
             'DisplayName', [model_names{m} ' Error']);
    end
    
    xlabel('Trial');
    ylabel('Estimation Error |True μ - Estimated μ|');
    title(sprintf('Estimation Error - Condition %d (True μ = %.1f)', condition_val, true_mu));
    legend('Location', 'best');
    xlim([1 max_trials]);
    ylim([0, inf]); % Let MATLAB auto-scale the upper limit
    grid on;
end

% Add overall title
sgtitle('Learning Curves and Estimation Error Across Conditions', 'FontSize', 16, 'FontWeight', 'bold');

%%

%% =================== DIAGNOSTIC ANALYSIS =========================
% Investigate why mu curves are similar but error curves differ

unique_conditions = unique(all_condition(~isnan(all_condition)));
true_mu_values = [0.7, 0.9];
colors = [[0.5,0.5,0.5]; lines(nModels)];

figure('Position', [100, 100, 1400, 1000]);

for cond_idx = 1:2
    condition_val = unique_conditions(cond_idx);
    true_mu = true_mu_values(cond_idx);
    
    % Focus on a specific trial range for detailed analysis
    analysis_trials = [10, 20]; % Mid and late learning
    
    for trial_idx = 1:2
        trial = analysis_trials(trial_idx);
        subplot_idx = (cond_idx-1)*4 + (trial_idx-1)*2 + 1;
        
        % Get data for this trial
        mask = (all_condition == condition_val) & (all_trial_in_block == trial);
        
        if sum(mask) > 0
            % Extract actual data
            actual_mu_vals = all_mu_actual(mask);
            
            % SUBPLOT 1: Distribution comparison
            subplot(4, 2, subplot_idx);
            hold on;
            
            % Plot distributions
            edges = 0.4:0.05:1.0;
            
            % Actual data histogram
            [counts_actual, ~] = histcounts(actual_mu_vals, edges);
            bar_centers = edges(1:end-1) + diff(edges)/2;
            bar(bar_centers, counts_actual/sum(counts_actual), 'FaceColor', colors(1,:), ...
                'FaceAlpha', 0.6, 'DisplayName', 'Actual');
            
            % Model distributions
            for m = 1:nModels
                model_mu_vals = sim_mu_means(mask, m);
                [counts_model, ~] = histcounts(model_mu_vals, edges);
                plot(bar_centers, counts_model/sum(counts_model), 'o-', ...
                     'Color', colors(m+1,:), 'LineWidth', 2, ...
                     'DisplayName', model_names{m});
            end
            
            % Add true value line
            xline(true_mu, '--k', 'LineWidth', 2, 'DisplayName', 'True μ');
            
            title(sprintf('Cond %d, Trial %d - μ Distributions', condition_val, trial));
            xlabel('μ value');
            ylabel('Probability');
            legend('Location', 'best');
            grid on;
            
            % SUBPLOT 2: Variance and bias analysis
            subplot(4, 2, subplot_idx + 1);
            
            % Calculate metrics
            metrics = [];
            labels = {'Actual'};
            
            % Actual data metrics
            actual_mean = mean(actual_mu_vals);
            actual_var = var(actual_mu_vals);
            actual_bias = actual_mean - true_mu;
            actual_rmse = sqrt(mean((actual_mu_vals - true_mu).^2));
            metrics = [metrics; actual_mean, actual_var, actual_bias, actual_rmse];
            
            % Model metrics
            for m = 1:nModels
                model_mu_vals = sim_mu_means(mask, m);
                model_mean = mean(model_mu_vals);
                model_var = var(model_mu_vals);
                model_bias = model_mean - true_mu;
                model_rmse = sqrt(mean((model_mu_vals - true_mu).^2));
                metrics = [metrics; model_mean, model_var, model_bias, model_rmse];
                labels{end+1} = model_names{m};
            end
            
            % Plot metrics
            metric_names = {'Mean', 'Variance', 'Bias', 'RMSE'};
            x_pos = 1:4;
            
            for i = 1:size(metrics, 1)
                plot(x_pos, metrics(i,:), 'o-', 'Color', colors(i,:), ...
                     'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', labels{i});
                hold on;
            end
            
            % Add reference lines
            yline(0, '--k', 'Alpha', 0.5);
            if any(strcmp(metric_names, 'Mean'))
                yline(true_mu, '--r', 'Alpha', 0.5);
            end
            
            title(sprintf('Cond %d, Trial %d - Performance Metrics', condition_val, trial));
            xticks(x_pos);
            xticklabels(metric_names);
            ylabel('Value');
            legend('Location', 'best');
            grid on;
        end
    end
end

sgtitle('Detailed Analysis: Why Similar μ Curves Have Different Error Patterns', ...
        'FontSize', 14, 'FontWeight', 'bold');

%% =================== QUANTITATIVE COMPARISON =========================

fprintf('\n=================== DETAILED PERFORMANCE ANALYSIS ===================\n');

for cond_idx = 1:2
    condition_val = unique_conditions(cond_idx);
    true_mu = true_mu_values(cond_idx);
    
    fprintf('\nCondition %d (True μ = %.1f):\n', condition_val, true_mu);
    fprintf('%-15s %-10s %-10s %-10s %-10s %-10s\n', 'Model', 'Mean μ', 'Bias', 'Variance', 'RMSE', 'Mean |Error|');
    fprintf('%s\n', repmat('-', 1, 75));
    
    mask = (all_condition == condition_val);
    
    % Actual data
    actual_mu_vals = all_mu_actual(mask);
    actual_mean = mean(actual_mu_vals);
    actual_bias = actual_mean - true_mu;
    actual_var = var(actual_mu_vals);
    actual_rmse = sqrt(mean((actual_mu_vals - true_mu).^2));
    actual_mae = mean(abs(actual_mu_vals - true_mu));
    
    fprintf('%-15s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n', ...
            'Actual', actual_mean, actual_bias, actual_var, actual_rmse, actual_mae);
    
    % Model data
    for m = 1:nModels
        model_mu_vals = sim_mu_means(mask, m);
        model_mean = mean(model_mu_vals);
        model_bias = model_mean - true_mu;
        model_var = var(model_mu_vals);
        model_rmse = sqrt(mean((model_mu_vals - true_mu).^2));
        model_mae = mean(abs(model_mu_vals - true_mu));
        
        fprintf('%-15s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n', ...
                model_names{m}, model_mean, model_bias, model_var, model_rmse, model_mae);
    end
end

%% =================== ERROR DECOMPOSITION =========================
% Decompose error into bias and variance components

fprintf('\n=================== ERROR DECOMPOSITION ===================\n');
fprintf('Mean Squared Error = Bias² + Variance\n');
fprintf('Mean Absolute Error ≈ √(Bias² + Variance) for normal distributions\n\n');

for cond_idx = 1:2
    condition_val = unique_conditions(cond_idx);
    true_mu = true_mu_values(cond_idx);
    
    fprintf('Condition %d:\n', condition_val);
    fprintf('%-15s %-10s %-10s %-10s %-15s\n', 'Model', 'Bias²', 'Variance', 'MSE', 'Bias²+Var');
    fprintf('%s\n', repmat('-', 1, 65));
    
    mask = (all_condition == condition_val);
    
    % Calculate for actual and all models
    all_models = [{'Actual'}, model_names];
    for i = 1:length(all_models)
        if i == 1
            mu_vals = all_mu_actual(mask);
        else
            mu_vals = sim_mu_means(mask, i-1);
        end
        
        bias = mean(mu_vals) - true_mu;
        variance = var(mu_vals);
        mse = mean((mu_vals - true_mu).^2);
        bias_squared = bias^2;
        
        fprintf('%-15s %-10.4f %-10.4f %-10.4f %-15.4f\n', ...
                all_models{i}, bias_squared, variance, mse, bias_squared + variance);
    end
    fprintf('\n');
end

%% =================== TRIAL-BY-TRIAL CORRELATION =========================
% Check if the models track human trial-by-trial variability

fprintf('=================== TRIAL-BY-TRIAL CORRELATIONS ===================\n');

for cond_idx = 1:2
    condition_val = unique_conditions(cond_idx);
    fprintf('\nCondition %d - Correlation with actual μ values:\n', condition_val);
    
    mask = (all_condition == condition_val);
    actual_vals = all_mu_actual(mask);
    
    for m = 1:nModels
        model_vals = sim_mu_means(mask, m);
        correlation = corr(actual_vals, model_vals, 'rows', 'complete');
        fprintf('%s: r = %.4f\n', model_names{m}, correlation);
    end
end

%% =================== SUMMARY STATISTICS ===================
% Calculate and display final performance metrics

fprintf('\n=================== FINAL PERFORMANCE SUMMARY ===================\n');
fprintf('%-15s %-10s %-15s %-15s %-15s %-15s\n', 'Condition', 'True μ', 'Data Type', 'Final μ', 'Final Error', 'Mean Error');
fprintf('%s\n', repmat('-', 1, 90));

for cond_idx = 1:n_conditions
    condition_val = unique_conditions(cond_idx);
    true_mu = true_mu_values(cond_idx);
    
    % Get final trials (last 5 trials for stability)
    final_trials = 21:25;
    
    % Actual data
    mask_final = ismember(all_trial_in_block, final_trials) & (all_condition == condition_val);
    if sum(mask_final) > 0
        final_mu_actual = mean(all_mu_actual(mask_final));
        final_error_actual = abs(true_mu - final_mu_actual);
        mean_error_actual = mean(abs(true_mu - all_mu_actual(all_condition == condition_val)));
        
        fprintf('%-15d %-10.1f %-15s %-15.4f %-15.4f %-15.4f\n', ...
                condition_val, true_mu, 'Actual', final_mu_actual, final_error_actual, mean_error_actual);
    end
    
    % Simulated data for each model
    for m = 1:nModels
        if sum(mask_final) > 0
            final_mu_sim = mean(sim_mu_means(mask_final, m));
            final_error_sim = abs(true_mu - final_mu_sim);
            mean_error_sim = mean(abs(true_mu - sim_mu_means(all_condition == condition_val, m)));
            
            fprintf('%-15d %-10.1f %-15s %-15.4f %-15.4f %-15.4f\n', ...
                    condition_val, true_mu, model_names{m}, final_mu_sim, final_error_sim, mean_error_sim);
        end
    end
    fprintf('\n');
end

%% =================== ADDITIONAL ANALYSIS ===================
% Compare learning rates between conditions and models

fprintf('=================== LEARNING RATE ANALYSIS ===================\n');
fprintf('Average absolute estimation error by condition and model:\n\n');

for cond_idx = 1:n_conditions
    condition_val = unique_conditions(cond_idx);
    true_mu = true_mu_values(cond_idx);
    
    fprintf('Condition %d (True μ = %.1f):\n', condition_val, true_mu);
    
    % Actual data
    cond_mask = (all_condition == condition_val);
    actual_errors = abs(true_mu - all_mu_actual(cond_mask));
    fprintf('  Actual: Mean Error = %.4f ± %.4f\n', mean(actual_errors), std(actual_errors));
    
    % Model data
    for m = 1:nModels
        sim_errors = abs(true_mu - sim_mu_means(cond_mask, m));
        fprintf('  %s: Mean Error = %.4f ± %.4f\n', model_names{m}, mean(sim_errors), std(sim_errors));
    end
    fprintf('\n');
end

%%
unique_conditions = unique(all_condition(~isnan(all_condition)));
n_conditions = min(2, length(unique_conditions));

for cond_idx = 1:n_conditions
    subplot(2, 2, 2 + cond_idx);
    hold on;
    condition_val = unique_conditions(cond_idx);
    
    max_trials = 40;
    actual_curve = zeros(max_trials, 1);
    actual_curve_se = zeros(max_trials, 1);
    sim_curves = zeros(max_trials, nModels);
    sim_curves_se = zeros(max_trials, nModels);
    
    % Calculate actual learning curve (using mu_hat from simulations)
    for trial = 1:max_trials
        mask = (all_condition == condition_val) & (all_trial_in_block == trial);
        if sum(mask) > 0
            actual_curve(trial) = mean(all_mu_actual(mask));
            actual_curve_se(trial) = std(all_mu_actual(mask)) / sqrt(sum(mask));
        end
    end
    
    % Calculate simulated learning curves using predicted mu_hat
    for m = 1:nModels
        for trial = 1:max_trials
            mask = (all_condition == condition_val) & (all_trial_in_block == trial);
            if sum(mask) > 0
                sim_curves(trial, m) = mean(sim_mu_means(mask, m));
                sim_curves_se(trial, m) = mean(sim_mu_ses(mask, m));
            end
        end
    end
    
    % Plot actual learning curve
    errorbar(1:max_trials, actual_curve, actual_curve_se, 'o-', ...
        'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 4, 'DisplayName', 'Actual μ');
    
    % Plot simulated learning curves
    for m = 1:nModels
        errorbar(1:max_trials, sim_curves(:, m), sim_curves_se(:, m), 's-', ...
            'Color', colors(m+1,:), 'LineWidth', 1.5, 'MarkerSize', 3, ...
            'DisplayName', [model_names{m} ' μ']);
    end
    
    xlabel('Trial in Block');
    ylabel('Belief State (μ)');
    title(sprintf('Learning Curves - Condition %d', condition_val));
    legend('Location', 'best');
    grid on;
end

%% =================== FIGURE 2: ABSOLUTE VALUE PLOTS ==================
figure('Position', figure_size);
sgtitle('Absolute Value Analysis', 'FontSize', 16, 'FontWeight', 'bold');

% Calculate learning rates (avoiding division by zero)
all_actual_lr = zeros(size(all_actual_up));
sim_lr = zeros(total_trials, nModels);

% Calculate actual learning rates
valid_pe_mask = abs(all_pe) > 1e-6; % Avoid division by very small numbers
all_actual_lr(valid_pe_mask) = all_actual_up(valid_pe_mask) ./ all_pe(valid_pe_mask);

% Calculate simulated learning rates
for m = 1:nModels
    sim_lr(valid_pe_mask, m) = sim_means(valid_pe_mask, m) ./ all_pe(valid_pe_mask);
end

%% PLOT 1: |PE| vs |UPDATE|
subplot(2, 3, 1);
hold on;

abs_pe = abs(all_pe);
abs_actual_up = abs(all_actual_up);

[bin_centers_abs, actual_binned_abs, actual_se_abs] = create_binned_data(abs_pe, abs_actual_up, 20);

% errorbar(bin_centers_abs, actual_binned_abs, actual_se_abs, 'o-', ...
%     'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Actual');

for m = 1:nModels
    abs_sim_up = abs(sim_means(:, m));
    [~, sim_binned_abs, sim_se_abs] = create_binned_data(abs_pe, abs_sim_up, 20);
    errorbar(bin_centers_abs, sim_binned_abs, sim_se_abs, 's-', ...
        'Color', colors(m+1,:), 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', model_names{m});
end

xlabel('|Prediction Error|');
ylabel('|Update|');
title('|PE| vs |Update|');
legend('Location', 'best');
grid on;

%% PLOT 2: |PE| vs |LR|
subplot(2, 3, 2);
hold on;

abs_actual_lr = abs(all_actual_lr);
valid_lr_mask = valid_pe_mask & isfinite(abs_actual_lr);

if sum(valid_lr_mask) > 0
    [bin_centers_lr, actual_binned_lr, actual_se_lr] = create_binned_data(abs_pe(valid_lr_mask), abs_actual_lr(valid_lr_mask), 15);
    
    errorbar(bin_centers_lr, actual_binned_lr, actual_se_lr, 'o-', ...
        'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Actual');
    
    for m = 1:nModels
        abs_sim_lr = abs(sim_lr(:, m));
        valid_sim_lr = valid_pe_mask & isfinite(abs_sim_lr);
        if sum(valid_sim_lr) > 0
            [~, sim_binned_lr, sim_se_lr] = create_binned_data(abs_pe(valid_sim_lr), abs_sim_lr(valid_sim_lr), 15);
            errorbar(bin_centers_lr, sim_binned_lr, sim_se_lr, 's-', ...
                'Color', colors(m+1,:), 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', model_names{m});
        end
    end
end

xlabel('|Prediction Error|');
ylabel('|Learning Rate|');
title('|PE| vs |LR|');
legend('Location', 'best');
grid on;

%% PLOT 3: |PE| vs LR
subplot(2, 3, 3);
hold on;

if sum(valid_pe_mask) > 0
    [bin_centers_lr2, actual_binned_lr2, actual_se_lr2] = create_binned_data(abs_pe(valid_pe_mask), all_actual_lr(valid_pe_mask), 15);
    
    errorbar(bin_centers_lr2, actual_binned_lr2, actual_se_lr2, 'o-', ...
        'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Actual');
    
    for m = 1:nModels
        valid_sim_lr2 = valid_pe_mask & isfinite(sim_lr(:, m));
        if sum(valid_sim_lr2) > 0
            [~, sim_binned_lr2, sim_se_lr2] = create_binned_data(abs_pe(valid_sim_lr2), sim_lr(valid_sim_lr2, m), 15);
            errorbar(bin_centers_lr2, sim_binned_lr2, sim_se_lr2, 's-', ...
                'Color', colors(m+1,:), 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', model_names{m});
        end
    end
end

xlabel('|Prediction Error|');
ylabel('Learning Rate');
title('|PE| vs LR');
legend('Location', 'best');
grid on;

%% PLOT 4: |CONDIFF| vs |UPDATE|
subplot(2, 3, 4);
hold on;

abs_condiff = abs(all_condiff);

[bin_centers_condiff_abs, actual_binned_condiff_abs, actual_se_condiff_abs] = create_binned_data(abs_condiff, abs_actual_up, 15);

errorbar(bin_centers_condiff_abs, actual_binned_condiff_abs, actual_se_condiff_abs, 'o-', ...
    'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Actual');

for m = 1:nModels
    abs_sim_up = abs(sim_means(:, m));
    [~, sim_binned_condiff_abs, sim_se_condiff_abs] = create_binned_data(abs_condiff, abs_sim_up, 15);
    errorbar(bin_centers_condiff_abs, sim_binned_condiff_abs, sim_se_condiff_abs, 's-', ...
        'Color', colors(m+1,:), 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', model_names{m});
end

xlabel('|Contrast Difference|');
ylabel('|Update|');
title('|ConDiff| vs |Update|');
legend('Location', 'best');
grid on;

%% PLOT 5: |CONDIFF| vs |LR|
subplot(2, 3, 5);
hold on;

if sum(valid_lr_mask) > 0
    [bin_centers_condiff_lr, actual_binned_condiff_lr, actual_se_condiff_lr] = create_binned_data(abs_condiff(valid_lr_mask), abs_actual_lr(valid_lr_mask), 10);
    
    errorbar(bin_centers_condiff_lr, actual_binned_condiff_lr, actual_se_condiff_lr, 'o-', ...
        'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Actual');
    
    for m = 1:nModels
        abs_sim_lr = abs(sim_lr(:, m));
        valid_sim_lr = valid_pe_mask & isfinite(abs_sim_lr);
        if sum(valid_sim_lr) > 0
            [~, sim_binned_condiff_lr, sim_se_condiff_lr] = create_binned_data(abs_condiff(valid_sim_lr), abs_sim_lr(valid_sim_lr), 10);
            errorbar(bin_centers_condiff_lr, sim_binned_condiff_lr, sim_se_condiff_lr, 's-', ...
                'Color', colors(m+1,:), 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', model_names{m});
        end
    end
end

xlabel('|Contrast Difference|');
ylabel('|Learning Rate|');
title('|ConDiff| vs |LR|');
legend('Location', 'best');
grid on;

%% PLOT 6: |CONDIFF| vs LR
subplot(2, 3, 6);
hold on;

if sum(valid_pe_mask) > 0
    [bin_centers_condiff_lr2, actual_binned_condiff_lr2, actual_se_condiff_lr2] = create_binned_data(abs_condiff(valid_pe_mask), all_actual_lr(valid_pe_mask), 10);
    
    errorbar(bin_centers_condiff_lr2, actual_binned_condiff_lr2, actual_se_condiff_lr2, 'o-', ...
        'Color', colors(1,:), 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Actual');
    
    for m = 1:nModels
        valid_sim_lr2 = valid_pe_mask & isfinite(sim_lr(:, m));
        if sum(valid_sim_lr2) > 0
            [~, sim_binned_condiff_lr2, sim_se_condiff_lr2] = create_binned_data(abs_condiff(valid_sim_lr2), sim_lr(valid_sim_lr2, m), 10);
            errorbar(bin_centers_condiff_lr2, sim_binned_condiff_lr2, sim_se_condiff_lr2, 's-', ...
                'Color', colors(m+1,:), 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', model_names{m});
        end
    end
end

xlabel('|Contrast Difference|');
ylabel('Learning Rate');
title('|ConDiff| vs LR');
legend('Location', 'best');
grid on;

%% =================== STATISTICAL SUMMARY ==========================
fprintf('\n=== ENHANCED SIMULATION ANALYSIS SUMMARY ===\n');
fprintf('Number of subjects: %d\n', length(analysis_data));
fprintf('Total trials analyzed: %d\n', length(all_pe));
fprintf('Models compared: %s\n', strjoin(model_names, ', '));

% Correlation analysis
fprintf('\n=== CORRELATIONS WITH ACTUAL DATA ===\n');
actual_pe_corr = corr(all_pe, all_actual_up, 'rows', 'complete');
actual_condiff_corr = corr(all_condiff, all_actual_up, 'rows', 'complete');

fprintf('Actual PE vs Updates: r = %.3f\n', actual_pe_corr);
fprintf('Actual ConDiff vs Updates: r = %.3f\n', actual_condiff_corr);

% Learning rate correlations
if sum(valid_pe_mask) > 0
    actual_pe_lr_corr = corr(abs_pe(valid_pe_mask), abs_actual_lr(valid_pe_mask), 'rows', 'complete');
    fprintf('Actual |PE| vs |LR|: r = %.3f\n', actual_pe_lr_corr);
end

fprintf('\n=== SIMULATED CORRELATIONS ===\n');
for m = 1:nModels
    pe_corr = corr(all_pe, sim_means(:, m), 'rows', 'complete');
    condiff_corr = corr(all_condiff, sim_means(:, m), 'rows', 'complete');
    
    % Absolute correlations
    abs_pe_corr = corr(abs_pe, abs(sim_means(:, m)), 'rows', 'complete');
    abs_condiff_corr = corr(abs_condiff, abs(sim_means(:, m)), 'rows', 'complete');
    
    fprintf('%s:\n', model_names{m});
    fprintf('  PE vs Updates: r = %.3f\n', pe_corr);
    fprintf('  ConDiff vs Updates: r = %.3f\n', condiff_corr);
    fprintf('  |PE| vs |Updates|: r = %.3f\n', abs_pe_corr);
    fprintf('  |ConDiff| vs |Updates|: r = %.3f\n', abs_condiff_corr);
end

% Model fit comparison (RMSE)
fprintf('\n=== MODEL FIT COMPARISON ===\n');
fprintf('Update Prediction RMSE:\n');
for m = 1:nModels
    rmse_updates = sqrt(mean((all_actual_up - sim_means(:, m)).^2, 'omitnan'));
    fprintf('  %s: %.4f\n', model_names{m}, rmse_updates);
end

fprintf('\nBelief State Prediction RMSE:\n');
for m = 1:nModels
    rmse_mu = sqrt(mean((all_mu_actual - sim_mu_means(:, m)).^2, 'omitnan'));
    fprintf('  %s: %.4f\n', model_names{m}, rmse_mu);
end

% Learning rate analysis
fprintf('\n=== LEARNING RATE ANALYSIS ===\n');
valid_trials = sum(valid_pe_mask);
