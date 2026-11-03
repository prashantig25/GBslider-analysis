clc; clearvars; close all;

%% ================== Load Data and Parameters ==================

% Load model parameters
params_basicRL       = importdata("params_basicRL.mat");
params_RLsigma       = importdata("params_RLsigma.mat");
params_PWRL          = importdata("params_PWRL.mat");
params_bayesianAgent = importdata("params_bayesianAgent.mat");

% Load preprocessed data
data = importdata("preprocessed_dataFitting.mat");
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;
data = data(data.choice_cond ~= 3,:);
% data = data(data.pe ~= 0, :);
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

%% ================ Simulation Setup ============================
model_names = {'basicRL','RL+EstSens','PWRL','BayesianAgent'};
nModels = numel(model_names);
nSimulations = 5;

% Estimate maximum number of trials per subject
maxTrials = max(histcounts(data.ID, [uniqueID; max(uniqueID)+1]));

% Initialize storage for all model simulations
% simulated_model_updates{model} = 1 x numSubjs cell array
% Each cell = [numTrials x nSimulations]
simulated_model_updates = cell(1, nModels);
actual_model_updates = cell(1, nModels);

for m = 1:nModels
    simulated_model_updates{m} = cell(1, numSubjs);
    actual_model_updates{m} = cell(1, numSubjs);
    for subjIdx = 1:numSubjs
        simulated_model_updates{m}{subjIdx} = NaN(maxTrials, nSimulations);
        actual_model_updates{m}{subjIdx} = NaN(maxTrials, 1);
    end
end

%% ================= Simulate for Each Subject ==================

for subjIdx = 1:numSubjs
    subjID = uniqueID(subjIdx);

    % Filter subject data
    subj_data = data(data.ID == subjID, :);

    cond = importdata("preprocessed_dataFitting.mat");
    cond = cond(cond.ID == uniqueID(subjIdx),:);
    cond = cond(cond.choice_cond ~= 3, :);
    condition = cond.condition(cond.trials./1 == 1);
    subj = preprocess_fitSlider(data, subjID);
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));

    T = height(subj_data);

    % Simulate each model
    for m = 1:nModels
        for s = 1:nSimulations
            switch m
                case 1  % basicRL
                    [~, sim_mu] = predict_allModels.predict_basicRL( ...
                        [params_basicRL.alpha(subjIdx), params_basicRL.kappa(subjIdx)], ...
                        subj.blocks, rewards, subj.state, 'sample');
                    % simulated_model_updates{m}{subjIdx}(1:T, s) = sim_mu;
                case 2  % RL+EstSens
                    [~, sim_mu] = predict_allModels.predict_RLsigma( ...
                        [params_RLsigma.alpha(subjIdx), params_RLsigma.kappa(subjIdx), params_RLsigma.sigma(subjIdx)], ...
                        subj.mu_hat, subj.blocks, rewards, subj.condiff, 'sample');
                    % simulated_model_updates{m}{subjIdx}(1:T, s) = sim_mu;
                case 3  % PWRL
                    [~, sim_mu] = predict_allModels.predict_PWRL( ...
                        [params_RLsigma.alpha(subjIdx), params_RLsigma.kappa(subjIdx), params_RLsigma.sigma(subjIdx)], ...
                        subj.blocks, rewards, subj.condiff, subj.choices, ...
                        subj.recoded_rewards, subj.state, 'sample');
                    % simulated_model_updates{m}{subjIdx}(1:T, s) = sim_mu;
                case 4  % BayesianAgent
                    
                    [~, sim_mu] = predict_allModels.predict_bayesianAgent( ...
                    [params_bayesianAgent.kappa(subjIdx), params_bayesianAgent.sigma(subjIdx)], ...
                    subj.blocks, subj.rewards, subj.condiff, subj.choices, condition, 'sample');
                    % simulated_model_updates{m}{subjIdx}(1:T, s) = sim_mu;
            end

            updates = [NaN; diff(sim_mu)];
            T = length(updates);

            % Store updates in the cell array (trials x simulations)
            simulated_model_updates{m}{subjIdx}(1:T, s) = updates;
        end
    end

    % actual_model_updates{m}{subjIdx}(1:T, s) = subj.dataTable.mu_congruence;
end

%% ==================== Aggregate & Plot ==============================

figure;
for m = 1:nModels
    subplot(2,2,m)
    hold on
    meanSimulated = zeros(numSubjs,1);
    seSimulated = zeros(numSubjs,1);
    meanActual = zeros(numSubjs,1);
    
    % Store all data for cross-participant binning
    all_actual = [];
    all_sim = [];
    subj_actual_data = cell(numSubjs, 1);
    subj_sim_data = cell(numSubjs, 1);
    
    % First pass: collect all data
    for subjIdx = 1:numSubjs
        simUpdates = simulated_model_updates{m}{subjIdx}; % trials x simulations
        simMeanPerTrial = mean(simUpdates, 2, 'omitnan'); % mean across simulations per trial
        simMeanAllTrials = mean(simMeanPerTrial, 'omitnan'); % mean across trials
        meanSimulated(subjIdx) = simMeanAllTrials;
        
        % Calculate std error of the mean (across simulations, per trial, then averaged)
        simStdPerTrial = std(simUpdates, 0, 2, 'omitnan');
        simSEPerTrial = simStdPerTrial ./ sqrt(size(simUpdates, 2));
        seSimulated(subjIdx) = mean(simSEPerTrial, 'omitnan');
        
        % Actual update per subject - calculate from data:
        subjID = uniqueID(subjIdx);
        subj_data = data(data.ID == subjID, :);
        actual_mu_hat = subj_data.mu_congruence; % Assuming mu_hat is a column in your data
        actual_updates = [NaN; diff(actual_mu_hat)];
        meanActual(subjIdx) = mean(actual_updates, 'omitnan');
        
        % Remove NaN values
        valid_idx = ~isnan(actual_updates) & ~isnan(simMeanPerTrial);
        actual_clean = actual_updates(valid_idx);
        sim_clean = simMeanPerTrial(valid_idx);
        
        % Store for global binning
        all_actual = [all_actual; actual_clean];
        all_sim = [all_sim; sim_clean];
        subj_actual_data{subjIdx} = actual_clean;
        subj_sim_data{subjIdx} = sim_clean;
        
        % Individual subject plot (beautified)
        % if length(actual_clean) > 5 % Only plot if enough data points
        %     figure('Name', ['Subject ' num2str(subjIdx) ' - ' model_names{m}], 'Position', [100, 100, 800, 600]);
        % 
        %     % Calculate correlation and p-value
        %     [r, p] = corrcoef(actual_clean, sim_clean);
        %     r_val = r(1,2);
        %     p_val = p(1,2);
        % 
        %     % Define bins for this subject
        %     nBins = min(8, floor(length(actual_clean)/3));
        %     [~, edges] = histcounts(actual_clean, nBins);
        % 
        %     % Bin the data
        %     binned_sim = zeros(nBins, 1);
        %     binned_actual = zeros(nBins, 1);
        %     binned_se = zeros(nBins, 1);
        %     binned_n = zeros(nBins, 1);
        % 
        %     for binIdx = 1:nBins
        %         if binIdx == nBins
        %             bin_mask = actual_clean >= edges(binIdx) & actual_clean <= edges(binIdx+1);
        %         else
        %             bin_mask = actual_clean >= edges(binIdx) & actual_clean < edges(binIdx+1);
        %         end
        % 
        %         if sum(bin_mask) > 0
        %             binned_actual(binIdx) = mean(actual_clean(bin_mask));
        %             binned_sim(binIdx) = mean(sim_clean(bin_mask));
        %             binned_se(binIdx) = std(sim_clean(bin_mask)) / sqrt(sum(bin_mask));
        %             binned_n(binIdx) = sum(bin_mask);
        %         else
        %             binned_actual(binIdx) = NaN;
        %             binned_sim(binIdx) = NaN;
        %             binned_se(binIdx) = NaN;
        %             binned_n(binIdx) = 0;
        %         end
        %     end
        % 
        %     % Remove empty bins
        %     valid_bins = ~isnan(binned_actual) & binned_n > 0;
        %     binned_actual_clean = binned_actual(valid_bins);
        %     binned_sim_clean = binned_sim(valid_bins);
        %     binned_se_clean = binned_se(valid_bins);
        % 
        %     % Beautiful plotting
        %     hold on
        % 
        %     % Plot error bars in black
        %     errorbar(binned_actual_clean, binned_sim_clean, binned_se_clean, ...
        %             'Color', [0.2 0.2 0.2], 'LineWidth', 2, 'LineStyle', 'none', 'CapSize', 8);
        % 
        %     % Plot dots with nice colors and edges
        %     scatter(binned_actual_clean, binned_sim_clean, 120, [0.2 0.6 0.8], 'filled', ...
        %            'MarkerEdgeColor', [0.1 0.3 0.5]);
        % 
        %     % Add reference line y = x
        %     xlims = xlim;
        %     ylims = ylim;
        %     lim_range = [min([xlims(1), ylims(1)]), max([xlims(2), ylims(2)])];
        %     plot(lim_range, lim_range, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 2);
        % 
        %     % Beautify axes and labels
        %     xlabel('Actual Updates', 'FontSize', 14, 'FontWeight', 'bold');
        %     ylabel('Simulated Updates (Mean)', 'FontSize', 14, 'FontWeight', 'bold');
        %     title(sprintf('r = %.3f, p = %.3f', r_val, p_val), 'FontSize', 16, 'FontWeight', 'bold');
        % 
        %     grid on
        %     grid minor
        %     set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1);
        %     set(gca, 'FontSize', 12, 'LineWidth', 1.5);
        %     axis equal
        % 
        %     % Set background color
        %     set(gca, 'Color', [0.98 0.98 0.98]);
        % 
        %     hold off
        % end
    end
    
    % Cross-participant binning analysis (10 bins)
    if length(all_actual) > 30 % Only if enough total data
        % figure('Name', ['Cross-Participant Binned - ' model_names{m}], 'Position', [200, 200, 900, 700]);
        
        % Define 10 bins based on all actual data
        nBins = 10;
        [~, edges] = histcounts(all_actual, nBins);
        bin_centers = (edges(1:end-1) + edges(2:end)) / 2;
        
        % For each bin, collect mean predicted update for each participant
        cross_binned_sim = zeros(nBins, 1);
        cross_binned_actual = zeros(nBins, 1);
        cross_binned_sem = zeros(nBins, 1);
        cross_binned_n = zeros(nBins, 1);
        
        for binIdx = 1:nBins
            bin_participant_means = [];
            bin_actual_vals = [];
            
            for subjIdx = 1:numSubjs
                actual_clean = subj_actual_data{subjIdx};
                sim_clean = subj_sim_data{subjIdx};
                
                if ~isempty(actual_clean)
                    % Find data points in this bin for this participant
                    if binIdx == nBins
                        bin_mask = actual_clean >= edges(binIdx) & actual_clean <= edges(binIdx+1);
                    else
                        bin_mask = actual_clean >= edges(binIdx) & actual_clean < edges(binIdx+1);
                    end
                    
                    if sum(bin_mask) > 0
                        % Mean predicted update for this participant in this bin
                        participant_mean_sim = mean(sim_clean(bin_mask));
                        participant_mean_actual = mean(actual_clean(bin_mask));
                        bin_participant_means = [bin_participant_means; participant_mean_sim];
                        bin_actual_vals = [bin_actual_vals; participant_mean_actual];
                    end
                end
            end
            
            if length(bin_participant_means) > 0
                cross_binned_sim(binIdx) = mean(bin_participant_means);
                cross_binned_actual(binIdx) = mean(bin_actual_vals);
                cross_binned_sem(binIdx) = std(bin_participant_means) / sqrt(length(bin_participant_means));
                cross_binned_n(binIdx) = length(bin_participant_means);
            else
                cross_binned_sim(binIdx) = NaN;
                cross_binned_actual(binIdx) = NaN;
                cross_binned_sem(binIdx) = NaN;
                cross_binned_n(binIdx) = 0;
            end
        end
        
        % Remove empty bins
        valid_bins = ~isnan(cross_binned_actual) & cross_binned_n > 0;
        cross_binned_actual_clean = cross_binned_actual(valid_bins);
        cross_binned_sim_clean = cross_binned_sim(valid_bins);
        cross_binned_sem_clean = cross_binned_sem(valid_bins);
        
        % Calculate overall correlation
        [r_cross, p_cross] = corrcoef(cross_binned_actual_clean, cross_binned_sim_clean);
        r_val_cross = r_cross(1,2);
        p_val_cross = p_cross(1,2);
        
        % Beautiful cross-participant plot
        hold on
        
        % Plot error bars in black
        errorbar(cross_binned_actual_clean, cross_binned_sim_clean, cross_binned_sem_clean, ...
                'Color', [0.1 0.1 0.1], 'LineWidth', 1, 'LineStyle', 'none', 'CapSize', 10);
        
        % Plot dots with gradient colors and edges
        scatter(cross_binned_actual_clean, cross_binned_sim_clean, 30, [0.8 0.3 0.2], 'filled', ...
               'MarkerEdgeColor', [0.5 0.1 0.1]);
        
        % Add connecting line
        plot(cross_binned_actual_clean, cross_binned_sim_clean, '-', ...
             'Color', [0.6 0.6 0.6], 'LineWidth', 1);
        
        % Add reference line y = x
        % xlims = xlim;
        % ylims = ylim;
        % lim_range = [min([xlims(1), ylims(1)]), max([xlims(2), ylims(2)])];
        % plot(lim_range, lim_range, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 2.5);
        % 
        % Beautify axes and labels
        xlabel('Actual Updates (Binned)', 'FontSize', 16, 'FontWeight', 'bold');
        ylabel('Simulated Updates (Mean ± SEM)', 'FontSize', 16, 'FontWeight', 'bold');
        title(sprintf('Cross-Participant: r = %.3f, p = %.3f', r_val_cross, p_val_cross), ...
              'FontSize', 18, 'FontWeight', 'bold');
        
        % grid on
        % grid minor
        set(gca, 'GridAlpha', 0.4, 'MinorGridAlpha', 0.15);
        set(gca, 'FontSize', 7, 'LineWidth', 2);
        % axis equal
        
        % Set background color
        set(gca, 'Color', [0.97 0.97 0.99]);
        
        hold off
    end
end
    
    % Continue with your original summary plot
    figure(1) % Return to original figure
    subplot(2,2,m)
    scatter(meanActual, meanSimulated, 'o','SizeData',20)
    xlabel('Mean Actual Updates')
    ylabel('Mean Simulated Updates')
    title(model_names{m})
    grid on
    axis equal
    
    % Add reference line y = x
    xlims = xlim;
    ylims = ylim;
    lim_range = [min(xlims(1), ylims(1)), max(xlims(2), ylims(2))];
    plot(lim_range, lim_range, 'k--', 'LineWidth', 1);
    hold off


%% ======= MODEL PREDICTION FUNCTIONS (unchanged) ================

% BASIC RL
function predicted_updates = predict_updates_basicRL(params, blocks, rewards, state, mode)
if nargin < 5, mode = 'mean'; end
alpha = params(1); kappa = params(2);
nTrials = length(blocks); q_0_0 = 0.5; mu_hat = zeros(nTrials,1);
for n = 1:nTrials
    if n > 1 && blocks(n) ~= blocks(n-1), q_0_0 = 0.5; end
    if state(n) == 0
        q_0_0 = q_0_0 + alpha * (rewards(n) - q_0_0);
    else
        q_0_0 = q_0_0 + alpha * ((1 - rewards(n)) - q_0_0);
    end
    a = q_0_0 * kappa; b_beta = (1 - q_0_0) * kappa;
    switch lower(mode)
        case 'mean',   mu_hat(n) = a / (a + b_beta);
        case 'sample', mu_hat(n) = betarnd(a, b_beta);
        otherwise, error('Unknown mode. Use ''mean'' or ''sample''.');
    end
end
predicted_updates = [NaN; diff(mu_hat)];
end

% RL + Est. sensitivity
function predicted_updates = predict_updates_RLsigma(params, blocks, rewards, state, condiff, mode)
if nargin < 6, mode = 'mean'; end
alpha = params(1); kappa = params(2); sigma = params(3);
nTrials = length(blocks); q_0_0 = 0.5; mu_hat = zeros(nTrials,1);
for n = 1:nTrials
    belief_state = normcdf(condiff(n), 0, sigma);
    if condiff(n) < 0
        pi_0 = 1-belief_state;
        pi_1 = belief_state;
    else
        pi_1 = belief_state;
        pi_0 = 1-belief_state;
    end
    if n > 1 && blocks(n) ~= blocks(n-1), q_0_0 = 0.5; end
    if pi_0 >= pi_1
        q_0_0 = q_0_0 + pi_0 * alpha * (rewards(n) - q_0_0);
    else
        q_0_0 = q_0_0 + pi_1 * alpha * ((1 - rewards(n)) - q_0_0);
    end
    a = q_0_0 * kappa; b_beta = (1 - q_0_0) * kappa;
    switch lower(mode)
        case 'mean',   mu_hat(n) = a / (a + b_beta);
        case 'sample', mu_hat(n) = betarnd(a, b_beta);
        otherwise, error('Unknown mode. Use ''mean'' or ''sample''.');
    end
end
predicted_updates = [NaN; diff(mu_hat)];
end

function predicted_updates = predict_updates_bayesianAgent(params, blocks, rewards, state, condiff, choices, condition, mode)
kappa = params(1);
sigma = params(2);
G_vals = [];
uniqueBlocks = unique(blocks);
mu_hatAll = [];

for bl = 1:length(uniqueBlocks)
    agent = Agent();
    task = Task();
    agent.sigma = sigma;
    agent.condition = condition(bl);

    rewardsBlock = rewards(blocks == uniqueBlocks(bl));
    condiffBlock = condiff(blocks == uniqueBlocks(bl));

    for t = 1:height(rewardsBlock)
        agent.o_t = condiffBlock(t);
        agent.p_s_giv_o(agent.o_t);
        agent.a_t = choices(t);
        agent.learn(rewardsBlock(t));
        G_vals = [G_vals; agent.G];

        a = agent.G * kappa; b_beta = (1 - agent.G) * kappa;
        switch lower(mode)
            case 'mean',   mu_hat = a / (a + b_beta);
            case 'sample', mu_hat = betarnd(a, b_beta);
            otherwise, error('Unknown mode. Use ''mean'' or ''sample''.');
        end
        mu_hatAll = [mu_hatAll; mu_hat];
    end
end
predicted_updates = [NaN; diff(mu_hatAll)];
end

function predicted_updates = predict_updates_PWRL(params, blocks, rewards, condiff, choices, recoded_rewards, state, mode)
if nargin < 8, mode = 'mean'; end
alpha = params(1);
kappa = params(2);
sigma = params(3);
nTrials = length(blocks);

q_0_0 = 0.5;
q_1_0 = 1 - q_0_0;
q_1_1 = q_0_0;
q_0_1 = 1 - q_1_1;

mu_hat = zeros(nTrials,1);

for n = 1:nTrials
    if n > 1 && blocks(n) ~= blocks(n-1)
        q_0_0 = 0.5; q_1_0 = 0.5; q_1_1 = 0.5; q_0_1 = 0.5;
    end

    if condiff(n) < 0
        prediction_error = recoded_rewards(n) - q_0_0;
        q_0_0 = q_0_0 + alpha * prediction_error;
        q_1_0 = 1 - q_0_0;
        mu_hat(n) = q_0_0;
    else
        prediction_error = recoded_rewards(n) - q_1_1;
        q_1_1 = q_1_1 + alpha * prediction_error;
        q_0_1 = 1 - q_1_1;
        mu_hat(n) = q_1_1;
    end

    a = mu_hat(n) * kappa;
    b_beta = (1 - mu_hat(n)) * kappa;
    switch lower(mode)
        case 'mean',   mu_hat(n) = a / (a + b_beta);
        case 'sample', mu_hat(n) = betarnd(a, b_beta);
        otherwise, error('Unknown mode. Use ''mean'' or ''sample''.');
    end
end
predicted_updates = [NaN; diff(mu_hat)];
end
