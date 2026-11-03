
clc
clearvars

% Define a range of kappa values to test
kappa_values = linspace(1, 25, 100);

% Use representative/fixed values for alpha and sigma (e.g., from your fits)
alphaval = 0.2;
sigma = 0.04;


data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

dataContrast = data;
% dataContrast = data(data.contrast ==0,:);
dataContrast = dataContrast(dataContrast.choice_cond ~=3,:);
% dataContrast = dataContrast(dataContrast.choice_cond ~=1,:);
dataContrast.condiff_relative = (dataContrast.contrast_left - dataContrast.contrast_right)./2;

figure("Position",[200,200,1000,600])
hold on

for n = 1:10%:numSubjs

    dataSubj = dataContrast(dataContrast.ID == uniqueID(n),:);

    % Prepare your data
    mu_hat = dataSubj.mu_congruence; % slider reported probabilities (N_trials x 1)
    dataSubj(isnan(mu_hat),:) = []; % remove all trials with mu nans
    mu_hat = mu_hat(~isnan(mu_hat)); % remove nans from data
    blocks = dataSubj.blocks;
    state = dataSubj.state;
    condiff = dataSubj.condiff_relative;
    % choices = dataSubj.choice; %(dataSubj.blocks == 1); % Vector of subject's choices

    % for h = 1:height(dataSubj)
    %     if dataSubj.contrast(h) == 0
            recoded_rewards = dataSubj.recoded_reward;%(dataSubj.blocks == 1); % Vector of outcomes
    %     else
    %         recoded_rewards(h) = dataSubj.recoded_reward(h);%(dataSubj.blocks == 1); % Vector of outcomes
    %     end
    % end

    rewards = dataSubj.correct;
    choices = dataSubj.choice;
    contrast = dataSubj.contrast;

    % Use your data for one subject (or pooled)
    % [kappa_grid, nll_grid] = simulate_nll_surface_kappa(alphaval, sigma, kappa_values, mu_hat, ...
    %     blocks, rewards, condiff, choices, recoded_rewards);

    fixed_kappa = randi([1, 30]); % for example
    params_sim = [alphaval, fixed_kappa, sigma];
    % mu_hat_sim = simulate_slider_PWRL(params_sim, blocks, rewards, condiff, choices, recoded_rewards);

    [mu_hat_sim,data] = simulate_slider_PWRL(params_sim, blocks, rewards, condiff, recoded_rewards, 0, choices, [], []);
    nll_grid = NaN(length(kappa_values),1);
for i = 1:length(kappa_values)
    params = [alphaval, kappa_values(i), sigma];
    nll_grid(i) = nll_fitSlider_PWRL(params, mu_hat_sim, blocks, rewards, condiff, choices, recoded_rewards);
end

hold on 
subplot(5,5,n)
plot(kappa_values, nll_grid, 'LineWidth', 2);
xlabel('Kappa');
ylabel('NLL');
title('');
grid on;
hold on;
plot(fixed_kappa, min(nll_grid), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % Mark true kappa
% legend('NLL','True kappa');
    
end




function [kappa_grid, nll_grid] = simulate_nll_surface_kappa(alpha, sigma, kappa_values, mu_hat, blocks, rewards, condiff, choices, recoded_rewards)
    % simulate_nll_surface_kappa
    % Computes NLL for a range of kappa values, holding alpha and sigma fixed.
    %
    % Inputs:
    %   alpha           - fixed value for alpha parameter
    %   sigma           - fixed value for sigma parameter
    %   kappa_values    - vector of kappa values to evaluate
    %   mu_hat          - observed slider probabilities
    %   blocks          - block indices
    %   rewards         - reward outcomes
    %   condiff         - contrast difference
    %   choices         - subject choices
    %   recoded_rewards - recoded rewards
    %
    % Outputs:
    %   kappa_grid      - vector of kappa values
    %   nll_grid        - corresponding NLL values

    n_kappa = length(kappa_values);
    nll_grid = NaN(n_kappa,1);

    for i = 1:n_kappa
        params = [alpha, kappa_values(i), sigma];
        nll_grid(i) = nll_fitSlider_PWRL(params, mu_hat, blocks, rewards, condiff, choices, recoded_rewards);
    end

    kappa_grid = kappa_values;
end
