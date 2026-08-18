clc
clearvars

% Number of parameters
n_parameters = 60;
% Number of starting points
n_startingPoints = 15;

% PARAMETERS TO ESTIMATE: kappa, sigma, confirmBias, noconfirmBias
kappaRange = unifrnd(0, 25, [n_parameters, 1]);
sigmaRange = unifrnd(0, 0.05, [n_parameters, 1]);
confirmBiasRange = unifrnd(0, 0.4, [n_parameters, 1]);
noconfirmBiasRange = unifrnd(0, 0.4, [n_parameters, 1]);

n_simulations = 1;
% Number of trials in each simulation
n_trials = 200;

% Initial parameter guesses for KAPPA, SIGMA, CONFIRMBIAS, NOCONFIRMBIAS
initKappa = unifrnd(0, 25, [n_parameters, n_startingPoints]);
initSigma = unifrnd(0, 0.05, [n_parameters, n_startingPoints]);
initConfirmBias = unifrnd(0, 0.4, [n_parameters, n_startingPoints]);
initNoconfirmBias = unifrnd(0, 0.4, [n_parameters, n_startingPoints]);

% Bounds for KAPPA, SIGMA, CONFIRMBIAS, NOCONFIRMBIAS
lb = [0, 0, 0, 0];     % Lower bounds for [kappa, sigma, confirmBias, noconfirmBias]
ub = [25, 0.05, 0.4, 0.4];   % Upper bounds for [kappa, sigma, confirmBias, noconfirmBias]

% Arrays to store best recovered parameters
best_recovered_kappas = NaN(n_parameters, 1);
best_recovered_sigmas = NaN(n_parameters, 1);
best_recovered_confirmBias = NaN(n_parameters, 1);
best_recovered_noconfirmBias = NaN(n_parameters, 1);
best_nlls = Inf(n_parameters, 1); % Start with Inf to track the minimum

data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
data = data(data.choice_cond ~= 3,:);
numSubjs = length(uniqueID);
% Precompute contrast difference
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;
data(data.condition == 2,:) = [];

parfor p = 1:n_parameters
    subj = preprocess_fitSlider(data, uniqueID(p), 0);
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    
    
    for sp = 1:n_startingPoints
        
        % Generate predicted data using current parameter ranges
        [predictedUp, mu_hat] = predict_allModels.predict_RLsigma_confirmBias(...
            [kappaRange(p), sigmaRange(p), confirmBiasRange(p), noconfirmBiasRange(p)], ...
            subj.blocks, rewards, subj.state, subj.condiff, subj.dataTable.confirm_rew, 'sample');
        
        % Optimization options
        options = optimoptions('fmincon', ...
            'Display', 'off', ...
            'Algorithm', 'interior-point', ...
            'FiniteDifferenceType', 'central', ...
            'MaxIterations', 1000, ...
            'FunctionTolerance', 1e-6);
        
        % Define objective function that estimates kappa, sigma, confirmBias, noconfirmBias
        % You'll need to create the corresponding nll function
        nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_confirmBias(...
            [params(1), params(2), params(3), params(4)], ...
            mu_hat, subj.blocks, rewards, subj.condiff, subj.dataTable.confirm_rew);
        
        initial_guess = [initKappa(p, sp), initSigma(p, sp), ...
                        initConfirmBias(p, sp), initNoconfirmBias(p, sp)];

        % Optimize all 4 parameters
        [recovered_params, nll] = fmincon(nll_fun, initial_guess, ...
            [], [], [], [], lb, ub, [], options);
        
        % Check if this is the best solution so far
        if nll < best_nlls(p)
            best_nlls(p) = nll;
            best_recovered_kappas(p) = recovered_params(1);
            best_recovered_sigmas(p) = recovered_params(2);
            best_recovered_confirmBias(p) = recovered_params(3);
            best_recovered_noconfirmBias(p) = recovered_params(4);
        end
    end
    fprintf('Currently processing: Parameter %d/%d, Starting Point %d/%d\n', ...
        p, n_parameters, sp, n_startingPoints);
end

%% Plotting results
color = copper(5);

% Kappa recovery plot
figure
scatter(kappaRange, best_recovered_kappas, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual Kappa Parameter')
ylabel('Best Recovered Kappa Parameter')
title('Kappa Recovery')
axis equal
grid on

% Sigma recovery plot
figure
scatter(sigmaRange, best_recovered_sigmas, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual Sigma Parameter')
ylabel('Best Recovered Sigma Parameter')
title('Sigma Recovery')
axis equal
grid on

% ConfirmBias recovery plot
figure
scatter(confirmBiasRange, best_recovered_confirmBias, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual ConfirmBias Parameter')
ylabel('Best Recovered ConfirmBias Parameter')
title('ConfirmBias Recovery')
axis equal
grid on

% NoconfirmBias recovery plot
figure
scatter(noconfirmBiasRange, best_recovered_noconfirmBias, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual NoconfirmBias Parameter')
ylabel('Best Recovered NoconfirmBias Parameter')
title('NoconfirmBias Recovery')
axis equal
grid on

% Combined recovery plot (2x2 subplots)
figure('Position', [100, 100, 1200, 900])

subplot(2,2,1)
scatter(kappaRange, best_recovered_kappas, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual Kappa')
ylabel('Recovered Kappa')
title('Kappa Recovery')
axis equal
grid on

subplot(2,2,2)
scatter(sigmaRange, best_recovered_sigmas, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual Sigma')
ylabel('Recovered Sigma')
title('Sigma Recovery')
axis equal
grid on

subplot(2,2,3)
scatter(confirmBiasRange, best_recovered_confirmBias, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual ConfirmBias')
ylabel('Recovered ConfirmBias')
title('ConfirmBias Recovery')
axis equal
grid on

subplot(2,2,4)
scatter(noconfirmBiasRange, best_recovered_noconfirmBias, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual NoconfirmBias')
ylabel('Recovered NoconfirmBias')
title('NoconfirmBias Recovery')
axis equal
grid on

sgtitle('Parameter Recovery for RLsigma ConfirmBias Model')

%% Calculate and display recovery statistics
fprintf('\n=== PARAMETER RECOVERY STATISTICS ===\n');

% Correlation coefficients
r_kappa = corr(kappaRange, best_recovered_kappas, 'rows', 'complete');
r_sigma = corr(sigmaRange, best_recovered_sigmas, 'rows', 'complete');
r_confirmBias = corr(confirmBiasRange, best_recovered_confirmBias, 'rows', 'complete');
r_noconfirmBias = corr(noconfirmBiasRange, best_recovered_noconfirmBias, 'rows', 'complete');

fprintf('Correlation coefficients:\n');
fprintf('Kappa: r = %.3f\n', r_kappa);
fprintf('Sigma: r = %.3f\n', r_sigma);
fprintf('ConfirmBias: r = %.3f\n', r_confirmBias);
fprintf('NoconfirmBias: r = %.3f\n', r_noconfirmBias);

% Mean absolute error
mae_kappa = mean(abs(kappaRange - best_recovered_kappas), 'omitnan');
mae_sigma = mean(abs(sigmaRange - best_recovered_sigmas), 'omitnan');
mae_confirmBias = mean(abs(confirmBiasRange - best_recovered_confirmBias), 'omitnan');
mae_noconfirmBias = mean(abs(noconfirmBiasRange - best_recovered_noconfirmBias), 'omitnan');

fprintf('\nMean Absolute Error:\n');
fprintf('Kappa: MAE = %.4f\n', mae_kappa);
fprintf('Sigma: MAE = %.4f\n', mae_sigma);
fprintf('ConfirmBias: MAE = %.4f\n', mae_confirmBias);
fprintf('NoconfirmBias: MAE = %.4f\n', mae_noconfirmBias);

% Root mean square error
rmse_kappa = sqrt(mean((kappaRange - best_recovered_kappas).^2, 'omitnan'));
rmse_sigma = sqrt(mean((sigmaRange - best_recovered_sigmas).^2, 'omitnan'));
rmse_confirmBias = sqrt(mean((confirmBiasRange - best_recovered_confirmBias).^2, 'omitnan'));
rmse_noconfirmBias = sqrt(mean((noconfirmBiasRange - best_recovered_noconfirmBias).^2, 'omitnan'));

fprintf('\nRoot Mean Square Error:\n');
fprintf('Kappa: RMSE = %.4f\n', rmse_kappa);
fprintf('Sigma: RMSE = %.4f\n', rmse_sigma);
fprintf('ConfirmBias: RMSE = %.4f\n', rmse_confirmBias);
fprintf('NoconfirmBias: RMSE = %.4f\n', rmse_noconfirmBias);