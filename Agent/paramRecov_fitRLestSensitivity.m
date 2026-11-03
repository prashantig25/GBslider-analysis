% Parameter Recovery Script
clc
clearvars

% Number of parameters
n_parameters = 20;

% Number of starting points
n_startingPoints = 50;

% Define true parameters
true_alpha = unifrnd(0,1,[n_parameters,1]);
true_beta = unifrnd(0,20,[n_parameters,1]); % We'll keep this fixed for simplicity
true_sigma = unifrnd(0,0.1,[n_parameters,1]); % We'll keep this fixed for simplicity

% Number of simulations
n_simulations = 1;

% Number of trials in each simulation
n_trials = 300;

options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point');

% Initial parameter guesses
initAlpha = unifrnd(0,1,[n_parameters,n_startingPoints]); % [alpha, beta]
initBeta = unifrnd(0,10,[n_parameters,n_startingPoints]); % [alpha, beta]
initSigma = unifrnd(0,0.04,[n_parameters,n_startingPoints]); % [alpha, beta]

lb = [0, 0, 0]; % Lower bounds (alpha, beta must be >= 0)
ub = [1, 20, 0.1]; % Upper bounds (alpha <= 1, beta reasonable range)

% Arrays to store best recovered parameters
best_recovered_alphas = NaN(n_parameters,1);
best_recovered_betas = NaN(n_parameters,1);
best_recovered_sigma = NaN(n_parameters,1);
best_nlls = Inf(n_parameters,1); % Start with Inf to track the minimum

% Arrays to store recovered parameters
recovered_alphas = NaN(n_startingPoints, n_parameters);
recovered_betas = NaN(n_startingPoints, n_parameters);
recovered_sigma = NaN(n_startingPoints, n_parameters);
nllAll = NaN(n_startingPoints,n_parameters);

for p = 1:n_parameters

    for sp = 1:n_startingPoints

        % Generate simulated data
        [choices, rewards, condiff, states] = simulate_dataEstSensitivity(true_alpha(p), true_beta(p), true_sigma(p), n_trials);

        % Optimize using fmincon
%         options = optimset('Display', 'iter');
        options = optimoptions('fmincon', ...
            'Display', 'off', ...
            'Algorithm', 'active-set', ...
            'FiniteDifferenceType', 'central', ...
            'MaxIterations', 1000, ...
            'FunctionTolerance', 1e-6);
        [recovered_params,nll] = fmincon(@(params) neg_log_likelihoodEstSensitivity([params(1), params(2), params(3)], choices, rewards, states, condiff), ...
            [initAlpha(p,sp), initBeta(p,sp), initSigma(p,sp)], [], [], [], [], lb, ub, [], options);

        % Store recovered parameters
        recovered_alphas(sp,p) = recovered_params(1);
        recovered_betas(sp,p) = recovered_params(2);
        recovered_sigma(sp,p) = recovered_params(3);
        nllAll(sp,p) = nll;

        % Check if this is the best solution so far
        if nll < best_nlls(p)
            best_nlls(p) = nll;
            best_recovered_alphas(p) = recovered_params(1);
            best_recovered_betas(p) = recovered_params(2);
            best_recovered_sigma(p) = recovered_params(3);
        end
    end

    disp("New set of parameters...")

end


% alphas_to_test = linspace(0, 1, 100);
% betas_to_test = linspace(1, 5, 100);
% nll_surface = NaN(length(alphas_to_test), length(betas_to_test));
% 
% for i = 1:length(alphas_to_test)
%     for j = 1:length(betas_to_test)
%         nll_surface(i, j) = neg_log_likelihood([alphas_to_test(i), betas_to_test(j)], choices, rewards, states);
%     end
% end
% 
% figure;
% surf(alphas_to_test, betas_to_test, nll_surface');
% xlabel('Alpha'); ylabel('Beta'); zlabel('NLL');

% Plot results
figure
hold on
scatter(true_alpha, best_recovered_alphas, "filled", "o", "MarkerEdgeColor", 'k', 'MarkerFaceColor', 'b', 'MarkerFaceAlpha', 0.4)
lsline
xlabel('True alpha')
ylabel('Best recovered alpha')
title('Best Parameter Recovery (Starting beta = [0,10])')

figure
hold on
scatter(true_beta, best_recovered_betas)
lsline
xlabel('True beta')
ylabel('Best recovered beta')
title('Best Parameter Recovery (Starting beta = [0,10])')

figure
hold on
scatter(true_sigma, best_recovered_sigma)
lsline
xlabel('True sigma')
ylabel('Best recovered sigma')
title('Best Parameter Recovery (Starting beta = [0,10])')

% Print summary statistics
fprintf('Mean True Alpha: %.2f, Mean Best Recovered Alpha: %.2f\n', mean(true_alpha), mean(best_recovered_alphas));
fprintf('Mean True Beta: %.2f, Mean Best Recovered Beta: %.2f\n', mean(true_beta), mean(best_recovered_betas));

%%

figure
hold on
scatter(true_alpha,nanmean(recovered_alphas,1),"filled","o","MarkerEdgeColor",'k','MarkerFaceColor','b','MarkerFaceAlpha',0.4)
lsline
xlabel('True alpha')
ylabel('Recovered alpha')
% title('Starting beta ([0,2])')

figure
hold on
scatter(true_beta,nanmean(recovered_betas,1))
lsline
xlabel('True beta')
ylabel('Recovered beta')
% title('Starting beta ([0,2])')

figure
hold on
scatter(true_sigma,nanmean(recovered_sigma,1))
lsline
xlabel('True sigma')
ylabel('Recovered sigma')

% Print summary statistics
% fprintf('True Alpha: %.2f, Mean Recovered Alpha: %.2f, SD: %.2f\n', true_alpha, mean(recovered_alphas), std(recovered_alphas));
% fprintf('True Beta: %.2f, Mean Recovered Beta: %.2f, SD: %.2f\n', true_beta, mean(recovered_betas), std(recovered_betas));