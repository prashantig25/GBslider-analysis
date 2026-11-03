% Parameter Recovery Script
clc
clearvars

% Number of parameters
n_parameters = 100;

% Number of starting points
n_startingPoints = 30;

% Define true parameters
true_alpha = unifrnd(0,0.3,[n_parameters,1]);
true_kappa = unifrnd(1,20,[n_parameters,1]); % We'll keep this fixed for simplicity

% Number of simulations
n_simulations = 1;

% Number of trials in each simulation
n_trials = 500;

options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point');

% Initial parameter guesses
initAlpha = unifrnd(0,0.3,[n_parameters,n_startingPoints]); % [alpha, beta]
initKappa = unifrnd(0,20,[n_parameters,n_startingPoints]); % [alpha, beta]
lb = [0, 1]; % Lower bounds (alpha, beta must be >= 0)
ub = [0.3, 100]; % Upper bounds (alpha <= 1, beta reasonable range)

% Arrays to store best recovered parameters
best_recovered_alphas = NaN(n_parameters,1);
best_recovered_kappas = NaN(n_parameters,1);
best_nlls = Inf(n_parameters,1); % Start with Inf to track the minimum

% Arrays to store recovered parameters
recovered_alphas = NaN(n_startingPoints, n_parameters);
recovered_kappas = NaN(n_startingPoints, n_parameters);
nllAll = NaN(n_startingPoints,n_parameters);

parfor p = 1:n_parameters

    for sp = 1:n_startingPoints

        % Generate simulated data
        rewards = rand(1,n_trials) < 0.7; 
        state = rand(1,n_trials) < 0.5;
        blocks_all = [];
        for nb = 1:n_trials./25
            blocks_all = [repelem(nb,25), blocks_all];
        end
        [mu_hat] = simulate_fixed_kappa(true_kappa(p),blocks_all,rewards,true_alpha(p),state);

        % Optimize using fmincon
%         options = optimset('Display', 'iter');
        options = optimoptions('fmincon', ...
            'Display', 'off', ...
            'Algorithm', 'active-set', ...
            'FiniteDifferenceType', 'central', ...
            'MaxIterations', 1000, ...
            'FunctionTolerance', 1e-6);
        [recovered_params,nll] = fmincon(@(params) nll_fitSlider_basicRL([params(1), params(2)], mu_hat, blocks_all, state, rewards), ...
            [initAlpha(p,sp), initKappa(p,sp)], [], [], [], [], lb, ub, [], options);

        % Store recovered parameters
        recovered_alphas(sp,p) = recovered_params(1);
        recovered_kappas(sp,p) = recovered_params(2);
        nllAll(sp,p) = nll;

        % Check if this is the best solution so far
        if nll < best_nlls(p)
            best_nlls(p) = nll;
            best_recovered_alphas(p) = recovered_params(1);
            best_recovered_kappas(p) = recovered_params(2);
        end
    end

    fprintf('Currently processing: Parameter %d/%d, Starting Point %d/%d\n', ...
                p, n_parameters, sp, n_startingPoints);

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
color = lines(1);
figure
hold on
scatter(true_alpha, best_recovered_alphas, "filled", "o", "MarkerEdgeColor", ...
    color, 'MarkerFaceColor',color, 'MarkerFaceAlpha', 0.4, SizeData=50)
lsline
xlabel('True alpha')
ylabel('Best recovered alpha')
title('RL model')

figure
hold on
scatter(true_kappa, best_recovered_kappas, "filled", "o", "MarkerEdgeColor", ...
    color, 'MarkerFaceColor', color, 'MarkerFaceAlpha', 0.4, SizeData=50)
lsline
xlabel('True kappa')
ylabel('Best recovered kappa')
title('RL model')

% Print summary statistics
% fprintf('Mean True Alpha: %.2f, Mean Best Recovered Alpha: %.2f\n', mean(true_alpha), mean(best_recovered_alphas));
% fprintf('Mean True Beta: %.2f, Mean Best Recovered Beta: %.2f\n', mean(true_beta), mean(best_recovered_kappas));

%%

figure
hold on
scatter(true_alpha,nanmean(recovered_alphas,1),"filled","o","MarkerEdgeColor",'k','MarkerFaceColor','b','MarkerFaceAlpha',0.4)
lsline
xlabel('True alpha')
ylabel('Recovered alpha')
title('Starting beta ([0,2])')

figure
hold on
scatter(true_beta,nanmean(recovered_kappas,1))
lsline
xlabel('True beta')
ylabel('Recovered beta')
title('Starting beta ([0,2])')

% Print summary statistics
% fprintf('True Alpha: %.2f, Mean Recovered Alpha: %.2f, SD: %.2f\n', true_alpha, mean(recovered_alphas), std(recovered_alphas));
% fprintf('True Beta: %.2f, Mean Recovered Beta: %.2f, SD: %.2f\n', true_beta, mean(recovered_betas), std(recovered_betas));