clc
clearvars

% Number of parameters
n_parameters = 10;

% Number of starting points
n_startingPoints = 30;

% Define true parameters
true_alpha = unifrnd(0,1,[n_parameters,1]);
true_beta = unifrnd(1,8,[n_parameters,1]); % We'll keep this fixed for simplicity

% Number of trials in each simulation
n_trials = 500;

for p = 1%:n_parameters
    [choices, rewards, condiff, states] = simulate_data(true_alpha(p), true_beta(p), n_trials);

    alpha_range = linspace(0.01, 0.99, 50);  % Avoid 0 and 1 boundaries
    beta_range = linspace(0.1, 20, 50);
    [Alpha, Beta] = meshgrid(alpha_range, beta_range);
    logLikelihood = zeros(size(Alpha));

    for i = 1:numel(Alpha)

        params = [Alpha(i), Beta(i)];

        % Get NLL
        logLikelihood(i) = neg_log_likelihood(params, choices, rewards, states, []);

    end
    logLikelihood = reshape(logLikelihood, size(Alpha));

    figure('Position', [100 100 300 300])

    % 3D Surface plot
    surf(Alpha, Beta, logLikelihood, 'EdgeColor','none');
    view(-45, 30);
    xlabel('\alpha');
    ylabel('\beta');
    zlabel('NLL');
    title(strcat("Alpha = ", num2str(true_alpha(p)), " ","Beta = ", num2str(true_beta(p))))
    colorbar
    saveas(gcf,num2str(p),'png')
end
