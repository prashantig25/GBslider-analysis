function params = fit_PWRL_model(choices, rewards, state)
    % Fit a simple Q-learning model to participant choice data
    % Inputs:
    %   choices - vector of choices (1 or 2)
    %   rewards - vector of rewards (0 or 1)
    % Output:
    %   params - fitted parameters [alpha, beta]

    % Initial parameter guesses
    init_params = [0.1, 0.05, 0.4]; % [alpha, beta]
    lb = [0, 0, 0]; % Lower bounds (alpha, beta must be >= 0)
    ub = [1, 1, 1]; % Upper bounds (alpha <= 1, beta reasonable range)
    
    % Optimize using fmincon
    options = optimset('Display', 'iter');
    params = fmincon(@(p) neg_log_likelihoodPWRL(p, choices, rewards, state), init_params, [], [], [], [], lb, ub, [], options);
end