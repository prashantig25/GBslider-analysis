function params = fit_RL_model(choices, rewards, state, condiff, contrast)
    % Fit a simple Q-learning model to participant choice data
    % Inputs:
    %   choices - vector of choices (1 or 2)
    %   rewards - vector of rewards (0 or 1)
    % Output:
    %   params - fitted parameters [alpha, beta]

    % Initial parameter guesses
    init_params = [0.1, 0.5, 0.01]; % [alpha, beta]
    lb = [0, 0, 0]; % Lower bounds (alpha, beta must be >= 0)
    ub = [1, 10, 0.1]; % Upper bounds (alpha <= 1, beta reasonable range)
    
    % Optimize using fmincon
    options = optimset('Display', 'iter');
    params = fmincon(@(p) neg_log_likelihoodEstSensitivity(p, choices, rewards, state, condiff, contrast), init_params, [], [], [], [], lb, ub, [], options);
end