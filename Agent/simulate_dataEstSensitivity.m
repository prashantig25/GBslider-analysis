function [choices, rewards, condiff, state, contrast] = simulate_dataEstSensitivity(alpha, beta, sigma, n_trials)
    % Simulate data for the negative log-likelihood function

    % Initialize variables
    choices = NaN(n_trials, 1);
    rewards = NaN(n_trials, 1);
    state = randi([0, 1], [n_trials, 1]); % Randomly assign states (0 or 1)
    condiff = NaN(n_trials, 1);
    for i = 1:length(state)

        if state(i) == 0
            condiff(i) = unifrnd(-0.1,0,[1,1]);    % Random values between -0.5 and 0.5
        else 
            condiff(i) = unifrnd(0,0.1,[1,1]);    % Random values between -0.5 and 0.5
        end

    end
    contrast = rand(n_trials, 1);        % Random contrast values between 0 and 1
    q_0_0 = 0.5;                         % Initial Q-value

    for t = 1:n_trials
        % Compute belief about perceptual state based on sigma
        belief_state = normcdf(condiff(t), 0, sigma);

        if condiff(t) < 0
            pi_0 = 1 - belief_state;
            pi_1 = belief_state;
        else
            pi_1 = belief_state;
            pi_0 = 1 - belief_state;
        end

        % Compute Q-values
        q_0_1 = 1 - q_0_0;
        q_1_0 = 1 - q_0_0;
        q_1_1 = q_0_0;

        capital_q_0 = q_0_0 * pi_0 + q_1_0 * pi_1;
        capital_q_1 = q_0_1 * pi_0 + q_1_1 * pi_1;

        capital_q_a = [capital_q_0, capital_q_1];

        % Compute choice probabilities using softmax
        p_a_t = exp(capital_q_a .* beta) / sum(exp(capital_q_a .* beta));

        % Simulate choice (binary: 0 or 1)
        choices(t) = binornd(1, p_a_t(2)); % Probability of choosing action "1"

        % Generate reward based on state and choice
        if state(t) == 0
            if choices(t) == 0
                rewards(t) = (rand() < 0.7); % 70% chance of reward for correct choice
            else
                rewards(t) = (rand() < 0.3); % 30% chance of reward for incorrect choice
            end
        else
            if choices(t) == 1
                rewards(t) = (rand() < 0.7); % 70% chance of reward for correct choice
            else
                rewards(t) = (rand() < 0.3); % 30% chance of reward for incorrect choice
            end
        end

        % Update Q-values based on reward and belief states
        if pi_0 >= pi_1
            q_0_0 = q_0_0 + pi_0 * alpha * (rewards(t) - q_0_0);
        else
            q_0_0 = q_0_0 + pi_1 * alpha * ((1 - rewards(t)) - q_0_0);
        end

        % Ensure Q-values stay within [0, 1]
        q_0_0 = max(0, min(1, q_0_0));
    end
end
