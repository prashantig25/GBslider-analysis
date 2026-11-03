function [mu_hat_sim,data] = simulate_slider_PWRL(params, blocks, rewards, condiff, recoded_rewards, genChoices, choices, states, condition)

% params = [alpha, kappa, sigma]
alpha = params(1);
kappa = params(2);
sigma = params(3);
N = length(blocks);

% Initialize Q-values
q_0_0 = 0.5;
q_1_0 = 1 - q_0_0;
q_1_1 = q_0_0;
q_0_1 = 1 - q_0_0;

mu_hat_sim = NaN(N,1);

% choices = [];
% rewards = [];
% recoded = [];
% rewardsAll = [];
% choicesAll = [];
data = table();

beta = 25;

for n = 1:N-1
    % Reset Q-values at block boundaries
    if blocks(n+1) - blocks(n) == 1
        q_0_0 = 0.5;
        q_1_0 = 1-q_0_0;
        q_1_1 = q_0_0;
        q_0_1 = 1 - q_0_0;
    end

    if genChoices

        Qvalues = [q_0_0, q_0_1];

        p = exp(Qvalues .* beta) / sum(exp(Qvalues .* beta));
        choices = binornd(1, p(2)); % action

        % Generate reward
        if states(n) == 0
            if choices == 0
                if condition(n) == 2
                    rewards = (rand() < 0.9);  % 70% chance of reward for correct choice
                else
                    rewards = (rand() < 0.7);  % 70% chance of reward for correct choice
                end
            else
                if condition(n) == 2
                    rewards = (rand() < 0.1);  % 30% chance of reward for incorrect choice
                else
                    rewards = (rand() < 0.3);  % 30% chance of reward for incorrect choice
                end
            end
        else
            if choices == 1
                if condition(n) == 2
                    rewards = (rand() < 0.9);  % 70% chance of reward for correct choice
                else
                    rewards = (rand() < 0.7);  % 70% chance of reward for correct choice
                end
            else
                if condition(n) == 2
                    rewards = (rand() < 0.1);  % 30% chance of reward for incorrect choice
                else
                    rewards = (rand() < 0.3);  % 30% chance of reward for incorrect choice
                end
            end
        end

            if choices == 1
                recoded_rewards = 1-rewards;
            else
                recoded_rewards = rewards;
            end
            % recoded = [recoded; recoded_rewards];
            % rewardsAll = [rewardsAll; rewards];
            % choicesAll = [choicesAll; choices];

        end
    
        % Compute belief about perceptual state based on sigma
        belief_state = normcdf(condiff(n), 0, sigma);
        if condiff(n) < 0
            pi_0 = 1 - belief_state;
            pi_1 = belief_state;
        else
            pi_1 = belief_state;
            pi_0 = 1 - belief_state;
        end

        % Posterior-weighted state inference
        if choices(n) == 0
            pw_pi0_a0 = (q_0_0*pi_0) / ((q_0_0*pi_0) + (q_1_0*pi_1));
            pw_pi1_a1 = (q_1_0*pi_1) / ((q_1_0*pi_1) + (q_0_0*pi_0));
        else
            pw_pi0_a0 = (q_0_1*pi_0) / ((q_0_1*pi_0) + (q_1_1*pi_1));
            pw_pi1_a1 = (q_1_1*pi_1) / ((q_1_1*pi_1) + (q_0_1*pi_0));
        end

        % Choose which posterior-weighted pi to use
        if rewards(n) == 1
            pw_pi = pw_pi0_a0;
        else
            pw_pi = pw_pi1_a1;
        end

        if q_0_0 == q_1_0
            if pi_0 >= pi_1
                pw_pi = pi_0;
            else
                pw_pi = pi_1;
            end
        end

        % Update Q-values based on reward
        if pi_0 >= pi_1 % if state = 0
            q_0_0 = q_0_0 + pw_pi * alpha * (recoded_rewards(n) - q_0_0);
        elseif pi_0 < pi_1
            q_0_0 = q_0_0 + pw_pi * alpha * ((1 - recoded_rewards(n)) - q_0_0);
        end
        q_1_0 = 1-q_0_0;
        q_1_1 = q_0_0;
        q_0_1 = 1 - q_0_0;
        
        % For each trial, compute alpha and beta for the Beta distribution
        a = q_0_0 * kappa;
        b = (1 - q_0_0) * kappa;

        % % Simulate a reported probability from the Beta distribution
        mu_hat_sim(n) = betarnd(a, b);
end

    % Handle the last trial (optional: copy previous value or set to NaN)
    mu_hat_sim(N) = mu_hat_sim(N-1);

    % data.choice = choicesAll;
    % data.recoded_rewards = recoded;
    % data.rewards = rewardsAll;
    data.mu = mu_hat_sim(1:end);
    data.condiff_relative = condiff(1:end);
    data.blocks = blocks(1:end);

end
