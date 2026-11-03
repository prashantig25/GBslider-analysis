% Function to simulate data
function [choices, rewards, condiff, states] = simulate_data(alpha, beta, n_trials)

    choices = NaN(n_trials, 1);
    rewards = NaN(n_trials, 1);
    states = randi([0,1],[n_trials,1]);
    condiff = rand(n_trials, 1) - 0.5; % Random values between -0.5 and 0.5
    q_0_0 = 0.5;

    
    for t = 1:n_trials

%         if mod(t,25) == 0
%             q_0_0 = 0.5;
%         end

        % Compute choice probabilities
        q_0_1 = 1 - q_0_0;

        if states(t) == 0
            Qvalues = [q_0_0, q_0_1];
        else
            Qvalues = [q_0_1, q_0_0];
        end
        p = exp(Qvalues .* beta) / sum(exp(Qvalues .* beta));
        choices(t) = binornd(1, p(2)); % action
%         choices(t) = double(rand() < p(1));  % Actual softmax implementation

%         [~, argmax] = max(p);
%         choices(t) = argmax - 1;

%         if rand() < 0.3
%             choices(t) = randi([0, 1]);
%         end

        % Generate reward
        if states(t) == 0
            if choices(t) == 0
                rewards(t) = (rand() < 0.7);  % 70% chance of reward for correct choice
            else
                rewards(t) = (rand() < 0.3);  % 30% chance of reward for incorrect choice
            end
        else
            if choices(t) == 1
                rewards(t) = (rand() < 0.7);  % 70% chance of reward for correct choice
            else
                rewards(t) = (rand() < 0.3);  % 30% chance of reward for incorrect choice
            end
        end


        if choices(t) == 1
            rewards(t) = 1-rewards(t);
        end
        
        % Update Q-value
        % Update Q-value with small noise
        if states(t) == 0
            q_0_0 = q_0_0 + alpha * (rewards(t) - q_0_0) + normrnd(0, 0.01);
        else
            q_0_0 = q_0_0 + alpha * ((1-rewards(t)) - q_0_0) + normrnd(0, 0.01);
        end
%         
        % Ensure Q-values stay within [0, 1]
        q_0_0 = max(0, min(1, q_0_0));
        
    end
end