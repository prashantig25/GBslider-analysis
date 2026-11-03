function [mu_hat] = simulate_fixed_kappa(kappa, blocks, rewards, alpha, state)
    % Inputs:
    % kappa - fixed concentration parameter
    % n_blocks - number of experimental blocks
    % trials_per_block - trials per block
    
    % Generate true μ values for each block (example: spaced between 0.1-0.9)
    % true_mu = linspace(0.1, 0.9, n_blocks)';
    
    % Create block structure
    % blocks = repelem(1:n_blocks, trials_per_block)';
    
    % Initialize output
    mu_hat = zeros(size(blocks));
    q_0_0 = 0.5;
    
    % Simulate data for each block
    for n = 1:length(blocks)-1

        if blocks(n+1) - blocks(n) == 1
            q_0_0 = 0.5;
        end

        if state(n) == 0
            q_0_0 = q_0_0 + alpha * (rewards(n) - q_0_0);
        else
            q_0_0 = q_0_0 + alpha * ((1 - rewards(n)) - q_0_0);
        end

        % Calculate Beta parameters
        a = q_0_0 * kappa;
        b_beta = (1 - q_0_0) * kappa;
        
        % Generate Beta-distributed values
        % block_trials = (blocks == b);
        mu_hat(n) = betarnd(a, b_beta);
    end
    
    % Apply bounds to prevent 0/1 values (matching original code)
    mu_hat = min(max(mu_hat, eps), 1-eps);
    
end
