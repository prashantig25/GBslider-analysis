function [mu_hat] = simulate_kappa_BayesianAgent(kappa, blocks, nBlocks, ...
    nTrials, data,rewards, sigma)
    % Inputs:
    % kappa - fixed concentration parameter
    % n_blocks - number of experimental blocks
    % trials_per_block - trials per block
    
    % Generate true μ values for each block (example: spaced between 0.1-0.9)
    % true_mu = linspace(0.1, 0.9, n_blocks)';
    
    % Create block structure
    % blocks = repelem(1:n_blocks, trials_per_block)';
    
    % Initialize output
    mu_hat = [];
    uniqueBlocks = unique(blocks);

    % Simulate data for each block
    for bl = 1:nBlocks

    agent = Agent();
    agent.sigma = 0.04;
    agent.beta = 30;

    dataBlocks = data(data.blocks == uniqueBlocks(bl),:);
    choices = dataBlocks.choice;
    condiff = dataBlocks.condiff_relative;
    rewardsBlocks = rewards(data.blocks == uniqueBlocks(bl),:);
    % rewardsBlocks = dataBlocks.correct;

    for t = 1:nTrials

        % get observed contrast difference
        agent.o_t = condiff(t);
    
        % compute value
        agent.p_s_giv_o(agent.o_t); % belief state
        agent.compute_valence(); % update mu and calculate values
        agent.softmax(); % use softmax to get probabilities
        agent.a_t = choices(t);     
    
        % learn
        agent.learn(rewardsBlocks(t))
    
        % For each trial, compute alpha and beta for the Beta distribution
        a = agent.G * kappa;
        b = (1 - agent.G) * kappa;
        
        % Generate Beta-distributed values
        mu_hat = [mu_hat; betarnd(a, b)];
    end
        
    end

    % Apply bounds to prevent 0/1 values (matching original code)
    mu_hat = min(max(mu_hat, eps), 1-eps);
end
