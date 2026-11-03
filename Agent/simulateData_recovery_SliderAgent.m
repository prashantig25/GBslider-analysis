function [mu_hat, data] = simulateData_recovery_SliderAgent(kappa, blocks, nBlocks, ...
    nTrials, sigma, condition)
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
    choices = [];
    condiff = [];
    rewards = [];
    trials = [];
    data = table();
    uniqueBlocks = unique(blocks);

    % Simulate data for each block
    for bl = 1:nBlocks

    agent = Agent();
    agent.sigma = sigma;
    % agent.beta = 30;

    task = Task();
    task.condition = condition(bl);

    for t = 1:nTrials

        task.taskMu(0);
        task.state_sample();
        task.contrast_sample();

        % get observed contrast difference
        agent.o_t = task.c_t;
    
        % compute value
        agent.p_s_giv_o(agent.o_t); % belief state
        agent.decide_e(agent.o_t);
        task.reward_sample(agent.a_t); 
    
        % learn
        agent.learn(task.r_t)
    
        % For each trial, compute alpha and beta for the Beta distribution
        a = agent.G * kappa;
        b = (1 - agent.G) * kappa;
        
        % Generate Beta-distributed values
        mu_hat = [mu_hat; betarnd(a, b)];
        rewards = [rewards; task.r_t];
        choices = [choices; agent.a_t];
        condiff = [condiff; task.c_t];
        trials = [trials; t];
    end
        
    end

    % Apply bounds to prevent 0/1 values (matching original code)
    mu_hat = min(max(mu_hat, eps), 1-eps);

    data.choice = choices;
    data.rewards = rewards;
    data.mu = mu_hat;
    data.condiff_relative = condiff;
    data.blocks = blocks.';
    data.trials = trials;
end
