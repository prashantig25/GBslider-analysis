function [nll] = NLL_fitBeta_Agent(params, data, sigma)
    % INITIALISE STUFF
    beta = params(1);  % Inverse temperature
    numTrials = 25; % number of trials
    blocks = unique(data.blocks);
    numBlocks = length(blocks);
    nll = NaN(numTrials,numBlocks);
    mu = NaN(numTrials,numBlocks);

    for b = 1:numBlocks

        agent = Agent();
        agent.beta = beta;

        dataBlocks = data(data.blocks == blocks(b),:);
        condiff = dataBlocks.condiff_relative;
        reward = dataBlocks.correct;
        state = dataBlocks.state;
        choices = dataBlocks.choice;
        agent.sigma = sigma;
        contrast = dataBlocks.contrast;

        for t = 1:numTrials

            % get observed contrast difference
            agent.o_t = condiff(t);
%             agent.observation_sample(condiff(t));

            % compute value
            agent.p_s_giv_o(agent.o_t); % belief state
            agent.compute_valence(); % update mu and calculate values
            agent.softmax(); % use softmax to get probabilities
            agent.a_t = choices(t);
            
            % Compute log-likelihood for specific trial t and block b
            nll(t, b) = log(max(agent.p_a_t(choices(t) + 1),eps));  % +1 for MATLAB's 1-based indexing

            % learn
            agent.learn(reward(t))

%             mu(t, b) = agent.G;

        end

    end

    nll = -1 * sum(nll,"all");
end
