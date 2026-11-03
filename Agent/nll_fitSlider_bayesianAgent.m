function [nll] = nll_fitSlider_bayesianAgent(params, mu_hat, data, nBlocks, nTrials, blocks, rewards)

% beta = params(3); % beta
sigma = params(2); % sigma
kappa = params(1); % kappa concentration parameter
nll_trial = NaN(nTrials,nBlocks);

% mu_blocks = mu_hat(blocks == b);
eps = 1e-9;
mu_hat(mu_hat <= 0) = eps;
mu_hat(mu_hat >= 1) = 1 - eps;
mu = [];

uniqueBlocks = unique(blocks);

for bl = 1:nBlocks

    agent = Agent();
    % agent.beta = beta;
    agent.sigma = sigma;

    dataBlocks = data(data.blocks == uniqueBlocks(bl),:);
    choices = dataBlocks.choice;
    condiff = dataBlocks.condiff_relative;
    rewardsBlocks = rewards(data.blocks == uniqueBlocks(bl),:);
    mu_hatBlocks = mu_hat(data.blocks == uniqueBlocks(bl),:);
    % rewardsBlocks = correct;

    for t = 1:height(dataBlocks)

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

    if any(a <= 0 | b <= 0)
        disp('Invalid a or b detected:');
        disp([a(:), b(:)]);
    end

    % Evaluate the probability density of the observed mu_hat
    p = betapdf(mu_hatBlocks(t), a, b);
    % hold on
    % plot(p)

    % Negative log-likelihood
    nll_trial(t,bl) = log(p);
    mu = [mu;agent.G];
    end
end
nll = -nansum(nll_trial,"all");

% figure
% plot(1:200,mu_hat,"Color",'r','LineWidth',2)
% hold on
% plot(1:200,mu,"Color",'b','LineWidth',2)
% hold on
% xlabel("Trials")
% ylabel("mu")
% legend("Simulated mu","Model-predicted mu")
end
