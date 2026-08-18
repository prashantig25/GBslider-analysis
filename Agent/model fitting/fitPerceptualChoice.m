clc
clearvars

data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;
data(data.condition ~= 2,:) = [];
data(data.trials <=5,:) = [];
for h = 1:height(data)
    if data.contrast(h) == 1
        data.choice(h) = 1-data.choice(h);
    end
end

sigmaParameter = NaN(numSubjs, 1);
nll_bayesianAgent = NaN(numSubjs, 1);
init_params = 0.02; 
lb = 0.001;
ub = 0.1;

for n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n), 0);
    nll_fun = @(params) nll_perceptualChoice(params, subj.dataTable, ...
        length(unique(subj.blocks)), 20, unique(subj.blocks));
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    sigmaParameter(n) = params(1);
    nll_bayesianAgent(n) = nll;

    fprintf('Subject number: %d\n', n);

    % 3. Plot likelihood landscape for one subject
    % sigma_range = linspace(0.01, 0.5, 50);
    % nll_values = arrayfun(@(s) nll_perceptualChoice(s, subj.dataTable, ...
    %     length(unique(subj.blocks)), 20, unique(subj.blocks)), sigma_range);
    % hold on; plot(sigma_range, nll_values);
    % xlabel('Sigma'); ylabel('Negative Log-Likelihood');
end
params_bayesianAgent.sigma = sigmaParameter;
% safe_saveall('sigma_perceptualChoice.mat', params_bayesianAgent);
% safe_saveall('nll_perceptualChoice.mat', nll_bayesianAgent);

%%

function nll = nll_perceptualChoice(params, data, nBlocks, nTrials, blocks)
sigma = params(1);
nll_trial = NaN(nTrials,nBlocks);
uniqueBlocks = unique(blocks);

for bl = 1:nBlocks
    agent = Agent();
    agent.task_agent_analysis = 1;
    agent.confirmation_bias = 0;
    agent.sigma = sigma;
    dataBlocks = data(data.blocks == uniqueBlocks(bl),:);
    choices = dataBlocks.choice;
    condiff = dataBlocks.condiff_relative;
   for t = 1:height(dataBlocks)

        % Bayesian agent inference and learning steps
        agent.o_t = condiff(t);
        agent.p_s_giv_o(agent.o_t);
        agent.decide_p();

        % Log-likelihood for this trial
        % nll_trial(t,bl) = log(agent.p_d_t(choices(t) + 1));

        p = max(min(agent.p_d_t(choices(t) + 1), 1 - 1e-10), 1e-10);
        nll_trial(t,bl) = log(p);
    end
end
nll = -nansum(nll_trial,"all");
end