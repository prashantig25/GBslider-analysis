%% ========================================================================
%  Script: Fit RL Models to Participant Data and Save Parameters (+AIC/BIC)
%  ------------------------------------------------------------------------
clc; clearvars;

%% =================== LOAD AND PREPARE DATA =============================
data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

% Precompute contrast difference
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;
data = data(data.choice_cond ~= 3,:);

%% =================== BASIC RL MODEL ====================================
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
nll_basicRL = NaN(numSubjs, 1);

init_params = [0.1, 5]; % [alpha, kappa]
lb = [0, 1];
ub = [1, 100];

for n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n));
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL(params, subj.mu_hat, subj.blocks, subj.state, rewards);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    alphaParameter(n) = params(1);
    kappaParameter(n) = params(2);
    nll_basicRL(n) = nll;
end

params_basicRL.alpha = alphaParameter;
params_basicRL.kappa = kappaParameter;
safe_saveall('params_basicRL.mat', params_basicRL);

%% =================== BAYESIAN AGENT MODEL ==============================
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
nll_bayesianAgent = NaN(numSubjs, 1);

init_params = [5, 0.05]; % [kappa, sigma]
lb = [1, 0];
ub = [100, 0.1];

for n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n));
    rewards = arrayfun(@(h) ...
        subj.rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent(params, subj.mu_hat, subj.dataTable, ...
        length(unique(subj.blocks)), 25, unique(subj.blocks), rewards.');
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    sigmaParameter(n) = params(2);
    kappaParameter(n) = params(1);
    nll_bayesianAgent(n) = nll;
end

params_bayesianAgent.sigma = sigmaParameter;
params_bayesianAgent.kappa = kappaParameter;
safe_saveall('params_bayesianAgent.mat', params_bayesianAgent);

%% =================== PWRL MODEL ========================================
alphaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
nll_PWRL = NaN(numSubjs, 1);

init_params = [0.1, 15, 0.01]; % [alpha, kappa, sigma]
lb = [0, 1, 0];
ub = [1, 100, 0.1];

% data = importdata("preprocessed_dataFitting.mat");
% data = data(data.contrast ~= 0,:);

for n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n));
    nll_fun = @(params) fitSlider_ALLmodels.nll_PWRL(params, subj.mu_hat, subj.blocks, subj.recoded_rewards, subj.condiff, subj.choices, subj.recoded_rewards, subj.contrast);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    alphaParameter(n) = params(1);
    sigmaParameter(n) = params(3);
    kappaParameter(n) = params(2);
    nll_PWRL(n) = nll;
end

params_PWRL.alpha = alphaParameter;
params_PWRL.sigma = sigmaParameter;
params_PWRL.kappa = kappaParameter;
safe_saveall('params_PWRL.mat', params_PWRL);

%% =================== RL + EST SENSITIVITY MODEL ========================
alphaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
nll_RLsigma = NaN(numSubjs, 1);

init_params = [0.1, 5, 0.01]; % [alpha, kappa, sigma]
lb = [0, 1, 0];
ub = [1, 100, 0.1];

for n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n));
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    alphaParameter(n) = params(1);
    sigmaParameter(n) = params(3);
    kappaParameter(n) = params(2);
    nll_RLsigma(n) = nll;
end

params_RLsigma.alpha = alphaParameter;
params_RLsigma.sigma = sigmaParameter;
params_RLsigma.kappa = kappaParameter;
safe_saveall('params_RLsigma.mat', params_RLsigma);

%% =================== COMPUTE AND SAVE AIC/BIC ==========================
disp('Computing AIC and BIC for all models...');

% Number of trials per subject (assuming all subjects have same number)
num_trials = arrayfun(@(n) length(preprocess_fitSlider(data, uniqueID(n)).mu_hat), 1:numSubjs)';

% Model parameter counts
num_params = [2, 3, 3, 2]; % [basicRL, RLsigma, PWRL, BayesianAgent]

% Preallocate
AIC = NaN(numSubjs, 4);
BIC = NaN(numSubjs, 4);

for n = 1:numSubjs
    nlls = [nll_basicRL(n), nll_RLsigma(n), nll_PWRL(n), nll_bayesianAgent(n)];
    [AIC(n,:), BIC(n,:)] = fitSlider_ALLmodels.compute_aic_bic(nlls, num_params, num_trials(n));
end

model_names = {'basicRL','RLsigma','PWRL','BayesianAgent'};
AICBIC_table = array2table([AIC, BIC], 'VariableNames', ...
    {'AIC_basicRL','AIC_RLsigma','AIC_PWRL','AIC_BayesianAgent', ...
     'BIC_basicRL','BIC_RLsigma','BIC_PWRL','BIC_BayesianAgent'});
AICBIC_table.SubjectID = uniqueID;
AICBIC_table = movevars(AICBIC_table, 'SubjectID', 'Before', 1);

% Compute delta BIC for each subject and model
% delta_BIC = BIC - min(BIC,[],2)
delta_BIC = BIC - min(BIC,[],2);

% Add delta_BIC to the table
AICBIC_table.delta_BIC_basicRL = delta_BIC(:,1);
AICBIC_table.delta_BIC_RLsigma = delta_BIC(:,2);
AICBIC_table.delta_BIC_PWRL = delta_BIC(:,3);
AICBIC_table.delta_BIC_BayesianAgent = delta_BIC(:,4);

% Save updated table
safe_saveall('AICBIC_allModels_withDeltaBIC.mat', AICBIC_table);

%% =================== END OF SCRIPT ======================================
disp('All model parameters and AIC/BIC estimated and saved.');
