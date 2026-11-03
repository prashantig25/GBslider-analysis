%% ========================================================================
%  Script: Fit RL Models to Participant Data and Save Parameters (+AIC/BIC)
%  ------------------------------------------------------------------------
clc; clearvars;
%% =================== LOAD AND PREPARE DATA =============================
pupil = 0; % fit models to the pupil dataset
if pupil == 1
    data = readtable("/Users/prashantig/Brown Dropbox/Prashanti Ganesh/PhD/" + ...
        "Semester 8/pupil_manuscript/Perceptual_unc_aug_task_pupil-main/data/" + ...
        "GB data peak corrected/behavior/model fitting/preprocessed_lr_pupil_no_zerope.xlsx");
    uniqueID = unique(data.id);
    data.ID = data.id;
    for h = 1:height(data)
        if data.congruence(h) == 0
            data.mu_congruence(h) = 1-data.mu(h);
        else
            data.mu_congruence(h) = data.mu(h);
        end
    end
else
    data = importdata("preprocessed_dataFitting.mat");
    uniqueID = unique(data.ID);
    data = data(data.choice_cond ~= 3,:);
end
numSubjs = length(uniqueID);
% Precompute contrast difference
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;
%% =================== BASIC RL MODEL ====================================
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
betaParameter = NaN(numSubjs, 1);
nll_basicRL = NaN(numSubjs, 1);
init_params = [0.1, 5, 0.5]; % [alpha, kappa]
lb = [0, 1, 0.01];
ub = [1, 100, 2];
for n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n), pupil);
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    alphaParameter(n) = params(1);
    kappaParameter(n) = params(2);
    betaParameter(n) = params(3);
    nll_basicRL(n) = nll;

    fprintf('Subject number: %d\n', n);

end
params_basicRL.alpha = alphaParameter;
params_basicRL.kappa = kappaParameter;
params_basicRL.beta = betaParameter;
safe_saveall('params_basicRL_integral_betaPG_bothbiases.mat', params_basicRL);
safe_saveall('nll_basicRL_integral_betaPG_bothbiases.mat', nll_basicRL);
%% =================== BAYESIAN AGENT MODEL ==============================
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
betaParameter = NaN(numSubjs, 1);
nll_bayesianAgent = NaN(numSubjs, 1);
init_params = [5, 0.05, 0.5]; % [kappa, sigma]
lb = [1, 0, 0.01];
ub = [100, 0.1, 2];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n), pupil);
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
    betaParameter(n) = params(3);
    nll_bayesianAgent(n) = nll;

    fprintf('Subject number: %d\n', n);
end
params_bayesianAgent.sigma = sigmaParameter;
params_bayesianAgent.kappa = kappaParameter;
params_bayesianAgent.beta = betaParameter;
safe_saveall('params_bayesianAgent_integrate_evalana_betaPG.mat', params_bayesianAgent);
safe_saveall('nll_bayesianAgent_integrate_evalana_betaPG.mat', nll_bayesianAgent);
%% =================== BAYESIAN AGENT CONFIRMATION BIAS MODEL ============
sigmaParameter_CB = NaN(numSubjs, 1);
kappaParameter_CB = NaN(numSubjs, 1);
confirmBiasParameter_CB = NaN(numSubjs, 1);
betaParameter = NaN(numSubjs, 1);
nll_bayesianAgent_confirmBias = NaN(numSubjs, 1);
init_params = [5, 0.05, 0.75,0.5]; % [kappa, sigma]
lb = [1, 0, 0.5,0.01];
ub = [100, 0.1, 1,2];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n),pupil);
    rewards = arrayfun(@(h) ...
        subj.rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent_confirmBias(params, subj.mu_hat, subj.dataTable, ...
        length(unique(subj.blocks)), 25, unique(subj.blocks), rewards.');
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    sigmaParameter_CB(n) = params(2);
    kappaParameter_CB(n) = params(1);
    confirmBiasParameter_CB(n) = params(3);
    betaParameter(n) = params(4);
    nll_bayesianAgent_confirmBias(n) = nll;

    fprintf('Subject number: %d\n', n);

end
params_bayesianAgent_confirmBias.sigma = sigmaParameter_CB;
params_bayesianAgent_confirmBias.kappa = kappaParameter_CB;
params_bayesianAgent_confirmBias.confirmBias = confirmBiasParameter_CB;
params_bayesianAgent_confirmBias.beta = betaParameter;
safe_saveall('nll_bayesianAgent_confirmBias_freeParam_NochangeSigma_integrate_evalana_betaPG.mat', nll_bayesianAgent_confirmBias);
safe_saveall('params_bayesianAgent_confirmBias_freeParam_NochangeSigma_integrate_evalana_betaPG.mat', params_bayesianAgent_confirmBias);

%% =================== BAYESIAN AGENT CONFIRMATION BIAS MODEL ============
sigmaParameter_CB = NaN(numSubjs, 1);
kappaParameter_CB = NaN(numSubjs, 1);
nll_bayesianAgent_confirmBias = NaN(numSubjs, 1);
init_params = [5, 0.05]; % [kappa, sigma]
lb = [1, 0];
ub = [100, 0.1];
for n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n),pupil);
    rewards = arrayfun(@(h) ...
        subj.rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent_confirmBias(params, subj.mu_hat, subj.dataTable, ...
        length(unique(subj.blocks)), 25, unique(subj.blocks), rewards.');
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    sigmaParameter_CB(n) = params(2);
    kappaParameter_CB(n) = params(1);
    nll_bayesianAgent_confirmBias(n) = nll;
end
params_bayesianAgent_confirmBias.sigma = sigmaParameter_CB;
params_bayesianAgent_confirmBias.kappa = kappaParameter_CB;
safe_saveall('nll_bayesianAgent_confirmBias_fixedParam_NochangeSigma.mat', nll_bayesianAgent_confirmBias);
safe_saveall('params_bayesianAgent_confirmBias_fixedParam_NochangeSigma.mat', params_bayesianAgent_confirmBias);

%% =================== RL + EST SENSITIVITY MODEL ========================
alphaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
betaParameter = NaN(numSubjs, 1);
nll_RLsigma = NaN(numSubjs, 1);
init_params = [0.1, 5, 0.01, 0.5]; % [alpha, kappa, sigma]
lb = [0, 1, 0, 0.01];
ub = [1, 100, 0.1, 2];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n),pupil);
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    alphaParameter(n) = params(1);
    sigmaParameter(n) = params(3);
    kappaParameter(n) = params(2);
    betaParameter(n) = params(4); 
    nll_RLsigma(n) = nll;

    fprintf('Subject number: %d\n', n);
end
params_RLsigma.alpha = alphaParameter;
params_RLsigma.sigma = sigmaParameter;
params_RLsigma.kappa = kappaParameter;
params_RLsigma.beta = betaParameter;
safe_saveall('params_RLsigma_integral_betaPG2_bothbiases.mat', params_RLsigma);
safe_saveall('nll_RLsigma_integral_betaPG2_bothbiases.mat', nll_RLsigma);

%% =================== RL + EST SENSITIVITY + CONFIRMBIAS MODEL ========================
alphaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
confirmBiasParameter = NaN(numSubjs, 1);
noconfirmBiasParameter = NaN(numSubjs, 1);
betaParameter = NaN(numSubjs, 1);
nll_RLsigma = NaN(numSubjs, 1);
init_params = [5, 0.01, 0.1, 0.1, 0.5]; % [alpha, kappa, sigma]
lb = [1, 0, 0, 0, 0.01];
ub = [100, 0.1, 1, 1, 2];
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(data, uniqueID(n),pupil);
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_confirmBias_VOI(params, subj.mu_hat, subj.blocks, rewards, subj.condiff, subj.dataTable.confirm_rew);
    options = optimset('Display', 'off');
    [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    % alphaParameter(n) = params(1);
    sigmaParameter(n) = params(2);
    kappaParameter(n) = params(1);
    confirmBiasParameter(n) = params(3);
    noconfirmBiasParameter(n) = params(4);
    betaParameter(n) = params(5);
    nll_RLsigma(n) = nll;

    fprintf('Subject number: %d\n', n);
end
% params_RLsigma.alpha = alphaParameter;
params_RLsigma.sigma = sigmaParameter;
params_RLsigma.kappa = kappaParameter;
params_RLsigma.confirmBias = confirmBiasParameter;
params_RLsigma.noconfirmBias = noconfirmBiasParameter;
params_RLsigma.beta = betaParameter;
safe_saveall('params_RLsigma_confirmBias_integral_betaPG2_bothbiases.mat', params_RLsigma);
safe_saveall('nll_RLsigma_confirmBias_integral_betaPG2_bothbiases.mat', nll_RLsigma);

%% =================== COMPUTE AND SAVE AIC/BIC ==========================
nll_basicRL = importdata("nll_basicRL_integral_betaPG_bothbiases.mat");
nll_RLsigma = importdata("nll_RLsigma_integral_betaPG2_bothbiases.mat");
nll_RLsigma_confirmBias = importdata("nll_RLsigma_confirmBias_integral_betaPG2_bothbiases.mat");
nll_bayesianAgent = importdata("nll_bayesianAgent_integrate_evalana_betaPG.mat");
nll_bayesianAgent_confirmBias = importdata("nll_bayesianAgent_confirmBias_freeParam_NochangeSigma_integrate_evalana_betaPG.mat");

disp('Computing AIC and BIC for all models...');
% Number of trials per subject (assuming all subjects have same number)
num_trials = repelem(200,numSubjs);%arrayfun(@(n) length(preprocess_fitSlider(data, uniqueID(n)).mu_hat,pupil), 1:numSubjs)';
% Model parameter counts (updated to include confirmation bias model)
num_params = [3, 4, 5, 3, 3]; % [basicRL, RLsigma, PWRL, BayesianAgent, BayesianAgent_confirmBias]
% Preallocate (updated dimensions)
AIC = NaN(numSubjs, 5);
BIC = NaN(numSubjs, 5);
for n = 1:numSubjs
    % nlls = [nll_basicRL(n), nll_RLsigma(n), nll_PWRL(n), nll_bayesianAgent(n), nll_bayesianAgent_confirmBias(n)];
    nlls = [nll_basicRL(n), nll_RLsigma(n), nll_RLsigma_confirmBias(n), nll_bayesianAgent(n), nll_bayesianAgent_confirmBias(n)];
    [AIC(n,:), BIC(n,:)] = fitSlider_ALLmodels.compute_aic_bic(nlls, num_params, num_trials(n));
end
model_names = {'basicRL','RLsigma','RLSigma_confirmBias','BayesianAgent','BayesianAgent_confirmBias'};
AICBIC_table = array2table([AIC, BIC], 'VariableNames', ...
    {'AIC_basicRL','AIC_RLsigma','AIC_RLSigma_confirmBias','AIC_BayesianAgent','AIC_BayesianAgent_confirmBias', ...
     'BIC_basicRL','BIC_RLsigma','BIC_RLSigma_confirmBias','BIC_BayesianAgent','BIC_BayesianAgent_confirmBias'});
AICBIC_table.SubjectID = uniqueID;
AICBIC_table = movevars(AICBIC_table, 'SubjectID', 'Before', 1);
% Compute delta BIC for each subject and model
% delta_BIC = BIC - min(BIC,[],2)
delta_BIC = BIC - min(BIC,[],2);
% Add delta_BIC to the table (updated to include confirmation bias model)
AICBIC_table.delta_BIC_basicRL = delta_BIC(:,1);
AICBIC_table.delta_BIC_RLsigma = delta_BIC(:,2);
AICBIC_table.delta_BIC_RLsigma_confirmBias = delta_BIC(:,3);
AICBIC_table.delta_BIC_BayesianAgent = delta_BIC(:,4);
AICBIC_table.delta_BIC_BayesianAgent_confirmBias = delta_BIC(:,5);
% Save updated table
safe_saveall('AICBIC_allModels_withDeltaBIC_confirmBiasALl_freeParam_NochangeSigma_IntegralAll_evalana_betaPG2_bothbiases.mat', AICBIC_table);
disp('All model parameters and AIC/BIC estimated and saved.');


%% =================== PWRL MODEL ========================================
% alphaParameter = NaN(numSubjs, 1);
% sigmaParameter = NaN(numSubjs, 1);
% kappaParameter = NaN(numSubjs, 1);
% nll_PWRL = NaN(numSubjs, 1);
% init_params = [0.1, 15, 0.01]; % [alpha, kappa, sigma]
% lb = [0, 1, 0];
% ub = [1, 100, 0.1];
% % data = importdata("preprocessed_dataFitting.mat");
% % data = data(data.contrast ~= 0,:);
% for n = 1:numSubjs
%     subj = preprocess_fitSlider(data, uniqueID(n));
%     nll_fun = @(params) fitSlider_ALLmodels.nll_PWRL(params, subj.mu_hat, subj.blocks, subj.recoded_rewards, subj.condiff, subj.choices, subj.recoded_rewards, subj.contrast);
%     options = optimset('Display', 'off');
%     [params, nll] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
%     alphaParameter(n) = params(1);
%     sigmaParameter(n) = params(3);
%     kappaParameter(n) = params(2);
%     nll_PWRL(n) = nll;
% end
% params_PWRL.alpha = alphaParameter;
% params_PWRL.sigma = sigmaParameter;
% params_PWRL.kappa = kappaParameter;
% safe_saveall('params_PWRL.mat', params_PWRL);
