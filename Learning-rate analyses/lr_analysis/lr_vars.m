classdef lr_vars < handle
% LR_VARS superclass specifies variables for the model-based analysis of
% learning rates.

    properties
        filename %= 'Data/LR analyses/preprocessed_subj_split1.xlsx'; % filename with pre-processed data
        data % pre-processed data
        mdl % variable to store model definition
        lr_mdl  % run best behavioral model 
        risk_mdl % run model including risk regressor
        saliencechoice_mdl % run model including salience choice regressor
        EEanalysis % if model is to be fit to estimation error
        num_vars % number of predictor variables
        pred_vars % cell array with names of predictor variables
        resp_var = 'up'; % name of response variable
        cat_vars % cell array with names of categorical variables
        res_subjs = []; % empty array to store residuals
        weight_y_n % if non-weighted regression, weight_y_n = 0
        weighted
        num_subjs  % number of participants
        absolute_analysis % pre-process data for absolute LR analysis
        grouped % set to 1 if regression model needs to be fit separately for different groups of trials
        num_groups % number of groups for grouped regression
        agent % fit model to agent simulations
        online % fit model to online dataset
    end
end