function [betas,rsquared,residuals,coeffs_name,lm] = linear_fit(tbl,mdl,pred_vars,resp_var, ...
    cat_vars,num_vars,weight_y_n,varargin)
    
    % function linear_fit fits a linear regression model to the updates as a
    % function of prediction error and other task based computational
    % variables.
    %
    % INPUT:
    %   tbl: table with predictor vars data
    %   mdl: string with the model formula for fitlm
    %   pred_vars: cell array containing the name of all predictor
    %   variables
    %   resp_var: response variable of the model
    %   cat_vars: cell array contatining the name of all categorical
    %   variable
    %   num_vars: number of predictor variables in the model
    %   weight_y_n: if the function should run a weighted/non-weighted
    %   regression
    %
    % OUTPUT:
    %   betas: array containing beta value for each predictor by fitlm
    %   rsquared: rsquared after fitting mdl to the data
    %   residuals: residuals after fitting mdl to the data
    %   coeffs_name: cell array containing name of all regressors
    %   lm: fitted model
    
    % FIT THE MODEL USING WEIGHTED/NON-WEIGHTED REGRESSION
    if weight_y_n == 1
        lm = fitlm(tbl,mdl,'ResponseVar',resp_var,'PredictorVars',pred_vars, ...
            'CategoricalVars',cat_vars,'Weights',varargin{1});
    else
        lm = fitlm(tbl,mdl,'ResponseVar',resp_var,'PredictorVars',pred_vars, ...
            'CategoricalVars',cat_vars);
    end

    % SAVE R-SQUARED, RESIDUALS AND BETA VALUES
    rsquared = lm.Rsquared.Adjusted;
    residuals = lm.Residuals.Raw;
    betas = nan(1,num_vars+1);
    for b = 1:num_vars+1
        betas(1,b) = lm.Coefficients.Estimate(b);
    end
    coeffs_name = lm.CoefficientNames;
end