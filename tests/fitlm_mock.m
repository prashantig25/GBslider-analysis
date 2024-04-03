function lm_mock = fitlm_mock(tbl,varargin)
    %
    % function fitlm_mock is mock version of MATLAB's fitlm.
    %
    % INPUTS:
    % tbl: table to be fit to the linear model.
    % mdl: linear model
    %
    % OUTPUT:
    % lm: mock linear model
    %
    rng(123)
    num_coeffs = 6; 
    coefficients = randn(num_coeffs, 1);
    
    % Construct a mock linear model object
    lm_mock = struct();
    lm_mock.Coefficients = table(coefficients, 'VariableNames', {'Estimate'});
    lm_mock.CoefficientNames = {'Intercept','pe','pe:contrast_diff','pe:congruence','pe:salience','pe:pe_sign_1'}; % Include intercept in coefficient names
    lm_mock.Rsquared.Adjusted = rand(); % Random adjusted Rsquared
    lm_mock.Residuals.Raw = randn(size(tbl, 1), 1); % Random residuals
    
end
