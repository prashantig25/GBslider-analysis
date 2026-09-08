function lm_mock = fitlm_mock(tbl,varargin)
    % function fitlm_mock is mock version of MATLAB's fitlm.
    %
    % INPUTS:
    % tbl: table to be fit to the linear model.
    % mdl: linear model
    %
    % OUTPUT:
    % lm: mock linear model
    %
    % CHANGED: replaced rng(123)+randn/rand with fixed/formula-based values.
    % The seed made every call already reproduce the same numbers, so the
    % randomness was only obscuring what those numbers were -- callers now
    % get the same result without needing to re-seed/re-call this mock.
    num_coeffs = 6;
    coefficients = (1:num_coeffs).';

    % Construct a mock linear model object
    lm_mock = struct();
    lm_mock.Coefficients = table(coefficients, 'VariableNames', {'Estimate'});
    lm_mock.CoefficientNames = {'Intercept','pe','pe:contrast_diff','pe:congruence','pe:salience','pe:pe_sign_1'}; % Include intercept in coefficient names
    lm_mock.Rsquared.Adjusted = 0.5; % CHANGED: fixed value (was rand())
    lm_mock.Residuals.Raw = (1:size(tbl, 1)).'; % CHANGED: fixed sequence (was randn(size(tbl,1),1))

end
