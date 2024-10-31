function equality_check = compare_LinearModels(mdl1, mdl2)

    % function equality_check checks if two MATLAB estimated linear models
    % are equal or not.
    % INPUTS:
    %   mdl1: linear model number 1
    %   mdl2: linear model number 2
    % OUTPUT:
    %   equality_check: if the two models are equal or not
    
    equality_check = isequal(round(mdl1.Coefficients.Estimate,3), ...
        round(mdl2.Coefficients.Estimate,3));
end
