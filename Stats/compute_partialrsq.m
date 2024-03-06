function [partial_rsq] = compute_partialrsq(rsq_reduced,rsq_full)
    % COMPUTE_PARTIALRSQ computes partial r-squared value for a single
    % coefficient from a multi-variate model.
    % INPUT:
        % rsq_reduced: R-squared of the reduced model (model excluding the term 
        % that needs to be evaluated).
        % rsq_full: R-squared of the full model
    % OUTPUT:
        % partial_rsq: partial r-squared
    partial_rsq = (rsq_full - rsq_reduced)./rsq_full;
end