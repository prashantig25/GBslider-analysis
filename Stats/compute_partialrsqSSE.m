function [partial_rsq] = compute_partialrsqSSE(SSE_reduced,SSE_full)
    
    % function compute_partialrsq computes partial r-squared value for a single
    % coefficient from a multi-variate model.
    %
    % INPUTS:
    %   rsq_reduced: R-squared of the reduced model (model excluding the term
    %   that needs to be evaluated).
    %   rsq_full: R-squared of the full model
    %
    % OUTPUT:
    %   partial_rsq: partial r-squared
    
    partial_rsq = (SSE_reduced - SSE_full)./SSE_reduced; % source: https://online.stat.psu.edu/stat462/node/138/
end