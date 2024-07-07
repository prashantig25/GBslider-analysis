function y = nanmean(x, dim)

    % function nanmean is a replacement for MATLAB's mean which returns an
    % average of all elements in the array which has NaNs.
    %
    % INPUTS:
    %   x: array with data to be averaged
    %   dim: dimension to operate along
    %
    % OUTPUTS:
    %   y: mean of elements in array x

    if nargin<2 % compute mean along the predefined dimensions
        N = sum(~isnan(x));
        y = nansum(x) ./ N;
    else
        N = sum(~isnan(x), dim);
        y = nansum(x, dim) ./ N;
    end
end 