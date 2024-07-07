function y = nansum(x, dim)

    % function nansum is a replacement for MATLAB's sum which returns a sum
    % of array which has NaNs.
    %
    % INPUTS:
    %   x: array with data to be summed
    %   dim: dimension to operate along
    %
    % OUTPUTS:
    %   y: sum of elements in array x

    x(isnan(x)) = 0; % replace NaNs with 0
    if nargin==1 % use the dimension provided
        y = sum(x);
    else
        y = sum(x,dim);
    end
end 