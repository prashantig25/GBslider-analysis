function Y = nanstd(varargin)

    % function nanstd is a replacement for MATLAB's std which returns 
    % a standard deviation of elements of array which has NaNs.
    %
    % INPUTS:
    %   varargin: array with data for with standard deviation needs to be
    %   calculated
    %
    % OUTPUTS:
    %   Y: standard deviation of elements in array 

    Y = sqrt(nanvar(varargin{:}));
end 