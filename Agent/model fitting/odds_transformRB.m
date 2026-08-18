function p_out = odds_transformRB(p, beta)
% ODDS_TRANSFORM - Transform probabilities using odds ratio manipulation
%
% Syntax: p_out = odds_transform(p, beta)
%
% Input:
%   p - input probabilities between 0 and 1 (scalar, vector, or matrix)
%   beta - transformation parameter (optional, default = 0.5)
%          beta = 1: no transformation (identity)
%          beta < 1: compression toward 0.5 (anti-sigmoid effect)
%          beta > 1: expansion away from 0.5 (sigmoid-like effect)
%
% Output:
%   p_out - transformed probabilities between 0 and 1
%
% Formula: p_out = 1 / (1 + ((1-p)/p)^beta)
%
% Properties:
%   - More direct than logit-based transformations
%   - beta < 1 reduces extreme response bias
%   - beta > 1 enhances differences from 0.5
%   - p = 0.5 always maps to p_out = 0.5

    if nargin < 2
        beta = 0.5;  % Default: moderate compression
    end
    
    % % Input validation
    % if any(p <= 0) || any(p >= 1)
    %     error('Input probabilities must be strictly between 0 and 1');
    % end
    
    if beta <= 0
        error('Beta parameter must be positive');
    end
    
    % Apply odds-based transformation
    odds_ratio = (1 - p) ./ p;           % Calculate odds ratio (1-p)/p
    powered_odds = odds_ratio .^ beta;    % Raise to power beta
    p_out = 1 ./ (1 + powered_odds);     % Convert back to probability
end