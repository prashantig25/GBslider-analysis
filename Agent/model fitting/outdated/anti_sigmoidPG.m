function y = anti_sigmoidPG(x, strength)
% ANTI_SIGMOID - Transform data with inverse sigmoid curve
%
% Syntax: y = anti_sigmoid(x, strength)
%
% Input:
%   x - input values between 0 and 1 (scalar, vector, or matrix)
%   strength - transformation strength (optional, default = 1)
%              Higher values = more compression toward 0.5
%              strength = 0 gives no transformation (y = x)
%              strength = 1 gives moderate compression
%              strength > 1 gives stronger compression
%
% Output:
%   y - transformed values between 0 and 1
%
% Properties:
%   - Values around 0.5 remain close to 0.5
%   - Values near 0 are overestimated (pushed toward 0.5)
%   - Values near 1 are underestimated (pushed toward 0.5)
%   - Creates compression toward the center

    if nargin < 2
        strength = 1;  % Default strength
    end
    
    % Input validation
    if any(x < 0) || any(x > 1.01)
        error('Input values must be between 0 and 1');
        disp(x)
    end
    
    if strength < 0
        error('Strength parameter must be non-negative');
    end
    
    % Anti-sigmoid transformation
    % Method 1: Using logit and scaled inverse logit
    epsilon = 1e-10;  % Small value to avoid log(0)
    x_safe = max(min(x, 1-epsilon), epsilon);  % Clamp to avoid edge cases
    
    % Transform to logit space, scale down, then back to probability space
    logit_x = log(x_safe ./ (1 - x_safe));
    scaled_logit = logit_x ./ (1+strength);
    y = 1 ./ (1 + exp(-scaled_logit));
end