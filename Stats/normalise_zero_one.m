function norm_data = normalise_zero_one(x,norm_data)

    % function normalise_zero_one normalises any data array between 0 and 1.
    %
    % INPUT:
        % x = data to be normalised
        % norm_data = initialised array to store normalised x
    %
    % OUTPUT:
        % norm_data = normalised data
     
    min_x = min(x); % minimum
    max_x = max(x); % maximum
    denom = max_x - min_x; % difference
    
    for i = 1:length(x) % normalise
        num = x(i) - min_x;
        norm_data(i) = num./denom;
    end
end