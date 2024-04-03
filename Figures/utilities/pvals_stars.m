function bar_labels = pvals_stars(p_vals,selected_regressors,bar_labels,stars)
    
    % function pval_stars creates a cell-array with stars, according to p-values.
    %
    % INPUTS:
    %   p_vals: array with p-values
    %   selected_regressors: indices for selected regressors
    %   bar_labels: initialised cell-array for stars
    %   stars: 1 if stars need to be plotted, 0 if p-values need to be
    %   directly plotted
    %
    % OUTPUT:
    %   bar_labels: cell-array with stars
    
    p_vals_regressor = p_vals(:,selected_regressors); % p-values for required regressors
    if stars == 1 % stars to indicate significance
        for i = 1:numel(bar_labels)
            if p_vals_regressor(:,i) < 0.001
                bar_labels{:,i} = "***";
            elseif p_vals_regressor(:,i) < 0.01
                bar_labels{:,i} = "**";
            elseif p_vals_regressor(:,i) < 0.05
                bar_labels{:,i} = "*";
            elseif p_vals_regressor(:,i) > 0.05
                bar_labels{:,i} = "ns";
            end
        end
    else % display p-values to indicate significance
        for i = 1:numel(bar_labels)
            if p_vals_regressor(:,i) < 0.001
                bar_labels{:,i} = "\itp\rm < 0.001";
            elseif p_vals_regressor(:,i) < 0.01
                bar_labels{:,i} = "\itp\rm < 0.01";
            elseif p_vals_regressor(:,i) < 0.05
                bar_labels{:,i} = strcat("\itp\rm = ",num2str(round(p_vals_regressor(:,i),2)));
            elseif p_vals_regressor(:,i) > 0.05
                bar_labels{:,i} = strcat("\itp\rm = ",num2str(round(p_vals_regressor(:,i),2)));
            end
        end
    end
end