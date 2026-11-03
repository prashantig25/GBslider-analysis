function pval_str = format_pval(pval)
    % function format_pval formats a p-value into a tex-markup display
    % string: "p < 0.001" if below 0.001, "p < 0.01" if below 0.01,
    % otherwise the exact value rounded to 2 decimal places.
    %
    % INPUTS:
    %   pval: p-value to format
    %
    % OUTPUT:
    %   pval_str: formatted p-value string (tex markup, for use in
    %   title()/text()/xlabel() etc.)

    if pval < 0.001
        pval_str = '\itp\rm < 0.001';
    elseif pval < 0.01
        pval_str = '\itp\rm < 0.01';
    else
        pval_str = strcat("\itp\rm = ",num2str(round(pval,2)));
    end
end
