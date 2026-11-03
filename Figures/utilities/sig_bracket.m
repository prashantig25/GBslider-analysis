function sig_bracket(x1,x2,y,tick_height,label,font_size)
    % function sig_bracket draws a horizontal significance bracket
    % between two x-axis positions (e.g. two bars), with a label (e.g. a
    % p-value from pvals_stars) centered above it. Used for pairwise
    % between-condition comparisons that don't belong above a single bar.
    %
    % INPUTS:
    %   x1,x2: x-axis positions of the two bars being compared
    %   y: y-axis height at which to draw the bracket
    %   tick_height: length of the small vertical ticks at each end of
    %   the bracket
    %   label: text label to display above the bracket
    %   font_size: font size for the label

    plot([x1 x1 x2 x2],[y-tick_height y y y-tick_height],'k','LineWidth',0.5);
    text(mean([x1 x2]),y,label,'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom','FontSize',font_size,'FontWeight','normal');
end
