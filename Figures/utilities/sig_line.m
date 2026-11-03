function sig_line(x1,x2,y,label,font_size)
    % function sig_line draws a horizontal line between two x-axis
    % positions (e.g. two bars) with a label (e.g. a p-value from
    % pvals_stars) sitting directly on top of it. Used for pairwise
    % between-condition comparisons that don't belong above a single bar.
    %
    % INPUTS:
    %   x1,x2: x-axis positions of the two bars being compared
    %   y: y-axis height at which to draw the line
    %   label: text label to display on the line
    %   font_size: font size for the label

    plot([x1 x2],[y y],'k','LineWidth',0.5);
    text(mean([x1 x2]),y,label,'HorizontalAlignment','center', ...
        'VerticalAlignment','middle','FontSize',font_size,'FontWeight','normal', ...
        'BackgroundColor','w','Margin',0.5);
end
