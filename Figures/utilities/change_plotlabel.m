function [label_x,label_y] = change_plotlabel(axes,adjust_x,adjust_y)
    
    % function change_plotlabel adjusts plot label accordingly.
    %
    % INPUTS:
    %   axes: current axes
    %   adjust_x: dimensions to be adjusted for x-values
    %   adjust_y: dimensions to be adjusted for y-values
    %
    % OUTPUTS:
    %   label_x: adjusted x-position
    %   label_y: adjusted y-position
    
    axes_pos = axes.Position;
    label_x = axes_pos(1) + adjust_x;
    label_y = axes_pos(2) + adjust_y;
end