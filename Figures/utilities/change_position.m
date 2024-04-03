function new_pos = change_position(axes,position_change)
    
    % function change_position adjusts axes position for a tile, figure,
    % subplot.
    %
    % INPUTS:
    %   axes: given current axes
    %   position_change: dimensions to be adjusted. Should be in the
    %   format of [left, bottom, width, height]
    %
    % OUTPUT:
    %   new_pos: new position for current axes
    
    pos = axes.Position;
    new_pos = pos + position_change;
end