function adjust_figprops(axes,font_name,font_size,line_width,varargin)
    
    % ADJUST_FIGPROPS adjusts various figure or tile properties.
    
    % INPUTS:
        % axes: current axes
        % font_name: font name
        % font_size: font size
        % line_width: line width
        % varargin{1}: limit for x-axis
        % varargin{2}: limit for y-axis
      
    set(axes,'Color','none','FontName',font_name,'FontSize',font_size)
    set(axes,'LineWidth',line_width)
    if ~isempty(varargin)
        xlim(varargin{1})
        ylim(varargin{2})
    end
end