function annotate_textbox(gca,position,string,font_name,font_size, ...
    horz_align,vert_align,bg_color,face_alpha,edge_color,varargin)
    
    % function annotate_textbox adds a textbox and customises it in a figure.
    %
    % INPUTS:
    %   gca: current axes
    %   position: textbox position
    %   string: text string
    %   font_name: font for string
    %   font_size: font size
    %   horz_align: horizontal alignment
    %   vert_align: vertical alignment
    %   bg_color: background color
    %   face_alpha: alpha of bg_color
    %   edge_color: color for box edge
    %   varargin{1}: line width
    %   varargin{2}: line style
    
    text1 = annotation("textbox");
    text1.Parent = gca;
    text1.Position = position;
    text1.String = string;
    text1.FontName = font_name;
    text1.FontSize = font_size;
    text1.HorizontalAlignment = horz_align;
    text1.VerticalAlignment = vert_align;
    text1.BackgroundColor = bg_color;
    text1.FaceAlpha = face_alpha;
    text1.EdgeColor = edge_color;    
    if ~isempty(varargin)
        text1.LineWidth = varargin{1};
        text1.LineStyle = varargin{2};
    end
end