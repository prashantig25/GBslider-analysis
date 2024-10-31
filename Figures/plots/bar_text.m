function bar_text(b,groupOffset,barWidth,fontsize,fontname)
% function bar_text plots text on top of bar.
%
% INPUTS:
%   b: bars
%   groupOffset: offset from xtick depending on number of bar groups
%   barWidth: width of individual bar
%   fontsize: font size

for g = 1:length(b)
    x = b(g).XData + groupOffset(g) * barWidth; % X coordinates for each bar in the current group
    y = b(g).YData; % Y coordinates for each bar in the current group
    data = round(b(g).YData, 1); % Heights of bars in the current group

    % Add text on top of each bar in the current group
    for i = 1:numel(data)
        text(x(i), y(i), num2str(data(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom','FontSize',fontsize,'FontName',fontname);
    end
end
end