function add_corr_text(ax,corr_matrix,font_size)
    % function add_corr_text overlays each cell's value as text on top of
    % an imagesc-plotted correlation matrix, choosing white or black text
    % per cell so the number stays legible against that cell's color.
    %
    % INPUTS:
    %   ax: axes handle containing the imagesc-plotted correlation matrix
    %   corr_matrix: the plotted correlation matrix (square)
    %   font_size: font size for the text labels

    cmap = colormap(ax);
    clim_vals = clim(ax);
    for r = 1:size(corr_matrix,1)
        for c = 1:size(corr_matrix,2)
            val = corr_matrix(r,c);

            % MAP THE CELL'S VALUE TO ITS COLORMAP ROW, TO CHECK BRIGHTNESS
            frac = (val - clim_vals(1)) / (clim_vals(2) - clim_vals(1));
            frac = min(max(frac,0),1);
            cmap_idx = round(frac * (size(cmap,1)-1)) + 1;
            brightness = mean(cmap(cmap_idx,:)); % luminance proxy for the cell's color

            if brightness < 0.5
                text_color = 'w';
            else
                text_color = 'k';
            end

            text(ax,c,r,sprintf('%.2f',val),'HorizontalAlignment','center', ...
                'VerticalAlignment','middle','FontSize',font_size,'Color',text_color);
        end
    end
end
