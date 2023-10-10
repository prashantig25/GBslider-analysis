function lg_curves(x,mean_curves,sem_curves,colors_name,legend_names,title_name,xlabelname,ylabelname,fontsize,linewidth,fontname)
    
    % LG_CURVES PLOTS MEAN OF DATA ACROSS PARTICIPANTS, ACROSS TRIALS IN A
    % BLOCK.
    % INPUTS:
        % x = array containing trial numbers in a block for x-axis
        % mean_curves = mean, across subjects, for each trial in a block
        % sem_curves = SEM, across subjects, for each trial in a block
        % colors_name = matrix with color RGB values for all curves
        % legend_names = cell array with legend names for each curve
        % title_name = cell array with title for plot
        % xlabelname = cell array with label name of x-axis
        % ylabelname = cell array with label name of y-axis
        % fontsize = font size of the text
        % linewidth = line width of the plot
        % fontname = font of the text

    figure
    hold on
    
    % PLOT MEAN CURVES
    for i = 1:height(mean_curves)
        plot(x,mean_curves(i,:),"Color",colors_name(i,:))
        hold on
    end
    
    % PLOT SHADED ERROR BARS FOR EACH CURVE
    for i = 1:height(mean_curves)
        shadedErrorBar(x,mean_curves(i,:),sem_curves(i,:),{ 'LineWidth', 2,'Color',colors_name(i,:)},1)
        hold on
    end

    % ADD/EDIT FIGURE PROPERTIES
    legend(legend_names,"AutoUpdate","off",'Location','best','Box','off','Color','none','EdgeColor','none')
    hold on
    box off
    title(title_name)
    xlabel(xlabelname)
    ylabel(ylabelname)
    set(gca,'LineWidth',linewidth)
    set(gca,'fontname',fontname) 
    pl = gca;
    pl.FontSize = fontsize;
end