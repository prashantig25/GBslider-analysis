function plot_boxchart(xdata,ydata,groupbycolor,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,disp_pval,fontsize,linewidth,fontname,nboxs,box_colors,box_width,marker_color ...
    ,markersize,varargin)
    % plot boxcharts with customisations

    if groupbycolor == 0
        h = boxchart(xdata,ydata,'MarkerSize',markersize);
    else
        h = boxchart(xdata,ydata,'GroupByColor',varargin{1},'MarkerSize',markersize);
    end

    for b = 1:nboxs
          h(b).BoxFaceColor = box_colors(b,:);
          h(b).MarkerColor = marker_color;
          h(b).BoxWidth = box_width;
    end

    l = legend(legend_names,"AutoUpdate","off",'Location','best','Box','off','Color', ...
        'none','EdgeColor','none');
    l.ItemTokenSize = [7,7];

    % CUSTOMIZE OTHER PLOT PROPERTIES
    set(gca,'XTick',xticks)
    set(gca,'XTicklabels',xticklabs)
    box off
    pl = gca;
    pl.FontSize = fontsize;  
    set(gca,'LineWidth',linewidth)
    set(gca,'fontname',fontname) 
    title(title_name,'FontWeight','normal')
    xlabel(xlabelname)
    ylabel(ylabelname)

    % ADD BOX LABELS
    xt = get(gca, 'XTick');
    if disp_pval == 1
        labels = varargin{2};
        text(xt, varargin{3},labels, 'HorizontalAlignment','center', ['' ...
            'VerticalAlignment'],'bottom','FontSize',10,'FontWeight','bold')
    end
end