function bar_plots(y,mean_all,SEM_all,n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,varargin) 
    
    % function bar_plots creates bar plots with multiple bars, single data scatter
    % points, and SEM bars.
    %
    % INPUTS:
    %   y: array with single data points with size n x b where n is
    %   number of data points and b is number of bars.
    %   mean_all: mean across single data points with size n x b where n
    %   is number of groups on x-axis and b is the number of bars.
    %   SEM_all: SEM across single data points with size n x b where n
    %   is number of groups on x-axis and b is the number of bars.
    %   n: number of single data points for each group on x-axis.
    %   x_groups: number of groups to be plotted on x-axis.
    %   bars: number of bars.
    %   legend_names: cell array containing legend for bars.
    %   xticks: array of tick numbers for x-axis
    %   xticklabs: cell array with tick label for x-axis
    %   title_name: cell array containing name of the plot
    %   xlabelname: cell array containing label for x-axis
    %   ylabelname: cell array containing label for y-axis
    %   fontsize: font size for text
    %   linewidth: line width for plot
    %   fontname: font
    %   varargin{1}: face color for bars
    
    % ARRAY CONTAINING X-AXIS GROUP NUMBER FOR EACH DATA POINT
    x = [];
    for i = 1:x_groups
        x = [x; repelem(i,n,1)];
    end

    % CALCULATING MEAN OF Y FOR EACH GROUP ON X-AXIS
    data_plot = [];
    for i = 1:x_groups
        data_plot = [data_plot; nanmean(y(x==i,:))];
    end

    % PLOT BARS
    h = bar(data_plot);
    hold on

    % IF INPUT CONTAINS COLOR ARRAY FOR BARS, CHANGE FACE COLOR, EDGE
    % COLOR, LINE WIDTH, FACE ALPHA
    if nargin > 12
        for b = 1:bars
            h(b).FaceColor = varargin{1}(b,:);
            h(b).EdgeColor = varargin{1}(b,:);
            h(b).LineWidth = 1;
            h(b).FaceAlpha = 0.5;
        end
    end
    hold on
    
    % ADD LEGENDS
    legend(legend_names,"AutoUpdate","off",'Location','best','Box','off', ...
        'Color','none','EdgeColor','none')
    
    % PLOT SINGLE DATA POINTS ON THE BARS, FOR EACH BAR, ACROSS GROUPS ON
    % X-AXIS
    for b = 1:bars
        for i = 1:x_groups
            scatter(repmat(h(b).XEndPoints(i), sum(x==i),1), y(x==i,b),10,"o", ...
                'MarkerEdgeColor','k','MarkerFaceColor','auto','XJitter', ...
                'randn','XJitterWidth',.2)
        end
    end
    
    % CALCULATING THE WIDTH FOR EACH GROUP ON X-AXIS
    ngroups = size(mean_all, 1);
    nbars = size(mean_all, 2); 
    groupwidth = min(0.8, nbars/(nbars + 1.5));

    % PLOTTING ERROR BARS
    for i = 1:nbars
        a = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        errorbar(a, mean_all(:,i),SEM_all(:,i),  'k', 'linestyle', 'none','LineWidth',2);
    end
    hold on

    % ADDING TICKS, LABELS, TITLE
    set(gca,'XTick',xticks)
    set(gca,'XTicklabels',xticklabs)
    box off
    title(title_name)
    xlabel(xlabelname)
    ylabel(ylabelname)
    set(gca,'LineWidth',linewidth)
    set(gca,'fontname',fontname) 
    pl = gca;
    pl.FontSize = fontsize;